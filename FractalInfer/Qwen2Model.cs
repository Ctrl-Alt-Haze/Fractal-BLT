namespace FractalInfer;

/// <summary>
/// Float32 Qwen2 decoder: embedding, RMSNorm, NeoX RoPE, grouped-query attention
/// with a KV cache, SwiGLU, and a tied or separate LM head. One model instance is
/// not safe for concurrent Forward calls.
/// </summary>
public sealed class Qwen2Model
{
    private readonly Qwen2Config _config;
    private readonly float[] _embed;
    private readonly float[] _lmHead;
    private readonly float[] _finalNorm;
    private readonly Layer[] _layers;
    private readonly float[] _invFreq;
    private readonly float[] _keyCache;
    private readonly float[] _valueCache;
    private readonly float[] _hidden;
    private readonly float[] _normed;
    private readonly float[] _q;
    private readonly float[] _k;
    private readonly float[] _v;
    private readonly float[] _attn;
    private readonly float[] _projected;
    private readonly float[] _gate;
    private readonly float[] _up;
    private readonly float[] _scores;
    private readonly float[] _logits;
    private readonly float[] _sampleScratch;
    private readonly bool _trace;

    private Qwen2Model(Qwen2Config config, float[] embed, float[] lmHead, float[] finalNorm, Layer[] layers, int maxContext)
    {
        _config = config;
        _embed = embed;
        _lmHead = lmHead;
        _finalNorm = finalNorm;
        _layers = layers;
        MaxContext = maxContext;
        _trace = Environment.GetEnvironmentVariable("FRACTAL_TRACE") == "1";

        int half = config.HeadDim / 2;
        _invFreq = new float[half];
        for (int i = 0; i < half; i++)
        {
            float exponent = (2f * i) / config.HeadDim;
            _invFreq[i] = 1f / MathF.Pow(config.RopeTheta, exponent);
        }

        int kv = config.NumHiddenLayers * config.NumKeyValueHeads * maxContext * config.HeadDim;
        _keyCache = new float[kv];
        _valueCache = new float[kv];
        _hidden = new float[maxContext * config.HiddenSize];
        _normed = new float[maxContext * config.HiddenSize];
        _q = new float[maxContext * config.QueryDim];
        _k = new float[maxContext * config.KeyValueDim];
        _v = new float[maxContext * config.KeyValueDim];
        _attn = new float[maxContext * config.QueryDim];
        _projected = new float[maxContext * config.HiddenSize];
        _gate = new float[maxContext * config.IntermediateSize];
        _up = new float[maxContext * config.IntermediateSize];
        _scores = new float[maxContext];
        _logits = new float[config.VocabSize];
        _sampleScratch = new float[config.VocabSize];
    }

    public Qwen2Config Config => _config;
    public int VocabSize => _config.VocabSize;
    public int MaxContext { get; }
    public int CacheLength { get; private set; }
    public ReadOnlySpan<float> Logits => _logits;

    public static Qwen2Model Load(string modelDirectory, int maxContext = 2048, Action<string>? log = null)
    {
        if (maxContext <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxContext));
        string configPath = Path.Combine(modelDirectory, "config.json");
        string weightsPath = Path.Combine(modelDirectory, "model.safetensors");
        string generationPath = Path.Combine(modelDirectory, "generation_config.json");
        if (!File.Exists(configPath) || !File.Exists(weightsPath))
            throw new FileNotFoundException($"Expected config.json and model.safetensors in {modelDirectory}.");

        Qwen2Config config = Qwen2Config.Load(configPath, File.Exists(generationPath) ? generationPath : null);
        log?.Invoke($"Loading {config.NumHiddenLayers}-layer Qwen2 ({config.HiddenSize} hidden, vocab {config.VocabSize}) as float32.");
        var file = SafetensorsFile.Open(weightsPath);

        float[] embed = file.ReadFloat32("model.embed_tokens.weight");
        Expect(embed, "model.embed_tokens.weight", config.VocabSize, config.HiddenSize);
        float[] lmHead;
        if (file.Names.Contains("lm_head.weight"))
        {
            lmHead = file.ReadFloat32("lm_head.weight");
            Expect(lmHead, "lm_head.weight", config.VocabSize, config.HiddenSize);
        }
        else if (config.TieWordEmbeddings)
        {
            lmHead = embed;
        }
        else
        {
            throw new InvalidDataException("lm_head.weight is missing and tie_word_embeddings is false.");
        }

        float[] finalNorm = file.ReadFloat32("model.norm.weight");
        Expect(finalNorm, "model.norm.weight", config.HiddenSize);
        var layers = new Layer[config.NumHiddenLayers];
        for (int i = 0; i < layers.Length; i++)
        {
            string prefix = $"model.layers.{i}";
            var layer = new Layer
            {
                InputNorm = file.ReadFloat32($"{prefix}.input_layernorm.weight"),
                PostNorm = file.ReadFloat32($"{prefix}.post_attention_layernorm.weight"),
                Wq = file.ReadFloat32($"{prefix}.self_attn.q_proj.weight"),
                Wk = file.ReadFloat32($"{prefix}.self_attn.k_proj.weight"),
                Wv = file.ReadFloat32($"{prefix}.self_attn.v_proj.weight"),
                Wo = file.ReadFloat32($"{prefix}.self_attn.o_proj.weight"),
                Bq = file.ReadFloat32($"{prefix}.self_attn.q_proj.bias"),
                Bk = file.ReadFloat32($"{prefix}.self_attn.k_proj.bias"),
                Bv = file.ReadFloat32($"{prefix}.self_attn.v_proj.bias"),
                Gate = file.ReadFloat32($"{prefix}.mlp.gate_proj.weight"),
                Up = file.ReadFloat32($"{prefix}.mlp.up_proj.weight"),
                Down = file.ReadFloat32($"{prefix}.mlp.down_proj.weight"),
            };
            Expect(layer.InputNorm, "input_layernorm", config.HiddenSize);
            Expect(layer.PostNorm, "post_attention_layernorm", config.HiddenSize);
            Expect(layer.Wq, "q_proj", config.QueryDim, config.HiddenSize);
            Expect(layer.Wk, "k_proj", config.KeyValueDim, config.HiddenSize);
            Expect(layer.Wv, "v_proj", config.KeyValueDim, config.HiddenSize);
            Expect(layer.Wo, "o_proj", config.HiddenSize, config.QueryDim);
            Expect(layer.Bq, "q_bias", config.QueryDim);
            Expect(layer.Bk, "k_bias", config.KeyValueDim);
            Expect(layer.Bv, "v_bias", config.KeyValueDim);
            Expect(layer.Gate, "gate_proj", config.IntermediateSize, config.HiddenSize);
            Expect(layer.Up, "up_proj", config.IntermediateSize, config.HiddenSize);
            Expect(layer.Down, "down_proj", config.HiddenSize, config.IntermediateSize);
            layers[i] = layer;
            if ((i + 1) % 8 == 0)
                log?.Invoke($"  loaded layers 0..{i}");
        }

        log?.Invoke("Weights resident.");
        return new Qwen2Model(config, embed, lmHead, finalNorm, layers, maxContext);
    }

    public void Reset() => CacheLength = 0;

    public void CopyTokenEmbedding(int tokenId, Span<float> destination)
    {
        if ((uint)tokenId >= (uint)_config.VocabSize)
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        if (destination.Length < _config.HiddenSize)
            throw new ArgumentException("Destination is shorter than the hidden size.", nameof(destination));
        _embed.AsSpan(tokenId * _config.HiddenSize, _config.HiddenSize).CopyTo(destination);
    }

    public void Forward(ReadOnlySpan<int> tokenIds)
    {
        int seq = tokenIds.Length;
        if (seq == 0)
            throw new ArgumentException("Forward requires at least one token.", nameof(tokenIds));
        if (CacheLength + seq > MaxContext)
            throw new InvalidOperationException($"Context length {MaxContext} exceeded ({CacheLength} cached + {seq} new).");

        int hidden = _config.HiddenSize;
        for (int t = 0; t < seq; t++)
        {
            int token = tokenIds[t];
            if ((uint)token >= (uint)_config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token {token} is outside vocab {_config.VocabSize}.");
            _embed.AsSpan(token * hidden, hidden).CopyTo(_hidden.AsSpan(t * hidden, hidden));
        }

        int qDim = _config.QueryDim;
        int kvDim = _config.KeyValueDim;
        int inter = _config.IntermediateSize;
        int position0 = CacheLength;
        for (int layerIndex = 0; layerIndex < _layers.Length; layerIndex++)
        {
            Layer layer = _layers[layerIndex];
            unsafe
            {
                fixed (float* hiddenPtr = _hidden, normedPtr = _normed, weight = layer.InputNorm)
                {
                    TensorMath.RmsNormRows(hiddenPtr, weight, normedPtr, seq, hidden, _config.RmsNormEps);
                }
            }

            TensorMath.Gemm(_normed, layer.Wq, _q, seq, hidden, qDim, layer.Bq);
            TensorMath.Gemm(_normed, layer.Wk, _k, seq, hidden, kvDim, layer.Bk);
            TensorMath.Gemm(_normed, layer.Wv, _v, seq, hidden, kvDim, layer.Bv);
            ApplyRope(_q, seq, _config.NumAttentionHeads, position0);
            ApplyRope(_k, seq, _config.NumKeyValueHeads, position0);
            WriteCache(layerIndex, seq, position0);
            Attention(layerIndex, seq, position0);
            TensorMath.Gemm(_attn, layer.Wo, _projected, seq, qDim, hidden, null);
            unsafe
            {
                fixed (float* dst = _hidden, src = _projected)
                {
                    TensorMath.AddInPlace(dst, src, seq * hidden);
                }
                fixed (float* hiddenPtr = _hidden, normedPtr = _normed, weight = layer.PostNorm)
                {
                    TensorMath.RmsNormRows(hiddenPtr, weight, normedPtr, seq, hidden, _config.RmsNormEps);
                }
            }

            TensorMath.Gemm(_normed, layer.Gate, _gate, seq, hidden, inter, null);
            TensorMath.Gemm(_normed, layer.Up, _up, seq, hidden, inter, null);
            unsafe
            {
                fixed (float* gate = _gate, up = _up)
                {
                    TensorMath.SwiGluInPlace(gate, up, seq * inter);
                }
            }
            TensorMath.Gemm(_gate, layer.Down, _projected, seq, inter, hidden, null);
            unsafe
            {
                fixed (float* dst = _hidden, src = _projected)
                {
                    TensorMath.AddInPlace(dst, src, seq * hidden);
                }
            }

            if (_trace)
                TraceLayer(layerIndex, seq);
        }

        unsafe
        {
            fixed (float* hiddenPtr = _hidden, normedPtr = _normed, weight = _finalNorm)
            {
                TensorMath.RmsNormRows(hiddenPtr, weight, normedPtr, seq, hidden, _config.RmsNormEps);
            }
        }

        // The LM head only needs the last position. Park that row in _projected (free after the residual add).
        if (seq == 1)
            _normed.AsSpan(0, hidden).CopyTo(_projected.AsSpan(0, hidden));
        else
            _normed.AsSpan((seq - 1) * hidden, hidden).CopyTo(_projected.AsSpan(0, hidden));
        TensorMath.Gemm(_projected, _lmHead, _logits, 1, hidden, _config.VocabSize, null);
        CacheLength = position0 + seq;
    }

    public int NextToken(ReadOnlySpan<int> tokenIds, float temperature, Random? random)
    {
        Forward(tokenIds);
        if (temperature <= 0f)
            return TensorMath.ArgMax(_logits);
        if (random is null)
            throw new ArgumentNullException(nameof(random), "Temperature sampling requires a Random.");
        return TensorMath.SampleTemperature(_logits, temperature, random, _sampleScratch);
    }

    private void ApplyRope(float[] states, int seq, int heads, int position0)
    {
        int dim = _config.HeadDim;
        for (int s = 0; s < seq; s++)
        {
            int position = position0 + s;
            for (int h = 0; h < heads; h++)
            {
                int offset = (s * heads + h) * dim;
                TensorMath.ApplyNeoXRope(states.AsSpan(offset, dim), position, _invFreq);
            }
        }
    }

    private void WriteCache(int layerIndex, int seq, int position0)
    {
        int heads = _config.NumKeyValueHeads;
        int dim = _config.HeadDim;
        int layerStride = heads * MaxContext * dim;
        for (int s = 0; s < seq; s++)
        {
            int pos = position0 + s;
            for (int h = 0; h < heads; h++)
            {
                int src = (s * heads + h) * dim;
                int dst = layerIndex * layerStride + (h * MaxContext + pos) * dim;
                _k.AsSpan(src, dim).CopyTo(_keyCache.AsSpan(dst, dim));
                _v.AsSpan(src, dim).CopyTo(_valueCache.AsSpan(dst, dim));
            }
        }
    }

    private unsafe void Attention(int layerIndex, int seq, int position0)
    {
        int heads = _config.NumAttentionHeads;
        int kvHeads = _config.NumKeyValueHeads;
        int group = _config.HeadGroupSize;
        int dim = _config.HeadDim;
        float scale = 1f / MathF.Sqrt(dim);
        int layerStride = kvHeads * MaxContext * dim;
        fixed (float* q = _q, attn = _attn, keyCache = _keyCache, valueCache = _valueCache, scores = _scores)
        {
            float* keyLayer = keyCache + layerIndex * layerStride;
            float* valueLayer = valueCache + layerIndex * layerStride;
            for (int s = 0; s < seq; s++)
            {
                int length = position0 + s + 1;
                for (int h = 0; h < heads; h++)
                {
                    int kvHead = h / group;
                    float* query = q + (s * heads + h) * dim;
                    float* keys = keyLayer + kvHead * MaxContext * dim;
                    float* values = valueLayer + kvHead * MaxContext * dim;
                    float max = float.NegativeInfinity;
                    for (int t = 0; t < length; t++)
                    {
                        float score = TensorMath.Dot(query, keys + t * dim, dim) * scale;
                        scores[t] = score;
                        if (score > max)
                            max = score;
                    }

                    float sum = 0f;
                    for (int t = 0; t < length; t++)
                    {
                        float e = MathF.Exp(scores[t] - max);
                        scores[t] = e;
                        sum += e;
                    }
                    float inv = 1f / sum;
                    float* output = attn + (s * heads + h) * dim;
                    for (int d = 0; d < dim; d++)
                        output[d] = 0f;
                    for (int t = 0; t < length; t++)
                    {
                        float p = scores[t] * inv;
                        float* value = values + t * dim;
                        for (int d = 0; d < dim; d++)
                            output[d] += p * value[d];
                    }
                }
            }
        }
    }

    private void TraceLayer(int layerIndex, int seq)
    {
        double sum = 0d;
        int hidden = _config.HiddenSize;
        int offset = (seq - 1) * hidden;
        for (int i = 0; i < hidden; i++)
            sum += _hidden[offset + i];
        Console.Error.WriteLine($"trace layer {layerIndex} last-token sum {sum:G17}");
    }

    private static void Expect(float[] values, string name, int rows, int cols = 1)
    {
        long expected = (long)rows * cols;
        if (values.LongLength != expected)
            throw new InvalidDataException($"{name} has {values.LongLength} values, expected {rows}x{cols} ({expected}).");
    }

    private sealed class Layer
    {
        public required float[] InputNorm { get; init; }
        public required float[] PostNorm { get; init; }
        public required float[] Wq { get; init; }
        public required float[] Wk { get; init; }
        public required float[] Wv { get; init; }
        public required float[] Wo { get; init; }
        public required float[] Bq { get; init; }
        public required float[] Bk { get; init; }
        public required float[] Bv { get; init; }
        public required float[] Gate { get; init; }
        public required float[] Up { get; init; }
        public required float[] Down { get; init; }
    }
}
