using System.Text.Json;

namespace FractalInfer;

public sealed class Qwen2Config
{
    public required int HiddenSize { get; init; }
    public required int IntermediateSize { get; init; }
    public required int NumHiddenLayers { get; init; }
    public required int NumAttentionHeads { get; init; }
    public required int NumKeyValueHeads { get; init; }
    public required int HeadDim { get; init; }
    public required int VocabSize { get; init; }
    public required float RmsNormEps { get; init; }
    public required float RopeTheta { get; init; }
    public required bool TieWordEmbeddings { get; init; }
    public required int EosTokenId { get; init; }
    public required int[] StopTokenIds { get; init; }

    public int QueryDim => NumAttentionHeads * HeadDim;
    public int KeyValueDim => NumKeyValueHeads * HeadDim;
    public int HeadGroupSize => NumAttentionHeads / NumKeyValueHeads;

    public static Qwen2Config Load(string configPath, string? generationConfigPath = null)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(configPath));
        JsonElement root = doc.RootElement;
        string modelType = root.GetProperty("model_type").GetString() ?? "";
        if (modelType != "qwen2")
            throw new NotSupportedException($"Only model_type 'qwen2' is implemented (found '{modelType}').");

        string act = root.GetProperty("hidden_act").GetString() ?? "";
        if (act != "silu")
            throw new NotSupportedException($"hidden_act '{act}' is not implemented.");
        if (root.TryGetProperty("use_sliding_window", out var sliding) && sliding.ValueKind == JsonValueKind.True)
            throw new NotSupportedException("use_sliding_window=true is not implemented.");

        int hidden = root.GetProperty("hidden_size").GetInt32();
        int heads = root.GetProperty("num_attention_heads").GetInt32();
        int kvHeads = root.GetProperty("num_key_value_heads").GetInt32();
        int headDim = root.TryGetProperty("head_dim", out var headDimEl) && headDimEl.ValueKind == JsonValueKind.Number
            ? headDimEl.GetInt32()
            : hidden / heads;
        if (hidden <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads % kvHeads != 0)
            throw new InvalidDataException("Qwen2 attention dimensions are invalid.");
        if (headDim % 2 != 0)
            throw new InvalidDataException("RoPE requires an even head dimension.");

        int eos = root.TryGetProperty("eos_token_id", out var eosEl) && eosEl.ValueKind == JsonValueKind.Number
            ? eosEl.GetInt32()
            : -1;
        int[] stops = eos >= 0 ? [eos] : [];
        if (generationConfigPath != null && File.Exists(generationConfigPath))
            stops = ReadStopIds(generationConfigPath, stops);

        return new Qwen2Config
        {
            HiddenSize = hidden,
            IntermediateSize = root.GetProperty("intermediate_size").GetInt32(),
            NumHiddenLayers = root.GetProperty("num_hidden_layers").GetInt32(),
            NumAttentionHeads = heads,
            NumKeyValueHeads = kvHeads,
            HeadDim = headDim,
            VocabSize = root.GetProperty("vocab_size").GetInt32(),
            RmsNormEps = (float)root.GetProperty("rms_norm_eps").GetDouble(),
            RopeTheta = (float)root.GetProperty("rope_theta").GetDouble(),
            TieWordEmbeddings = root.TryGetProperty("tie_word_embeddings", out var tie) && tie.ValueKind == JsonValueKind.True,
            EosTokenId = eos,
            StopTokenIds = stops,
        };
    }

    private static int[] ReadStopIds(string path, int[] fallback)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        if (!doc.RootElement.TryGetProperty("eos_token_id", out var eos))
            return fallback;
        if (eos.ValueKind == JsonValueKind.Number)
            return [eos.GetInt32()];
        if (eos.ValueKind == JsonValueKind.Array)
        {
            var ids = new List<int>();
            foreach (var item in eos.EnumerateArray())
                ids.Add(item.GetInt32());
            return ids.Count > 0 ? ids.ToArray() : fallback;
        }
        return fallback;
    }
}
