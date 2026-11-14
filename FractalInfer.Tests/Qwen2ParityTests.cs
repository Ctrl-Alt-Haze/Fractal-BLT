using System.Diagnostics;
using System.Text.Json;
using FractalInfer;

namespace FractalInfer.Tests;

public class Qwen2ParityTests
{
    private const string Prompt = "The capital of France is";
    private const int NewTokens = 32;

    [Fact]
    public void PrefillLogits_And_Greedy32_MatchHuggingFaceFloat32()
    {
        string? dir = TokenizerTests.ModelDirectory();
        bool weights = dir != null && File.Exists(Path.Combine(dir, "model.safetensors"));
        bool required = Environment.GetEnvironmentVariable("FRACTAL_REQUIRE_HF_PARITY") == "1";
        if (!weights)
        {
            if (required)
                throw new FileNotFoundException("FRACTAL_REQUIRE_HF_PARITY=1 but model.safetensors is missing. Run scripts/download-qwen25-0.5b.sh.");
            Console.WriteLine("SKIP HF parity: model.safetensors not downloaded.");
            return;
        }

        string referenceDir = Environment.GetEnvironmentVariable("FRACTAL_REFERENCE_DIR")
            ?? Path.Combine(Path.GetTempPath(), "fractal-hf-ref");
        bool reuse = Environment.GetEnvironmentVariable("FRACTAL_REUSE_REFERENCE") == "1"
            && File.Exists(Path.Combine(referenceDir, "logits.f32"))
            && File.Exists(Path.Combine(referenceDir, "meta.json"));
        if (!reuse)
            RunReference(dir!, referenceDir);

        using var metaDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDir, "meta.json")));
        JsonElement meta = metaDoc.RootElement;
        Assert.Equal(Prompt, meta.GetProperty("prompt").GetString());
        int[] inputIds = ReadInts(meta.GetProperty("input_ids"));
        int[] greedy = ReadInts(meta.GetProperty("greedy_ids"));
        Assert.Equal(NewTokens, greedy.Length);
        int vocab = meta.GetProperty("vocab_size").GetInt32();
        int probeToken = meta.GetProperty("embed_token").GetInt32();
        float[] probe = ReadFloats(meta.GetProperty("embed_probe"));

        var tokenizer = BpeTokenizer.Load(Path.Combine(dir!, "tokenizer.json"));
        Assert.Equal(inputIds, tokenizer.Encode(Prompt));

        var model = Qwen2Model.Load(dir!, maxContext: 256);
        Assert.Equal(vocab, model.VocabSize);
        float[] embed = new float[model.Config.HiddenSize];
        model.CopyTokenEmbedding(probeToken, embed);
        for (int i = 0; i < probe.Length; i++)
            Assert.Equal(probe[i], embed[i]);

        byte[] raw = File.ReadAllBytes(Path.Combine(referenceDir, "logits.f32"));
        Assert.Equal(vocab * sizeof(float), raw.Length);
        float[] hf = new float[vocab];
        Buffer.BlockCopy(raw, 0, hf, 0, raw.Length);

        model.Forward(inputIds);
        float maxAbs = 0f;
        int at = 0;
        for (int i = 0; i < vocab; i++)
        {
            float diff = MathF.Abs(model.Logits[i] - hf[i]);
            if (diff > maxAbs)
            {
                maxAbs = diff;
                at = i;
            }
        }

        Console.WriteLine(
            $"HF parity max abs logit diff: {maxAbs:G9} at {at} csharp={model.Logits[at]:G9} hf={hf[at]:G9}");
        Assert.True(maxAbs < 1e-2f, $"max abs logit diff {maxAbs:G9} is not < 1e-2 (index {at}).");

        var got = new int[NewTokens];
        got[0] = TensorMath.ArgMax(model.Logits);
        for (int i = 1; i < NewTokens; i++)
        {
            model.Forward(new[] { got[i - 1] });
            got[i] = TensorMath.ArgMax(model.Logits);
        }

        int matched = 0;
        for (; matched < NewTokens && got[matched] == greedy[matched]; matched++)
        {
        }
        Console.WriteLine($"HF parity greedy tokens matched: {matched}");
        string detail = matched >= NewTokens
            ? "ok"
            : $"Greedy diverged at step {matched}. C#={got[matched]} HF={greedy[matched]}.";
        Assert.True(matched >= NewTokens, detail);
        Console.WriteLine("greedy text: " + tokenizer.Decode(got));
    }

    private static void RunReference(string modelDir, string referenceDir)
    {
        string script = FindScript();
        Directory.CreateDirectory(referenceDir);
        var start = new ProcessStartInfo
        {
            FileName = "python3",
            Arguments = $"\"{script}\" --model \"{modelDir}\" --out \"{referenceDir}\" --prompt \"{Prompt}\" --max-new-tokens {NewTokens}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using var process = Process.Start(start) ?? throw new InvalidOperationException("Failed to start python3.");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"Reference script failed ({process.ExitCode}).\n{stdout}\n{stderr}");
        Console.WriteLine(stdout);
    }

    private static string FindScript()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "tools", "dump_hf_reference.py");
            if (File.Exists(candidate))
                return candidate;
        }
        dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "tools", "dump_hf_reference.py");
            if (File.Exists(candidate))
                return candidate;
        }
        throw new FileNotFoundException("tools/dump_hf_reference.py was not found.");
    }

    private static int[] ReadInts(JsonElement array)
    {
        var values = new int[array.GetArrayLength()];
        int i = 0;
        foreach (var item in array.EnumerateArray())
            values[i++] = item.GetInt32();
        return values;
    }

    private static float[] ReadFloats(JsonElement array)
    {
        var values = new float[array.GetArrayLength()];
        int i = 0;
        foreach (var item in array.EnumerateArray())
            values[i++] = (float)item.GetDouble();
        return values;
    }
}
