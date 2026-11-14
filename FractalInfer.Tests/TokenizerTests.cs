using FractalInfer;

namespace FractalInfer.Tests;

public class TokenizerTests
{
    public static string? ModelDirectory()
    {
        string? env = Environment.GetEnvironmentVariable("FRACTAL_MODEL_DIR");
        var candidates = new List<string>();
        if (!string.IsNullOrEmpty(env))
            candidates.Add(env);
        candidates.Add(Path.Combine(Directory.GetCurrentDirectory(), "models", "Qwen2.5-0.5B-Instruct"));
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
            candidates.Add(Path.Combine(dir.FullName, "models", "Qwen2.5-0.5B-Instruct"));
        foreach (string candidate in candidates)
        {
            if (File.Exists(Path.Combine(candidate, "tokenizer.json")))
                return candidate;
        }
        return null;
    }

    [Fact]
    public void Encode_MatchesHuggingFaceGoldens()
    {
        string? dir = ModelDirectory();
        if (dir is null)
        {
            if (Environment.GetEnvironmentVariable("CI") == "true")
                throw new FileNotFoundException("CI requires models/Qwen2.5-0.5B-Instruct/tokenizer.json. Run scripts/download-qwen25-0.5b.sh.");
            Console.WriteLine("SKIP tokenizer: tokenizer.json not downloaded.");
            return;
        }

        var tokenizer = BpeTokenizer.Load(Path.Combine(dir, "tokenizer.json"));
        AssertIds(tokenizer, "The capital of France is", [785, 6722, 315, 9625, 374]);
        AssertIds(tokenizer, "Hello, world!", [9707, 11, 1879, 0]);
        AssertIds(tokenizer, "It's a test.\nLine two.", [2132, 594, 264, 1273, 624, 2460, 1378, 13]);
        AssertIds(tokenizer, "12345", [16, 17, 18, 19, 20]);
        AssertIds(tokenizer, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", [151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198]);
        AssertIds(tokenizer, "  leading spaces", [220, 6388, 12621]);
        AssertIds(tokenizer, "use <tool_call> now", [810, 220, 151657, 1431]);
        AssertIds(tokenizer, "caf\u00e9", [924, 58858]);
        AssertIds(tokenizer, "\u4f60\u597d", [108386]);
        AssertIds(tokenizer, "\U0001F642", [145080]);
        string chat = BpeTokenizer.ApplyQwen25Instruct("The capital of France is");
        AssertIds(tokenizer, chat, [151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 785, 6722, 315, 9625, 374, 151645, 198, 151644, 77091, 198]);

        Assert.Equal("The capital of France is", tokenizer.Decode(tokenizer.Encode("The capital of France is")));
        Assert.Equal("Hello, world!", tokenizer.Decode(tokenizer.Encode("Hello, world!")));
        Assert.Equal("\u4f60\u597d", tokenizer.Decode(tokenizer.Encode("\u4f60\u597d")));
        Assert.Equal("<|im_start|>", tokenizer.Decode(new[] { 151644 }));
        Console.WriteLine("tokenizer goldens matched");
    }

    private static void AssertIds(BpeTokenizer tokenizer, string text, int[] expected)
    {
        int[] actual = tokenizer.Encode(text);
        Assert.True(expected.AsSpan().SequenceEqual(actual), $"'{text}' encoded to [{string.Join(",", actual)}], expected [{string.Join(",", expected)}].");
    }
}
