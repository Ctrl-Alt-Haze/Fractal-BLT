using System.Diagnostics;
using System.Globalization;
using FractalInfer;

namespace FractalInfer.Cli;

/// <summary>
/// CPU Qwen2.5 inference. This is the real text path. FractalServe's HTTP chat route is a labeled stub.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintUsage();
            return args.Length == 0 ? 1 : 0;
        }

        string command = args[0];
        if (command is not ("generate" or "bench"))
        {
            Console.Error.WriteLine($"Unknown command '{command}'.");
            PrintUsage();
            return 1;
        }

        var options = Parse(args.AsSpan(1));
        if (!options.TryGetValue("model", out string? modelDir))
        {
            Console.Error.WriteLine("Missing --model <directory with config.json, tokenizer.json, and model.safetensors>.");
            return 1;
        }

        string prompt = options.TryGetValue("prompt", out string? given)
            ? given
            : command == "bench"
                ? "The quick brown fox jumps over the lazy dog. "
                : "";
        if (prompt.Length == 0)
        {
            Console.Error.WriteLine("Missing --prompt.");
            return 1;
        }

        int maxNew = GetInt(options, "max-new-tokens", command == "bench" ? 32 : 64);
        int context = GetInt(options, "context", 2048);
        float temperature = GetFloat(options, "temperature", 0f);
        int seed = GetInt(options, "seed", 1);
        bool raw = options.ContainsKey("raw");
        bool noEos = options.ContainsKey("no-eos") || command == "bench";
        int prefillTarget = GetInt(options, "prefill-tokens", command == "bench" ? 128 : 0);

        var tokenizer = BpeTokenizer.Load(Path.Combine(modelDir, "tokenizer.json"));
        var model = Qwen2Model.Load(modelDir, context, Console.Error.WriteLine);
        string text = raw ? prompt : BpeTokenizer.ApplyQwen25Instruct(prompt, options.GetValueOrDefault("system"));
        int[] promptIds = tokenizer.Encode(text);
        if (command == "bench" && prefillTarget > 0 && promptIds.Length < prefillTarget)
        {
            var repeated = new List<int>(prefillTarget);
            while (repeated.Count < prefillTarget)
                repeated.AddRange(promptIds);
            promptIds = repeated.Take(prefillTarget).ToArray();
            text = tokenizer.Decode(promptIds);
        }

        if (promptIds.Length == 0)
        {
            Console.Error.WriteLine("Prompt tokenized to zero tokens.");
            return 1;
        }
        if (promptIds.Length >= model.MaxContext)
        {
            Console.Error.WriteLine($"Prompt is {promptIds.Length} tokens and context is {model.MaxContext}.");
            return 1;
        }

        var rng = temperature > 0f ? new Random(seed) : null;
        var stop = new HashSet<int>(model.Config.StopTokenIds);
        if (command == "bench")
            return Bench(model, tokenizer, promptIds, maxNew);
        return Generate(model, tokenizer, promptIds, maxNew, temperature, rng, stop, noEos);
    }

    private static int Generate(Qwen2Model model, BpeTokenizer tokenizer, int[] promptIds, int maxNew, float temperature, Random? rng, HashSet<int> stop, bool noEos)
    {
        var produced = new List<int>(maxNew);
        var clock = Stopwatch.StartNew();
        int first = model.NextToken(promptIds, temperature, rng);
        clock.Stop();
        double prefillSeconds = clock.Elapsed.TotalSeconds;
        produced.Add(first);
        clock.Restart();
        while (produced.Count < maxNew)
        {
            if (!noEos && stop.Contains(produced[^1]))
                break;
            int next = model.NextToken(new[] { produced[^1] }, temperature, rng);
            if (!noEos && stop.Contains(next))
            {
                produced.Add(next);
                break;
            }
            produced.Add(next);
        }
        clock.Stop();
        int decodeForwards = Math.Max(0, produced.Count - 1);
        double decodeSeconds = clock.Elapsed.TotalSeconds;

        Console.WriteLine(tokenizer.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(produced)));
        Console.Error.WriteLine(
            $"prefill_tokens={promptIds.Length} prefill_s={prefillSeconds.ToString("F4", CultureInfo.InvariantCulture)} prefill_tok_s={Rate(promptIds.Length, prefillSeconds)}");
        Console.Error.WriteLine(
            $"decode_tokens={decodeForwards} decode_s={decodeSeconds.ToString("F4", CultureInfo.InvariantCulture)} decode_tok_s={Rate(decodeForwards, decodeSeconds)} simd={TensorMath.SimdKind} dtype=fp32");
        Console.Error.WriteLine("decode_tokens counts one-token forwards after the prefill. The first sampled token comes from the prefill logits and is not included.");
        return 0;
    }

    private static int Bench(Qwen2Model model, BpeTokenizer tokenizer, int[] promptIds, int decodeForwards)
    {
        int warmupToken = promptIds[0];
        var warmup = Stopwatch.StartNew();
        model.Forward(new[] { warmupToken });
        model.Forward(new[] { TensorMath.ArgMax(model.Logits) });
        warmup.Stop();
        model.Reset();

        var clock = Stopwatch.StartNew();
        model.Forward(promptIds);
        clock.Stop();
        double prefillSeconds = clock.Elapsed.TotalSeconds;
        int next = TensorMath.ArgMax(model.Logits);

        clock.Restart();
        for (int i = 0; i < decodeForwards; i++)
        {
            model.Forward(new[] { next });
            next = TensorMath.ArgMax(model.Logits);
        }
        clock.Stop();
        double decodeSeconds = clock.Elapsed.TotalSeconds;

        Console.WriteLine(tokenizer.Decode(promptIds[..Math.Min(promptIds.Length, 32)]));
        Console.Error.WriteLine(
            $"warmup_s={warmup.Elapsed.TotalSeconds.ToString("F4", CultureInfo.InvariantCulture)} simd={TensorMath.SimdKind} threads={Environment.ProcessorCount} dtype=fp32");
        Console.Error.WriteLine(
            $"prefill_tokens={promptIds.Length} prefill_s={prefillSeconds.ToString("F4", CultureInfo.InvariantCulture)} prefill_tok_s={Rate(promptIds.Length, prefillSeconds)}");
        Console.Error.WriteLine(
            $"decode_tokens={decodeForwards} decode_s={decodeSeconds.ToString("F4", CultureInfo.InvariantCulture)} decode_tok_s={Rate(decodeForwards, decodeSeconds)}");
        Console.Error.WriteLine("Bench is greedy, after one warmup token plus one warmup decode, with weights already loaded. Prefill reads every matrix once across the prompt. Decode is that many single-token forwards.");
        return 0;
    }

    private static string Rate(int tokens, double seconds)
    {
        if (seconds <= 0d || tokens <= 0)
            return "0";
        return (tokens / seconds).ToString("F3", CultureInfo.InvariantCulture);
    }

    private static Dictionary<string, string> Parse(ReadOnlySpan<string> args)
    {
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            if (!arg.StartsWith("--", StringComparison.Ordinal) || arg.Length < 3)
                throw new ArgumentException($"Unexpected argument '{arg}'.");
            string name = arg[2..];
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                map[name] = args[++i];
            else
                map[name] = "true";
        }
        return map;
    }

    private static int GetInt(Dictionary<string, string> options, string name, int fallback)
    {
        if (!options.TryGetValue(name, out string? text))
            return fallback;
        if (!int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
            throw new ArgumentException($"--{name} must be an integer.");
        return value;
    }

    private static float GetFloat(Dictionary<string, string> options, string name, float fallback)
    {
        if (!options.TryGetValue(name, out string? text))
            return fallback;
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
            throw new ArgumentException($"--{name} must be a number.");
        return value;
    }

    private static void PrintUsage()
    {
        Console.WriteLine(
            """
            fractal-infer generate --model <dir> --prompt <text> [--system <text>] [--raw] [--max-new-tokens 64] [--temperature 0] [--seed 1] [--context 2048] [--no-eos]
            fractal-infer bench --model <dir> [--prompt <text>] [--prefill-tokens 128] [--max-new-tokens 32] [--context 2048]

            --raw skips the Qwen2.5-Instruct chat template. temperature 0 is greedy argmax.
            Weights are not committed. Download them with scripts/download-qwen25-0.5b.sh.
            """);
    }
}
