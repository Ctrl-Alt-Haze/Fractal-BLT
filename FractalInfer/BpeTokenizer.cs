using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace FractalInfer;

/// <summary>
/// Hugging Face tokenizers BPE reader for Qwen2 tokenizer.json:
/// NFC (when declared), special/added-token splitting, the Qwen regex pretokenizer,
/// GPT-2 byte-level encoding, and rank-ordered BPE merges.
/// </summary>
public sealed partial class BpeTokenizer
{
    private const string QwenRegex =
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string Left, string Right), int> _mergeRank;
    private readonly AddedToken[] _addedByLength;
    private readonly Dictionary<int, string> _addedLiterals;
    private readonly string[] _idToToken;
    private readonly char[] _byteToUnicode;
    private readonly byte[] _unicodeToByte;
    private readonly bool[] _unicodeKnown;
    private readonly bool _normalizeNfc;
    private readonly Dictionary<string, int[]> _cache = new(StringComparer.Ordinal);

    private BpeTokenizer(
        Dictionary<string, int> vocab,
        Dictionary<(string Left, string Right), int> mergeRank,
        AddedToken[] addedByLength,
        Dictionary<int, string> addedLiterals,
        string[] idToToken,
        char[] byteToUnicode,
        byte[] unicodeToByte,
        bool[] unicodeKnown,
        bool normalizeNfc)
    {
        _vocab = vocab;
        _mergeRank = mergeRank;
        _addedByLength = addedByLength;
        _addedLiterals = addedLiterals;
        _idToToken = idToToken;
        _byteToUnicode = byteToUnicode;
        _unicodeToByte = unicodeToByte;
        _unicodeKnown = unicodeKnown;
        _normalizeNfc = normalizeNfc;
    }

    public int MaxTokenId => _idToToken.Length - 1;

    public static BpeTokenizer Load(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJsonPath));
        JsonElement root = doc.RootElement;
        JsonElement model = root.GetProperty("model");
        if (model.GetProperty("type").GetString() != "BPE")
            throw new NotSupportedException("tokenizer.json model.type is not BPE.");
        if (model.TryGetProperty("byte_fallback", out var fallback) && fallback.ValueKind == JsonValueKind.True)
            throw new NotSupportedException("byte_fallback tokenizers are not implemented.");

        bool nfc = false;
        if (root.TryGetProperty("normalizer", out var normalizer) && normalizer.ValueKind == JsonValueKind.Object)
        {
            string? kind = normalizer.GetProperty("type").GetString();
            if (kind == "NFC")
                nfc = true;
            else if (!string.IsNullOrEmpty(kind))
                throw new NotSupportedException($"Normalizer '{kind}' is not implemented.");
        }

        ReadPretokenizer(root);

        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        int maxId = 0;
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
        {
            int id = entry.Value.GetInt32();
            vocab[entry.Name] = id;
            if (id > maxId)
                maxId = id;
        }

        var ranks = new Dictionary<(string Left, string Right), int>();
        int rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            string left;
            string right;
            if (merge.ValueKind == JsonValueKind.String)
            {
                string text = merge.GetString() ?? "";
                int space = text.IndexOf(' ');
                if (space <= 0 || space >= text.Length - 1)
                    throw new InvalidDataException($"Cannot split BPE merge '{text}'.");
                left = text[..space];
                right = text[(space + 1)..];
            }
            else if (merge.ValueKind == JsonValueKind.Array && merge.GetArrayLength() == 2)
            {
                left = merge[0].GetString() ?? "";
                right = merge[1].GetString() ?? "";
            }
            else
            {
                throw new InvalidDataException("BPE merge is neither a string nor a pair.");
            }
            ranks[(left, right)] = rank++;
        }

        var added = new List<AddedToken>();
        if (root.TryGetProperty("added_tokens", out var addedEl))
        {
            foreach (var token in addedEl.EnumerateArray())
            {
                string content = token.GetProperty("content").GetString() ?? "";
                int id = token.GetProperty("id").GetInt32();
                added.Add(new AddedToken(content, id));
                if (id > maxId)
                    maxId = id;
            }
        }

        added.Sort(static (a, b) =>
        {
            int byLength = b.Content.Length.CompareTo(a.Content.Length);
            return byLength != 0 ? byLength : a.Id.CompareTo(b.Id);
        });

        var idToToken = new string[maxId + 1];
        foreach (var pair in vocab)
            idToToken[pair.Value] = pair.Key;
        var addedLiterals = new Dictionary<int, string>();
        foreach (var token in added)
        {
            idToToken[token.Id] = token.Content;
            addedLiterals[token.Id] = token.Content;
        }

        (char[] b2u, byte[] u2b, bool[] known) = BuildByteMap();
        return new BpeTokenizer(vocab, ranks, added.ToArray(), addedLiterals, idToToken, b2u, u2b, known, nfc);
    }

    /// <summary>
    /// Qwen2.5-Instruct chat wrapper for one user turn and an optional system prompt.
    /// Matches the no-tools branch of the shipped Jinja template, including the default system text.
    /// </summary>
    public static string ApplyQwen25Instruct(string user, string? system = null)
    {
        system ??= "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
        return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    }

    public int[] Encode(string text)
    {
        var ids = new List<int>();
        int index = 0;
        while (index < text.Length)
        {
            int added = MatchAdded(text.AsSpan(index));
            if (added >= 0)
            {
                ids.Add(_addedByLength[added].Id);
                index += _addedByLength[added].Content.Length;
                continue;
            }

            int next = NextAdded(text, index);
            if (next > index)
            {
                string segment = text[index..next];
                if (_normalizeNfc)
                    segment = segment.Normalize(NormalizationForm.FormC);
                EncodeOrdinary(segment, ids);
            }
            index = next;
        }
        return ids.ToArray();
    }

    public string Decode(ReadOnlySpan<int> ids)
    {
        var utf8 = new List<byte>(ids.Length * 2);
        var text = new StringBuilder();
        for (int i = 0; i < ids.Length; i++)
        {
            int id = ids[i];
            if (_addedLiterals.TryGetValue(id, out string? literal))
            {
                FlushBytes(utf8, text);
                text.Append(literal);
                continue;
            }
            string token = TokenString(id);
            for (int c = 0; c < token.Length; c++)
            {
                char ch = token[c];
                if ((uint)ch >= (uint)_unicodeKnown.Length || !_unicodeKnown[ch])
                    throw new InvalidOperationException($"Cannot byte-decode U+{(int)ch:X4} from token {id}.");
                utf8.Add(_unicodeToByte[ch]);
            }
        }
        FlushBytes(utf8, text);
        return text.ToString();
    }

    private void EncodeOrdinary(string segment, List<int> ids)
    {
        if (segment.Length == 0)
            return;
        int cursor = 0;
        foreach (Match match in Pretokenizer().Matches(segment))
        {
            if (match.Index > cursor)
                EncodePretoken(segment[cursor..match.Index], ids);
            if (match.Length > 0)
                EncodePretoken(match.Value, ids);
            cursor = match.Index + match.Length;
        }
        if (cursor < segment.Length)
            EncodePretoken(segment[cursor..], ids);
    }

    private void EncodePretoken(string piece, List<int> ids)
    {
        string encoded = ToByteLevel(piece);
        if (_cache.TryGetValue(encoded, out int[]? cached))
        {
            ids.AddRange(cached);
            return;
        }

        var symbols = new List<string>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++)
            symbols.Add(encoded[i].ToString());
        symbols = Merge(symbols);

        var pieceIds = new int[symbols.Count];
        for (int i = 0; i < symbols.Count; i++)
        {
            if (!_vocab.TryGetValue(symbols[i], out int id))
                throw new InvalidOperationException($"BPE symbol is missing from the vocabulary (length {symbols[i].Length}).");
            pieceIds[i] = id;
        }
        _cache[encoded] = pieceIds;
        ids.AddRange(pieceIds);
    }

    private List<string> Merge(List<string> word)
    {
        while (word.Count >= 2)
        {
            int bestRank = int.MaxValue;
            string? bestLeft = null;
            string? bestRight = null;
            for (int i = 0; i < word.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((word[i], word[i + 1]), out int rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestLeft = word[i];
                    bestRight = word[i + 1];
                }
            }
            if (bestLeft is null || bestRight is null)
                break;

            var merged = new List<string>(word.Count - 1);
            for (int i = 0; i < word.Count;)
            {
                if (i < word.Count - 1 && word[i] == bestLeft && word[i + 1] == bestRight)
                {
                    merged.Add(bestLeft + bestRight);
                    i += 2;
                }
                else
                {
                    merged.Add(word[i]);
                    i++;
                }
            }
            word = merged;
        }
        return word;
    }

    private string ToByteLevel(string piece)
    {
        byte[] raw = Encoding.UTF8.GetBytes(piece);
        var chars = new char[raw.Length];
        for (int i = 0; i < raw.Length; i++)
            chars[i] = _byteToUnicode[raw[i]];
        return new string(chars);
    }

    private int MatchAdded(ReadOnlySpan<char> span)
    {
        for (int i = 0; i < _addedByLength.Length; i++)
        {
            string content = _addedByLength[i].Content;
            if (span.StartsWith(content, StringComparison.Ordinal))
                return i;
        }
        return -1;
    }

    private int NextAdded(string text, int start)
    {
        int next = text.Length;
        foreach (var token in _addedByLength)
        {
            int found = text.IndexOf(token.Content, start, StringComparison.Ordinal);
            if (found >= 0 && found < next)
                next = found;
        }
        return next;
    }

    private string TokenString(int id)
    {
        if ((uint)id >= (uint)_idToToken.Length || _idToToken[id] is null)
            throw new InvalidOperationException($"Token id {id} is not in this tokenizer.");
        return _idToToken[id];
    }

    private static void FlushBytes(List<byte> utf8, StringBuilder text)
    {
        if (utf8.Count == 0)
            return;
        text.Append(Encoding.UTF8.GetString(utf8.ToArray()));
        utf8.Clear();
    }

    private static void ReadPretokenizer(JsonElement root)
    {
        JsonElement pre = root.GetProperty("pre_tokenizer");
        if (pre.GetProperty("type").GetString() != "Sequence")
            throw new NotSupportedException("Only a Sequence pretokenizer is implemented.");

        string? regex = null;
        bool sawByteLevel = false;
        foreach (var step in pre.GetProperty("pretokenizers").EnumerateArray())
        {
            string type = step.GetProperty("type").GetString() ?? "";
            if (type == "Split")
            {
                regex = step.GetProperty("pattern").GetProperty("Regex").GetString();
                string behavior = step.GetProperty("behavior").GetString() ?? "";
                if (behavior != "Isolated")
                    throw new NotSupportedException($"Split behavior '{behavior}' is not implemented.");
                if (step.TryGetProperty("invert", out var invert) && invert.ValueKind == JsonValueKind.True)
                    throw new NotSupportedException("Inverted pretokenizer splits are not implemented.");
            }
            else if (type == "ByteLevel")
            {
                sawByteLevel = true;
                if (step.TryGetProperty("add_prefix_space", out var prefix) && prefix.ValueKind == JsonValueKind.True)
                    throw new NotSupportedException("ByteLevel add_prefix_space=true is not implemented.");
            }
            else
            {
                throw new NotSupportedException($"Pretokenizer step '{type}' is not implemented.");
            }
        }

        if (!sawByteLevel)
            throw new NotSupportedException("Tokenizer is missing a ByteLevel pretokenizer.");
        if (regex != QwenRegex)
            throw new NotSupportedException("The pretokenizer regex is not the Qwen2 pattern this loader implements.");
    }

    private static (char[] ByteToUnicode, byte[] UnicodeToByte, bool[] Known) BuildByteMap()
    {
        var bytes = new List<int>(256);
        for (int i = '!'; i <= '~'; i++)
            bytes.Add(i);
        for (int i = '¡'; i <= '¬'; i++)
            bytes.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++)
            bytes.Add(i);

        var codes = new List<int>(bytes);
        var seen = new bool[256];
        foreach (int value in bytes)
            seen[value] = true;
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            if (seen[b])
                continue;
            bytes.Add(b);
            codes.Add(256 + extra);
            extra++;
        }

        var byteToUnicode = new char[256];
        var unicodeToByte = new byte[512];
        var known = new bool[512];
        for (int i = 0; i < bytes.Count; i++)
        {
            int b = bytes[i];
            int code = codes[i];
            if ((uint)code >= (uint)known.Length)
                throw new InvalidOperationException("GPT-2 byte map exceeded the expected unicode range.");
            byteToUnicode[b] = (char)code;
            unicodeToByte[code] = (byte)b;
            known[code] = true;
        }
        return (byteToUnicode, unicodeToByte, known);
    }

    [GeneratedRegex(QwenRegex)]
    private static partial Regex Pretokenizer();

    private readonly record struct AddedToken(string Content, int Id);
}
