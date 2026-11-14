using System.Text;
using System.Text.Json;

namespace FractalInfer;

/// <summary>
/// Reads a single-file safetensors blob and materialises tensors as float32.
/// BF16 and FP16 values are widened; the on-disk bytes are not modified.
/// </summary>
public sealed class SafetensorsFile
{
    private readonly string _path;
    private readonly Dictionary<string, TensorEntry> _tensors;

    private SafetensorsFile(string path, Dictionary<string, TensorEntry> tensors)
    {
        _path = path;
        _tensors = tensors;
    }

    public IReadOnlyCollection<string> Names => _tensors.Keys;

    public static SafetensorsFile Open(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new NotSupportedException("Safetensors loading expects a little-endian machine.");

        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Span<byte> lengthBytes = stackalloc byte[8];
        if (stream.Read(lengthBytes) != 8)
            throw new InvalidDataException("Safetensors file is shorter than the header length.");

        long headerLength = BitConverter.ToInt64(lengthBytes);
        if (headerLength <= 0 || headerLength > int.MaxValue || headerLength > stream.Length - 8)
            throw new InvalidDataException($"Invalid safetensors header length {headerLength}.");

        byte[] header = new byte[headerLength];
        stream.ReadExactly(header);
        long dataStart = 8 + headerLength;

        using var doc = JsonDocument.Parse(header);
        var tensors = new Dictionary<string, TensorEntry>(StringComparer.Ordinal);
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            if (property.NameEquals("__metadata__"))
                continue;
            if (!property.Value.TryGetProperty("dtype", out var dtypeEl) ||
                !property.Value.TryGetProperty("shape", out var shapeEl) ||
                !property.Value.TryGetProperty("data_offsets", out var offsetEl))
            {
                throw new InvalidDataException($"Tensor '{property.Name}' is missing dtype, shape, or data_offsets.");
            }

            string dtype = dtypeEl.GetString() ?? "";
            int elem = ElementSize(dtype);
            var shape = new int[shapeEl.GetArrayLength()];
            long numel = 1;
            int axis = 0;
            foreach (var dim in shapeEl.EnumerateArray())
            {
                int size = dim.GetInt32();
                if (size < 0)
                    throw new InvalidDataException($"Tensor '{property.Name}' has a negative dimension.");
                shape[axis++] = size;
                numel *= size;
            }

            long start = offsetEl[0].GetInt64();
            long end = offsetEl[1].GetInt64();
            long nbytes = end - start;
            if (start < 0 || end < start || nbytes != numel * elem)
                throw new InvalidDataException($"Tensor '{property.Name}' has inconsistent data_offsets.");

            tensors.Add(property.Name, new TensorEntry(dtype, shape, dataStart + start, nbytes));
        }

        return new SafetensorsFile(path, tensors);
    }

    public float[] ReadFloat32(string name)
    {
        if (!_tensors.TryGetValue(name, out var entry))
            throw new KeyNotFoundException($"Safetensors tensor '{name}' was not found in {_path}.");

        int elem = ElementSize(entry.DType);
        long count64 = entry.NumBytes / elem;
        if (count64 > int.MaxValue)
            throw new InvalidDataException($"Tensor '{name}' does not fit in a single float array.");

        int count = (int)count64;
        var dst = new float[count];
        byte[] chunk = new byte[4 * 1024 * 1024];
        using var stream = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.Read, chunk.Length, FileOptions.SequentialScan);
        stream.Seek(entry.AbsoluteOffset, SeekOrigin.Begin);

        int written = 0;
        long remaining = entry.NumBytes;
        while (remaining > 0)
        {
            int toRead = (int)Math.Min(chunk.Length, remaining);
            stream.ReadExactly(chunk, 0, toRead);
            int produced = Convert(entry.DType, chunk.AsSpan(0, toRead), dst.AsSpan(written));
            written += produced;
            remaining -= toRead;
        }

        if (written != count)
            throw new InvalidDataException($"Tensor '{name}' converted {written} values, expected {count}.");
        return dst;
    }

    public static float Bf16ToSingle(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private static int Convert(string dtype, ReadOnlySpan<byte> src, Span<float> dst)
    {
        switch (dtype)
        {
            case "F32":
            {
                int n = src.Length / 4;
                for (int i = 0; i < n; i++)
                    dst[i] = BitConverter.ToSingle(src.Slice(i * 4, 4));
                return n;
            }
            case "BF16":
            {
                int n = src.Length / 2;
                for (int i = 0; i < n; i++)
                {
                    ushort bits = BitConverter.ToUInt16(src.Slice(i * 2, 2));
                    dst[i] = Bf16ToSingle(bits);
                }
                return n;
            }
            case "F16":
            {
                int n = src.Length / 2;
                for (int i = 0; i < n; i++)
                {
                    ushort bits = BitConverter.ToUInt16(src.Slice(i * 2, 2));
                    dst[i] = (float)BitConverter.UInt16BitsToHalf(bits);
                }
                return n;
            }
            default:
                throw new NotSupportedException($"Safetensors dtype '{dtype}' is not supported.");
        }
    }

    private static int ElementSize(string dtype) => dtype switch
    {
        "F32" => 4,
        "F16" or "BF16" => 2,
        _ => throw new NotSupportedException($"Safetensors dtype '{dtype}' is not supported. This loader accepts F32, F16, and BF16."),
    };

    private readonly record struct TensorEntry(string DType, int[] Shape, long AbsoluteOffset, long NumBytes);
}
