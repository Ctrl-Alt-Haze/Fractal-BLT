using System.Text;
using FractalInfer;

namespace FractalInfer.Tests;

public class TensorMathTests
{
    [Fact]
    public unsafe void Dot_MatchesExactIntegerAndHalfProducts_IncludingUnaligned()
    {
        float[] left = new float[48];
        float[] right = new float[48];
        for (int i = 0; i < 17; i++)
        {
            left[i] = 1f;
            right[i] = i + 1;
        }
        fixed (float* a = left, b = right)
        {
            Assert.Equal(153f, TensorMath.Dot(a, b, 17));
        }

        for (int i = 0; i < 32; i++)
        {
            left[i + 1] = 0.5f;
            right[i + 1] = 0.5f;
        }
        fixed (float* a = left, b = right)
        {
            Assert.Equal(8f, TensorMath.Dot(a + 1, b + 1, 32));
        }
    }

    [Fact]
    public void Gemm_UsesPyTorchLinearLayoutAndBias()
    {
        // W rows are output features. Two sequence rows.
        float[] weight =
        [
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
        ];
        float[] input =
        [
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f,
        ];
        float[] bias = [10f, 20f];
        float[] output = new float[4];
        TensorMath.Gemm(input, weight, output, seq: 2, inDim: 4, outDim: 2, bias);
        Assert.Equal(new float[] { 11f, 22f, 15f, 26f }, output);
    }

    [Fact]
    public void RmsNorm_MatchesDefinition()
    {
        float[] input = [1f, 2f, 3f];
        float[] weight = [1f, 1f, 1f];
        float[] output = new float[3];
        const float eps = 1e-6f;
        TensorMath.RmsNorm(input, weight, output, eps);
        double meanSquare = (1d + 4d + 9d) / 3d;
        double inv = 1d / Math.Sqrt(meanSquare + eps);
        Assert.Equal((float)(1d * inv), output[0], precision: 5);
        Assert.Equal((float)(2d * inv), output[1], precision: 5);
        Assert.Equal((float)(3d * inv), output[2], precision: 5);
    }

    [Fact]
    public void SiluAndRope_MatchClosedForms()
    {
        float silu = TensorMath.Silu(1f);
        Assert.Equal(1f / (1f + MathF.Exp(-1f)), silu, precision: 6);

        float[] head = [1f, 2f, 3f, 4f];
        float[] inv = [1f, 0.01f];
        TensorMath.ApplyNeoXRope(head, position: 1, inv);
        float c0 = MathF.Cos(1f);
        float s0 = MathF.Sin(1f);
        float c1 = MathF.Cos(0.01f);
        float s1 = MathF.Sin(0.01f);
        Assert.Equal(1f * c0 - 3f * s0, head[0], precision: 5);
        Assert.Equal(2f * c1 - 4f * s1, head[1], precision: 5);
        Assert.Equal(3f * c0 + 1f * s0, head[2], precision: 5);
        Assert.Equal(4f * c1 + 2f * s1, head[3], precision: 5);
    }

    [Fact]
    public void ArgMax_PicksFirstMaximum_AndTemperatureSampleIsPeaked()
    {
        Assert.Equal(1, TensorMath.ArgMax([0.1f, 0.4f, 0.4f, -1f]));
        float[] scratch = new float[4];
        int sampled = TensorMath.SampleTemperature([100f, 0f, 0f], temperature: 1f, new Random(1), scratch);
        Assert.Equal(0, sampled);
        int again = TensorMath.SampleTemperature([0f, 1f, 2f, 3f], 0.8f, new Random(7), scratch);
        int repeat = TensorMath.SampleTemperature([0f, 1f, 2f, 3f], 0.8f, new Random(7), scratch);
        Assert.Equal(again, repeat);
        Assert.InRange(again, 0, 3);
    }

    [Fact]
    public void Bf16RoundTrip_PreservesTheFloat32TopBits()
    {
        Assert.Equal(1f, SafetensorsFile.Bf16ToSingle(0x3F80));
        Assert.Equal(-1f, SafetensorsFile.Bf16ToSingle(0xBF80));
        Assert.Equal(2f, SafetensorsFile.Bf16ToSingle(0x4000));
    }

    [Fact]
    public void Safetensors_ReadsBf16AndF32()
    {
        string path = Path.Combine(Path.GetTempPath(), "fractal-infer-tiny.safetensors");
        byte[] json = Encoding.UTF8.GetBytes(
            """
            {"bf":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"f":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}
            """);
        using (var stream = File.Create(path))
        {
            Span<byte> length = stackalloc byte[8];
            BitConverter.TryWriteBytes(length, (long)json.Length);
            stream.Write(length);
            stream.Write(json);
            stream.Write(new byte[] { 0x80, 0x3F, 0x00, 0x40 });
            stream.Write(BitConverter.GetBytes(1.5f));
        }

        var file = SafetensorsFile.Open(path);
        Assert.Equal(new float[] { 1f, 2f }, file.ReadFloat32("bf"));
        Assert.Equal(1.5f, file.ReadFloat32("f")[0]);
    }
}
