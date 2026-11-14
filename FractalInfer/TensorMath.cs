using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace FractalInfer;

/// <summary>
/// Float32 kernels used by the Qwen2 forward pass. Dot products are vectorised
/// when AVX-512 or AVX2 is available; each output row is accumulated sequentially
/// so a parallel GEMM stays deterministic.
/// </summary>
public static class TensorMath
{
    public static string SimdKind =>
        Avx512F.IsSupported ? "avx512" :
        Avx2.IsSupported ? "avx2" :
        Vector.IsHardwareAccelerated ? $"vector{Vector<float>.Count}" : "scalar";

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float Dot(float* a, float* b, int n)
    {
        float sum = 0f;
        int i = 0;
        if (Avx512F.IsSupported)
        {
            Vector512<float> acc0 = Vector512<float>.Zero;
            Vector512<float> acc1 = Vector512<float>.Zero;
            int end = n & ~31;
            for (; i < end; i += 32)
            {
                acc0 = Avx512F.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), acc0);
                acc1 = Avx512F.FusedMultiplyAdd(Vector512.Load(a + i + 16), Vector512.Load(b + i + 16), acc1);
            }
            int end16 = n & ~15;
            for (; i < end16; i += 16)
                acc0 = Avx512F.FusedMultiplyAdd(Vector512.Load(a + i), Vector512.Load(b + i), acc0);
            sum = Vector512.Sum(acc0 + acc1);
        }
        else if (Avx2.IsSupported)
        {
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            int end = n & ~15;
            for (; i < end; i += 16)
            {
                Vector256<float> a0 = Vector256.Load(a + i);
                Vector256<float> b0 = Vector256.Load(b + i);
                Vector256<float> a1 = Vector256.Load(a + i + 8);
                Vector256<float> b1 = Vector256.Load(b + i + 8);
                if (Fma.IsSupported)
                {
                    acc0 = Fma.MultiplyAdd(a0, b0, acc0);
                    acc1 = Fma.MultiplyAdd(a1, b1, acc1);
                }
                else
                {
                    acc0 = Avx.Add(acc0, Avx.Multiply(a0, b0));
                    acc1 = Avx.Add(acc1, Avx.Multiply(a1, b1));
                }
            }
            sum = Vector256.Sum(acc0 + acc1);
        }

        for (; i < n; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>
    /// Y[seq, out] = X[seq, in] * W[out, in]^T + bias. W is stored row-major as in a PyTorch Linear weight.
    /// </summary>
    public static unsafe void Gemm(float[] x, float[] w, float[] y, int seq, int inDim, int outDim, float[]? bias)
    {
        if (seq < 0 || inDim < 0 || outDim < 0)
            throw new ArgumentOutOfRangeException(nameof(seq));
        if (seq == 0 || outDim == 0)
            return;
        if (w.LongLength != (long)outDim * inDim)
            throw new ArgumentException($"Weight length {w.LongLength} != {outDim}*{inDim}.", nameof(w));
        if (x.LongLength < (long)seq * inDim)
            throw new ArgumentException("Input is shorter than seq * inDim.", nameof(x));
        if (y.LongLength < (long)seq * outDim)
            throw new ArgumentException("Output is shorter than seq * outDim.", nameof(y));
        if (bias != null && bias.Length != outDim)
            throw new ArgumentException("Bias length does not match outDim.", nameof(bias));

        var hx = GCHandle.Alloc(x, GCHandleType.Pinned);
        var hw = GCHandle.Alloc(w, GCHandleType.Pinned);
        var hy = GCHandle.Alloc(y, GCHandleType.Pinned);
        GCHandle hb = default;
        bool hasBias = bias != null;
        if (hasBias)
            hb = GCHandle.Alloc(bias!, GCHandleType.Pinned);
        try
        {
            IntPtr px = hx.AddrOfPinnedObject();
            IntPtr pw = hw.AddrOfPinnedObject();
            IntPtr py = hy.AddrOfPinnedObject();
            IntPtr pb = hasBias ? hb.AddrOfPinnedObject() : IntPtr.Zero;
            int width = outDim >= 64 ? Environment.ProcessorCount : 1;
            Parallel.For(0, outDim, new ParallelOptions { MaxDegreeOfParallelism = width }, row =>
            {
                float* wrow = (float*)pw + (long)row * inDim;
                float b = pb != IntPtr.Zero ? ((float*)pb)[row] : 0f;
                for (int t = 0; t < seq; t++)
                {
                    float* xrow = (float*)px + (long)t * inDim;
                    ((float*)py)[(long)t * outDim + row] = b + Dot(wrow, xrow, inDim);
                }
            });
        }
        finally
        {
            hx.Free();
            hw.Free();
            hy.Free();
            if (hasBias)
                hb.Free();
        }
    }

    public static void RmsNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, Span<float> output, float eps)
    {
        if (input.Length != weight.Length || output.Length != input.Length)
            throw new ArgumentException("RMSNorm vectors must share a length.");
        float ss = 0f;
        for (int i = 0; i < input.Length; i++)
            ss += input[i] * input[i];
        float inv = 1f / MathF.Sqrt(ss / input.Length + eps);
        for (int i = 0; i < input.Length; i++)
            output[i] = weight[i] * input[i] * inv;
    }

    public static unsafe void RmsNormRows(float* input, float* weight, float* output, int rows, int dim, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            float* src = input + (long)r * dim;
            float* dst = output + (long)r * dim;
            float ss = 0f;
            for (int d = 0; d < dim; d++)
            {
                float v = src[d];
                ss += v * v;
            }
            float inv = 1f / MathF.Sqrt(ss / dim + eps);
            for (int d = 0; d < dim; d++)
                dst[d] = weight[d] * src[d] * inv;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Silu(float x)
    {
        if (x >= 80f)
            return x;
        if (x <= -80f)
            return 0f;
        return x / (1f + MathF.Exp(-x));
    }

    public static unsafe void SwiGluInPlace(float* gate, float* up, int n)
    {
        for (int i = 0; i < n; i++)
            gate[i] = Silu(gate[i]) * up[i];
    }

    public static unsafe void AddInPlace(float* destination, float* other, int n)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            for (; i + 8 <= n; i += 8)
            {
                Vector256<float> sum = Vector256.Load(destination + i) + Vector256.Load(other + i);
                Vector256.Store(sum, destination + i);
            }
        }
        for (; i < n; i++)
            destination[i] += other[i];
    }

    /// <summary>
    /// GPT-NeoX / Qwen2 rotate-half RoPE on one head. invFreq length is headDim/2.
    /// </summary>
    public static void ApplyNeoXRope(Span<float> head, int position, ReadOnlySpan<float> invFreq)
    {
        int half = head.Length / 2;
        if (invFreq.Length < half)
            throw new ArgumentException("invFreq is shorter than headDim/2.", nameof(invFreq));
        for (int i = 0; i < half; i++)
        {
            float angle = position * invFreq[i];
            float c = MathF.Cos(angle);
            float s = MathF.Sin(angle);
            float x0 = head[i];
            float x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x1 * c + x0 * s;
        }
    }

    public static int ArgMax(ReadOnlySpan<float> values)
    {
        if (values.Length == 0)
            throw new ArgumentException("Cannot argmax an empty span.", nameof(values));
        int best = 0;
        float max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Inverse-transform sample of softmax(logits / temperature). Scratch must be logits.Length.
    /// </summary>
    public static int SampleTemperature(ReadOnlySpan<float> logits, float temperature, Random random, Span<float> scratch)
    {
        if (temperature <= 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be positive. Use ArgMax for greedy decoding.");
        if (scratch.Length < logits.Length)
            throw new ArgumentException("Scratch is shorter than logits.", nameof(scratch));

        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
                max = logits[i];
        }

        double sum = 0d;
        float invT = 1f / temperature;
        for (int i = 0; i < logits.Length; i++)
        {
            float p = MathF.Exp((logits[i] - max) * invT);
            scratch[i] = p;
            sum += p;
        }

        double threshold = random.NextDouble() * sum;
        double cumulative = 0d;
        for (int i = 0; i < logits.Length; i++)
        {
            cumulative += scratch[i];
            if (cumulative >= threshold)
                return i;
        }
        return logits.Length - 1;
    }
}
