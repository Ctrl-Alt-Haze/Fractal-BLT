using System;
using System.Runtime.Intrinsics;

namespace FractalServe;

/// <summary>
/// Speculative Decoding (The Ghost Producer) - Phase 4
/// Spins up a lightweight draft model pass to predict 4-6 tokens ahead.
/// Overlaps compute with the primary 1.58-bit MoE verification pass to drastically multiply TPS.
/// </summary>
public unsafe class SpeculativeDecoder
{
    private readonly int _draftLookahead;

    public SpeculativeDecoder(int draftLookahead = 4)
    {
        _draftLookahead = draftLookahead;
    }

    /// <summary>
    /// Executes a fast, low-parameter draft prediction.
    /// Writes the speculated tokens into the provided Span to guarantee zero-allocation.
    /// </summary>
    public void GenerateDraftTokens(ReadOnlySpan<int> contextTokens, Span<int> draftTokens)
    {
        Console.WriteLine($"[Ghost Producer] Speculating next {draftTokens.Length} tokens via 1.5B Traffic Cop Model...");
        
        // Simulated draft prediction (In reality, this invokes the Harvester's small Phase 8 NativeAOT model instance)
        for (int i = 0; i < draftTokens.Length; i++)
        {
            // We use a dummy vocab hash simulation for the mock
            draftTokens[i] = (contextTokens.Length * 7 + i * 13) % 32000; 
        }
    }

    /// <summary>
    /// Evaluates the drafted tokens against the primary MoE logits in a single parallel batch.
    /// Uses greedy speculative decoding logic (argmax matching).
    /// Returns the number of accepted tokens and the final correction token.
    /// </summary>
    public int VerifyDraft(ReadOnlySpan<int> draftedTokens, float* primaryLogits, int vocabSize, out int correctionToken)
    {
        int accepted = 0;
        correctionToken = -1;
        
        for (int i = 0; i < draftedTokens.Length; i++)
        {
            // The primary model evaluates the entire draft sequence in parallel.
            // Logits for the i-th prediction are offset by i * vocabSize.
            float* stepLogits = primaryLogits + (i * vocabSize);
            
            // SIMD-accelerated ArgMax to find the primary model's predicted token
            int predictedToken = FindArgMaxSimd(stepLogits, vocabSize);

            if (predictedToken == draftedTokens[i])
            {
                accepted++;
            }
            else
            {
                // Speculation failed: reject this and all subsequent drafted tokens.
                // The correct token is the primary model's prediction.
                correctionToken = predictedToken;
                break;
            }
        }

        // If the entire drafted sequence was accepted flawlessly, we still need 
        // the primary model's prediction for the final drafted token.
        if (accepted == draftedTokens.Length)
        {
            float* stepLogits = primaryLogits + (accepted * vocabSize);
            correctionToken = FindArgMaxSimd(stepLogits, vocabSize);
        }

        Console.WriteLine($"[Ghost Producer] Verified {accepted}/{draftedTokens.Length} drafted tokens. TPS Multiplier: {accepted + 1}x");
        return accepted;
    }

    /// <summary>
    /// Hardware-accelerated SIMD ArgMax for extracting predictions over massive vocabularies (e.g., 128k+).
    /// </summary>
    private int FindArgMaxSimd(float* logits, int vocabSize)
    {
        if (vocabSize <= 0) return 0;
        
        int bestIndex = 0;
        float maxVal = logits[0];
        int i = 0;

        // Utilize AVX2/AVX512 inherently via generic Vector256
        if (Vector256.IsHardwareAccelerated && vocabSize >= Vector256<float>.Count)
        {
            int step = Vector256<float>.Count; // 8 lanes
            Vector256<float> maxVec = Vector256.Create(float.MinValue);
            Vector256<int> bestIndexVec = Vector256<int>.Zero;
            
            Vector256<int> indexVec = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            Vector256<int> incVec = Vector256.Create(8);

            for (; i <= vocabSize - step; i += step)
            {
                Vector256<float> curVec = Vector256.Load(logits + i);
                
                // Compare mask: 1s where curVec > maxVec
                Vector256<int> cmpMask = Vector256.GreaterThan(curVec, maxVec).AsInt32();
                
                // Blend values based on the mask
                maxVec = Vector256.ConditionalSelect(cmpMask.AsSingle(), curVec, maxVec);
                bestIndexVec = Vector256.ConditionalSelect(cmpMask, indexVec, bestIndexVec);
                
                indexVec += incVec;
            }

            // Collapse the 8-lane vector maximum down to a scalar
            for (int j = 0; j < step; j++)
            {
                float val = maxVec.GetElement(j);
                if (val > maxVal)
                {
                    maxVal = val;
                    bestIndex = bestIndexVec.GetElement(j);
                }
            }
        }

        // Tail sweep for any remaining vocab items
        for (; i < vocabSize; i++)
        {
            if (logits[i] > maxVal)
            {
                maxVal = logits[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
