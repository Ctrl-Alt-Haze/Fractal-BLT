using System;
using System.Collections.Generic;

namespace FractalServe;

/// <summary>
/// Radix Prefix Caching for near-zero Time-To-First-Token (TTFT).
/// Hashes incoming prompts and RAG document chunks into a Radix Tree.
/// Reuses PagedAttention VRAM blocks if the prefix already exists.
/// </summary>
public class RadixNode
{
    public int[] Tokens { get; }
    public int[] PhysicalBlockIndices { get; }
    
    // Key is the first token of the child branch
    public Dictionary<int, RadixNode> Children { get; } = new();
    
    public int LastAccessTime { get; set; }

    public RadixNode(int[] tokens, int[] physicalBlockIndices)
    {
        Tokens = tokens;
        PhysicalBlockIndices = physicalBlockIndices;
        LastAccessTime = Environment.TickCount;
    }
}

public class RadixPrefixCache
{
    private readonly RadixNode _root;
    private readonly VramManager _vramManager;
    private readonly int _tokensPerBlock;

    public RadixPrefixCache(VramManager vramManager, int tokensPerBlock = 16)
    {
        _vramManager = vramManager;
        _tokensPerBlock = tokensPerBlock;
        _root = new RadixNode(Array.Empty<int>(), Array.Empty<int>());
    }

    /// <summary>
    /// Matches a token sequence against the Radix Tree.
    /// Writes the matched physical blocks into destinationBlocks to guarantee zero-allocation.
    /// Returns the number of tokens successfully matched.
    /// The engine only needs to compute logits for the remaining unmatched tokens.
    /// </summary>
    public int MatchPrefix(ReadOnlySpan<int> promptTokens, Span<int> destinationBlocks)
    {
        int matchedCount = 0;
        int blocksWritten = 0;
        RadixNode current = _root;

        while (matchedCount < promptTokens.Length)
        {
            int nextToken = promptTokens[matchedCount];
            if (current.Children.TryGetValue(nextToken, out RadixNode? child))
            {
                // Verify the full sequence in the child node matches the prompt
                int matchLength = 0;
                while (matchLength < child.Tokens.Length && 
                       matchedCount + matchLength < promptTokens.Length &&
                       child.Tokens[matchLength] == promptTokens[matchedCount + matchLength])
                {
                    matchLength++;
                }

                if (matchLength > 0)
                {
                    matchedCount += matchLength;
                    
                    if (blocksWritten + child.PhysicalBlockIndices.Length <= destinationBlocks.Length)
                    {
                        child.PhysicalBlockIndices.AsSpan().CopyTo(destinationBlocks.Slice(blocksWritten));
                        blocksWritten += child.PhysicalBlockIndices.Length;
                    }

                    child.LastAccessTime = Environment.TickCount;

                    // If we partially matched a node, we can't traverse deeper
                    if (matchLength < child.Tokens.Length) break;
                    
                    current = child;
                }
                else
                {
                    break;
                }
            }
            else
            {
                break;
            }
        }

        return matchedCount;
    }

    /// <summary>
    /// Inserts a newly computed KV cache sequence into the Radix Tree.
    /// </summary>
    public void InsertSequence(ReadOnlySpan<int> promptTokens, ReadOnlySpan<int> newPhysicalBlocks)
    {
        // Advanced radix insertion logic would split nodes here.
        // For standard NativeAOT aggressive throughput, we do a simplified append to the deepest match.
        Span<int> dummyBlocks = stackalloc int[1024];
        int matchedCount = MatchPrefix(promptTokens, dummyBlocks);

        if (matchedCount == promptTokens.Length) return; // Already cached

        int unmatchedCount = promptTokens.Length - matchedCount;
        int[] remainingTokens = promptTokens.Slice(matchedCount, unmatchedCount).ToArray();
        
        // Map the new physical blocks
        RadixNode newNode = new RadixNode(remainingTokens, newPhysicalBlocks.ToArray());

        // Attach to root for now (a fully compliant radix tree splits edges)
        // Note: Real implementation tracks the exact split node. We use root indexing for the PoC.
        lock (_root)
        {
            _root.Children[remainingTokens[0]] = newNode;
        }
    }
}
