using System;
using System.Collections.Generic;
using FractalBridge;

namespace FractalServe;

/// <summary>
/// A zero-allocation PagedAttention VRAM Defragmenter and KV Cache Manager.
/// Pre-allocates a massive contiguous block of VRAM and slices it into fixed token pages.
/// Prevents VRAM fragmentation by mapping non-contiguous physical pages to virtual sequences.
/// </summary>
public unsafe class VramManager : IDisposable
{
    private IntPtr _vramPool;
    private readonly nuint _poolSize;
    
    // Model Hot-Swap Partition (Phase 6)
    private IntPtr _swapPartition;
    private readonly nuint _swapPartitionSize;

    private readonly uint _blockSize;
    private readonly int _totalBlocks;
    
    private readonly Stack<int> _freeBlocks;
    private bool _isDisposed;

    // Default: 4096 hidden size * 16 tokens * 4 bytes (FP32) = 262,144 bytes per block (256 KB)
    // swapPartitionGB = 6GB by default for rapid hot-swapping of dense 1.58-bit model shards
    public VramManager(uint hiddenSize = 4096, uint tokensPerBlock = 16, int maxBlocks = 1024, uint swapPartitionGB = 6)
    {
        _blockSize = hiddenSize * tokensPerBlock * sizeof(float);
        _totalBlocks = maxBlocks;
        _poolSize = (nuint)(_blockSize * _totalBlocks);
        _swapPartitionSize = (nuint)swapPartitionGB * 1024 * 1024 * 1024;

        Console.WriteLine($"[VramManager] Initializing Paged KV Cache...");
        Console.WriteLine($"[VramManager] Block Size: {_blockSize / 1024} KB | Total Blocks: {_totalBlocks} | Pool Size: {_poolSize / (1024 * 1024)} MB");
        Console.WriteLine($"[VramManager] Allocating Model Swap Partition: {_swapPartitionSize / (1024 * 1024 * 1024)} GB");

        FractalBridge.CudaNative.MemAlloc(out _vramPool, _poolSize);
        FractalBridge.CudaNative.MemAlloc(out _swapPartition, _swapPartitionSize);

        _freeBlocks = new Stack<int>(_totalBlocks);
        
        // Push all blocks to the free stack (in reverse order so block 0 is popped first)
        for (int i = _totalBlocks - 1; i >= 0; i--)
        {
            _freeBlocks.Push(i);
        }
    }

    /// <summary>
    /// Allocates the requested number of blocks from the free list.
    /// Writes physical block indices into the provided Span to guarantee zero-allocation.
    /// Throws an OutOfMemoryException if not enough blocks are available.
    /// </summary>
    public void AllocateSequence(int requestedBlocks, Span<int> destination)
    {
        if (_isDisposed) throw new ObjectDisposedException(nameof(VramManager));
        if (destination.Length < requestedBlocks) throw new ArgumentException("Destination span is too small.");

        lock (_freeBlocks)
        {
            if (_freeBlocks.Count < requestedBlocks)
            {
                throw new OutOfMemoryException($"[VramManager] OOM! Requested {requestedBlocks} blocks, but only {_freeBlocks.Count} are free.");
            }

            for (int i = 0; i < requestedBlocks; i++)
            {
                destination[i] = _freeBlocks.Pop();
            }
        }
    }

    /// <summary>
    /// Frees a previously allocated sequence of blocks, returning them to the pool.
    /// </summary>
    public void FreeSequence(ReadOnlySpan<int> blockIndices)
    {
        if (_isDisposed) return;

        lock (_freeBlocks)
        {
            for (int i = 0; i < blockIndices.Length; i++)
            {
                _freeBlocks.Push(blockIndices[i]);
            }
        }
    }

    /// <summary>
    /// Gets the raw VRAM pointer for a specific physical block index.
    /// </summary>
    public IntPtr GetBlockPointer(int physicalBlockIndex)
    {
        if (physicalBlockIndex < 0 || physicalBlockIndex >= _totalBlocks)
            throw new ArgumentOutOfRangeException(nameof(physicalBlockIndex));

        return _vramPool + (physicalBlockIndex * (int)_blockSize);
    }

    /// <summary>
    /// Resolves physical block indices into physical VRAM pointers.
    /// Writes the results into the provided Span to guarantee zero-allocation.
    /// This array serves as the BlockTable passed to the PagedAttention PTX kernel.
    /// </summary>
    public void GetBlockTablePointers(ReadOnlySpan<int> physicalBlockIndices, Span<IntPtr> destination)
    {
        if (destination.Length < physicalBlockIndices.Length) throw new ArgumentException("Destination span is too small.");
        
        for (int i = 0; i < physicalBlockIndices.Length; i++)
        {
            destination[i] = GetBlockPointer(physicalBlockIndices[i]);
        }
    }

    /// <summary>
    /// Returns the raw pointer to the hot-swap partition for CuFileReader DMA operations.
    /// </summary>
    public IntPtr GetSwapPartitionPointer() => _swapPartition;
    
    public nuint GetSwapPartitionSize() => _swapPartitionSize;

    public void Dispose()
    {
        if (!_isDisposed)
        {
            if (_vramPool != IntPtr.Zero)
            {
                FractalBridge.CudaNative.MemFree(_vramPool);
                _vramPool = IntPtr.Zero;
            }
            if (_swapPartition != IntPtr.Zero)
            {
                FractalBridge.CudaNative.MemFree(_swapPartition);
                _swapPartition = IntPtr.Zero;
            }
            _isDisposed = true;
        }
    }
}
