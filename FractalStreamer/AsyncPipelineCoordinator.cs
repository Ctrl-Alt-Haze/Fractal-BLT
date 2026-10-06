using System;
using System.Threading.Channels;
using System.Threading.Tasks;
using FractalStreamer;
using FractalBridge;

namespace FractalStreamer;

/// <summary>
/// Decouples NVMe disk reads and GPU memory copies using System.Threading.Channels.
/// </summary>
public class AsyncPipelineCoordinator : IDisposable
{
    private readonly MoEStagingBuffer _stagingBuffer;
    private readonly ITensorStreamer _streamer;
    private readonly Channel<MoEBlock> _gpuQueue;
    private IntPtr _cudaContext;
    private IntPtr _cudaStream;
    private readonly bool _useCuda;
    public MoEStagingBuffer StagingBuffer => _stagingBuffer;
    
    public AsyncPipelineCoordinator(MoEStagingBuffer stagingBuffer, ITensorStreamer streamer, int channelCapacity = 4, bool useCuda = true)
    {
        _stagingBuffer = stagingBuffer;
        _streamer = streamer;
        _useCuda = useCuda;
        
        // Initialize a bounded channel for back‑pressure
        _gpuQueue = Channel.CreateBounded<MoEBlock>(new BoundedChannelOptions(channelCapacity) { SingleReader = true, SingleWriter = true });

        if (_useCuda)
        {
            // Initialize CUDA context and stream (assuming device 0 for scaffolding)
            CudaNative.cuInit(0);
            CudaNative.cuDeviceGet(out int device, 0);
            CudaNative.cuCtxCreate(out _cudaContext, 0, device);
            CudaNative.cuStreamCreate(out _cudaStream, 0);
        }
    }

    public async Task RunAsync(string safetensorsPath, long[] offsets, long[] sizes)
    {
        using var cts = new CancellationTokenSource();
        var diskTask = StartDiskThreadAsync(safetensorsPath, offsets, sizes, cts);
        var gpuTask = StartGpuThreadAsync(cts);
        
        await Task.WhenAll(diskTask, gpuTask);
    }

    /// <summary>
    /// Disk I/O loop: Reads chunks from NVMe into Empty blocks and queues them.
    /// </summary>
    public async Task StartDiskThreadAsync(string safetensorsPath, long[] offsets, long[] sizes, CancellationTokenSource cts)
    {
        try
        {
            int blockIndex = 0;
            for (int i = 0; i < offsets.Length; i++)
            {
                cts.Token.ThrowIfCancellationRequested();

                var block = _stagingBuffer.Blocks[blockIndex];
                
                // Spin-wait or lock-free wait for the block to become Empty
                while (block.State != BlockState.Empty)
                {
                    cts.Token.ThrowIfCancellationRequested();
                    await Task.Yield();
                }

                block.State = BlockState.Filling;
                
                // Read directly into the block's host memory using zero‑allocation path
                unsafe
                {
                    _streamer.ReadInto(safetensorsPath, offsets[i], (int)sizes[i], block.HostPtr);
                }
                
                block.ValidBytes = sizes[i];
                block.State = BlockState.ReadyForGPU;
                await _gpuQueue.Writer.WriteAsync(block, cts.Token);

                blockIndex = (blockIndex + 1) % _stagingBuffer.Blocks.Length;
            }
            
            _gpuQueue.Writer.Complete();
        }
        catch (Exception ex)
        {
            cts.Cancel();
            _gpuQueue.Writer.TryComplete(ex);
            throw;
        }
    }

    /// <summary>
    /// GPU loop: Receives Ready blocks, initiates HtoD async copies, and recycles blocks.
    /// </summary>
    public async Task StartGpuThreadAsync(CancellationTokenSource cts)
    {
        try
        {
            await foreach (var block in _gpuQueue.Reader.ReadAllAsync(cts.Token))
            {
                block.State = BlockState.Copying;

                if (_useCuda)
                {
                    unsafe
                    {
                        // Submit async HtoD memory copy on the dedicated stream
                        CudaNative.cuMemcpyHtoDAsync(
                            block.DevicePtr,
                            (IntPtr)block.HostPtr,
                            (nuint)block.ValidBytes,
                            _cudaStream);
                    }

                    // Synchronize stream (blocks GPU thread, but not Disk thread)
                    CudaNative.cuStreamSynchronize(_cudaStream);
                }

                // Recycle the block
                block.State = BlockState.Empty;
            }
        }
        catch (Exception)
        {
            cts.Cancel();
            throw;
        }
    }

    private bool _disposed;

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_cudaStream != IntPtr.Zero)
            {
                try { CudaNative.cuStreamDestroy(_cudaStream); } catch { }
                _cudaStream = IntPtr.Zero;
            }
            if (_cudaContext != IntPtr.Zero)
            {
                try { CudaNative.cuCtxDestroy(_cudaContext); } catch { }
                _cudaContext = IntPtr.Zero;
            }
            _disposed = true;
        }
    }
}
