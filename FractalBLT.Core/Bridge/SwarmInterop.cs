using System;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace FractalBLT.Core.Bridge;

/// <summary>
/// Phase 15: Hermes Core Bridge
/// Ultra-lean Named Pipe bridge directly on top of the unmanaged SwarmBus.
/// Serializes unmanaged memory directly into wire formats without intermediate managed objects.
/// </summary>
public class SwarmInterop : IDisposable
{
    private readonly NamedPipeServerStream _pipeServer;
    private readonly IntPtr _busPtr; // Unmanaged SwarmBus pointer from HiveMindOrchestrator
    private bool _isRunning;

    // Struct mapping directly to wire bytes
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct HermesPacket
    {
        public int OpCode; // 1 = Prompt, 2 = Context Injection
        public int PayloadLength;
        // Followed immediately by payload bytes in unmanaged stream
    }

    public SwarmInterop(IntPtr unmanagedBusPtr)
    {
        _busPtr = unmanagedBusPtr;
        _pipeServer = new NamedPipeServerStream(
            "Opure_Hermes_SwarmBus",
            PipeDirection.InOut,
            NamedPipeServerStream.MaxAllowedServerInstances,
            PipeTransmissionMode.Byte,
            PipeOptions.Asynchronous | PipeOptions.WriteThrough,
            8192, 8192);
    }

    public async Task StartBridgeAsync(CancellationToken cancellationToken)
    {
        _isRunning = true;
        Console.WriteLine("[Hermes Bridge] Awaiting external SDK connections on 'Opure_Hermes_SwarmBus'...");
        
        while (!cancellationToken.IsCancellationRequested && _isRunning)
        {
            if (!_pipeServer.IsConnected)
            {
                await _pipeServer.WaitForConnectionAsync(cancellationToken);
                Console.WriteLine("[Hermes Bridge] Hermes Client Connected. Direct memory streaming activated.");
            }

            try
            {
                int packetSize = Marshal.SizeOf<HermesPacket>();
                byte[] headerBuffer = new byte[packetSize];
                int bytesRead = 0;
                while (bytesRead < packetSize)
                {
                    int r = await _pipeServer.ReadAsync(headerBuffer, bytesRead, packetSize - bytesRead, cancellationToken);
                    if (r == 0) break;
                    bytesRead += r;
                }
                
                if (bytesRead == packetSize)
                {
                    bool isPrompt = false;
                    int payloadLength = 0;
                    
                    unsafe
                    {
                        fixed (byte* pHeader = headerBuffer)
                        {
                            HermesPacket* packet = (HermesPacket*)pHeader;
                            isPrompt = packet->OpCode == 1 && packet->PayloadLength <= 2048 * sizeof(char);
                            payloadLength = packet->PayloadLength;
                        }
                    }

                    if (isPrompt) // Prompt
                    {
                        byte[] payloadBuffer = System.Buffers.ArrayPool<byte>.Shared.Rent(payloadLength);
                        try
                        {
                            int payloadRead = 0;
                            while (payloadRead < payloadLength)
                            {
                                int r = await _pipeServer.ReadAsync(payloadBuffer, payloadRead, payloadLength - payloadRead, cancellationToken);
                                if (r == 0) break;
                                payloadRead += r;
                            }
                            
                            unsafe
                            {
                                byte* busBase = (byte*)_busPtr;
                                byte* promptDest = busBase + 8; // simplified offset for ActiveAgentId + TaskState

                                fixed (byte* pPayload = payloadBuffer)
                                {
                                    Buffer.MemoryCopy(pPayload, promptDest, payloadLength, payloadLength);
                                }
                            }
                            Console.WriteLine($"[Hermes Bridge] Ingested Prompt ({payloadLength} bytes) directly to SwarmBus.");
                        }
                        finally
                        {
                            System.Buffers.ArrayPool<byte>.Shared.Return(payloadBuffer);
                        }
                    }
                    else
                    {
                        // Drain invalid payload
                        byte[] drain = new byte[payloadLength];
                        int drainRead = 0;
                        while (drainRead < drain.Length)
                        {
                            int r = await _pipeServer.ReadAsync(drain, drainRead, drain.Length - drainRead, cancellationToken);
                            if (r == 0) break;
                            drainRead += r;
                        }
                    }

                    // Acknowledge via unmanaged write
                    int successCode = 200;
                    byte[] ackBytes = BitConverter.GetBytes(successCode);
                    await _pipeServer.WriteAsync(ackBytes, 0, ackBytes.Length, cancellationToken);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Hermes Bridge] Connection dropped: {ex.Message}");
                if (_pipeServer.IsConnected) _pipeServer.Disconnect();
            }
        }
    }

    public void Dispose()
    {
        _isRunning = false;
        _pipeServer.Dispose();
    }
}
