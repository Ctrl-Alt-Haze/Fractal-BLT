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
                int bytesRead = await _pipeServer.ReadAsync(headerBuffer, 0, headerBuffer.Length, cancellationToken);
                
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
                        // Read directly into the unmanaged SwarmBus PromptBuffer (assuming 8-byte offset for PromptBuffer)
                        // In a full implementation, we'd cast _busPtr to SwarmBus* and use its span
                        unsafe
                        {
                            byte* busBase = (byte*)_busPtr;
                            byte* promptDest = busBase + 8; // simplified offset for ActiveAgentId + TaskState

                            Span<byte> destSpan = new Span<byte>(promptDest, payloadLength);
                            await _pipeServer.ReadAsync(destSpan, cancellationToken);
                        }
                        
                        Console.WriteLine($"[Hermes Bridge] Ingested Prompt ({payloadLength} bytes) directly to SwarmBus.");
                    }
                    else
                    {
                        // Drain invalid payload
                        byte[] drain = new byte[payloadLength];
                        await _pipeServer.ReadAsync(drain, 0, drain.Length, cancellationToken);
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
