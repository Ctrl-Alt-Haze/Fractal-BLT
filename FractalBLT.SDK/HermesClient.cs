using System;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace FractalBLT.SDK;

/// <summary>
/// Phase 15: Hermes Multi-Agent SDK Client
/// Exposes asynchronous endpoints for external Hermes agents to submit prompts, 
/// inject context into the PagedAttention memory pool, and request JIT tool compilations.
/// </summary>
public class HermesClient : IDisposable
{
    private NamedPipeClientStream _pipeClient;

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct HermesPacket
    {
        public int OpCode; // 1 = Prompt, 2 = Context Injection
        public int PayloadLength;
    }

    public HermesClient()
    {
        _pipeClient = new NamedPipeClientStream(".", "Opure_Hermes_SwarmBus", PipeDirection.InOut, PipeOptions.Asynchronous);
    }

    public async Task ConnectAsync(int timeoutMs = 5000, CancellationToken cancellationToken = default)
    {
        await _pipeClient.ConnectAsync(timeoutMs, cancellationToken);
    }

    public async Task<bool> SubmitPromptAsync(string prompt, CancellationToken cancellationToken = default)
    {
        if (!_pipeClient.IsConnected) throw new InvalidOperationException("HermesClient is not connected to the SwarmBus.");

        byte[] promptBytes = Encoding.UTF8.GetBytes(prompt);
        int maxLen = Math.Min(promptBytes.Length, 2048 * sizeof(char));

        var packet = new HermesPacket { OpCode = 1, PayloadLength = maxLen };
        
        // Write header
        int headerSize = Marshal.SizeOf<HermesPacket>();
        byte[] headerBuffer = new byte[headerSize];
        unsafe
        {
            fixed (byte* pHeader = headerBuffer)
            {
                *(HermesPacket*)pHeader = packet;
            }
        }
        await _pipeClient.WriteAsync(headerBuffer, 0, headerBuffer.Length, cancellationToken);

        // Write payload directly from memory without overhead
        await _pipeClient.WriteAsync(promptBytes, 0, maxLen, cancellationToken);

        // Await Ack
        byte[] ackBuffer = new byte[sizeof(int)];
        int read = await _pipeClient.ReadAsync(ackBuffer, 0, ackBuffer.Length, cancellationToken);
        if (read == sizeof(int))
        {
            int code = BitConverter.ToInt32(ackBuffer, 0);
            return code == 200;
        }

        return false;
    }

    public void Dispose()
    {
        _pipeClient?.Dispose();
    }
}
