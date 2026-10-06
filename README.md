# FRACTAL-BLT (Sovereign Apex)
(https://ctrl-alt-haze.github.io/Fractal-BLT/)
### *Disk-Native, Zero-Allocation Mixture-of-Experts (MoE) Inference Runtime in .NET 10 NativeAOT*

The **Absolute Apex of Local AI Inference.** Fractal-BLT is a massive 16-phase engineering leap, bringing 30B MoE models directly to consumer hardware (like the RTX 5070 Ti Blackwell) without cloud latency, Python wrappers, or GC bloat. It leverages a strict zero-allocation NativeAOT C# architecture.

## 🔥 The sovereign architecture
Fractal-BLT bypasses conventional bottlenecks:
* **2.4µs AVX2 SIMD Retrieval:** An unmanaged SIMD vectorization pathway performs HNSW graph traversal natively without memory allocations.
* **TMA GPUDirect Weight Streaming:** Models load via `cp.async.bulk.tensor` DMA directly from NVMe to VRAM.
* **1.58-Bit Ternary Quantization:** Crush models into an aggressively compact format to eliminate FP16 bloat using `ModelHarvester`.

## 🚀 Quick Start
Fractal-BLT configures itself effortlessly.
1. Run the bootstrapper:
   ```powershell
   ./Installer/Forge-Fractal.ps1
   ```
   This triggers `dotnet publish -c Release -r win-x64 /p:PublishAot=true`, locking the ecosystem in a Windows Job Object (OZTAE containment) and pulling the 1.5B traffic cop model automatically.

## 🧠 Gemini Hive-Mind & Hermes SDK
Fractal-BLT features a 4-lane Thought-Matrix (`Router`, `Coder`, `Critic`, `ToolForger`).
Interact via the **Hermes SDK** (over unmanaged gRPC/Named Pipes) for 0-latency programmatic queries.

*Welcome to the Resistance. Build your NativeAOT hive-minds locally.*
