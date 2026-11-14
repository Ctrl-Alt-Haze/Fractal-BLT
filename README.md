# Fractal-BLT

Local inference experiments in C# on .NET 10, with a NativeAOT publish path. The repository used to describe a finished disk-native MoE runtime. That description was ahead of the code. This file matches what the tree does today.

## What works

`FractalInfer` is a real float32 decoder for **Qwen2.5-0.5B-Instruct** (dense Qwen2, not the Qwen3-30B-A3B config under `Models/`):

- safetensors load (BF16, FP16, or FP32 widened to float32)
- `tokenizer.json` BPE (NFC, added tokens, Qwen regex, GPT-2 byte level)
- embedding, RMSNorm, NeoX RoPE, grouped-query attention, KV cache, SwiGLU, tied LM head
- greedy argmax and temperature sampling
- a CLI, `fractal-infer`

Qwen2.5-0.5B-Instruct is the first target because it is a small dense decoder (24 layers, hidden 896, about 0.5B parameters, one ~1 GB BF16 file) and it fits a CPU correctness test. TinyLlama-1.1B is larger and was not needed.

```bash
bash scripts/download-qwen25-0.5b.sh models/Qwen2.5-0.5B-Instruct
dotnet run --project FractalInfer.Cli -c Release -- generate \
  --model models/Qwen2.5-0.5B-Instruct \
  --prompt "The capital of France is" \
  --max-new-tokens 32 \
  --temperature 0
```

`--raw` skips the Qwen2.5-Instruct chat template. `--temperature 0` is greedy. A positive temperature samples with `--seed`. Weights are not in git.

`tools/dump_hf_reference.py` (test-only, Python + transformers) dumps float32 eager logits. `FractalInfer.Tests` checks the max absolute logit error against that dump and checks that greedy decoding matches for 32 tokens. The measured error is in the pull request description, not claimed here as a benchmark of a GPU.

```bash
dotnet test FractalInfer.Tests -c Release
```

Set `FRACTAL_MODEL_DIR` if the checkpoint is not in `models/Qwen2.5-0.5B-Instruct`. The Hugging Face parity test runs when `model.safetensors` is present. `FRACTAL_REQUIRE_HF_PARITY=1` makes a missing checkpoint a failure. On the 4-vCPU machine used to land this, that test reported a max absolute logit difference of `4.43e-5` and 32 matching greedy tokens. Re-run it after any math change. GPU tests in `FractalStreamer.Tests` catch a missing CUDA driver and return without failing.

Bench the CPU path after a one-token warmup (weights already loaded):

```bash
dotnet run --project FractalInfer.Cli -c Release -- bench \
  --model models/Qwen2.5-0.5B-Instruct \
  --prefill-tokens 128 \
  --max-new-tokens 32
```

NativeAOT:

```bash
dotnet publish FractalInfer.Cli/FractalInfer.Cli.csproj -c Release -r linux-x64 --self-contained true /p:PublishAot=true
```

That publish defaults to the build CPU's instruction set. CI publishes `x86-64-v3` (AVX2) so the artifact does not require AVX-512. A baseline publish (SSE only) is much slower because NativeAOT then deletes the AVX branches.

## What is real infrastructure, and what is a stub

| Piece | State |
| --- | --- |
| `FractalInfer` Qwen2 forward, BPE, CLI | Runs. Float32 CPU. No CUDA. |
| `SafetensorsHeaderParser` | Reads names and byte offsets, including sharded index files. |
| `UnbufferedFile` | Opens a file with `O_DIRECT` or `FILE_FLAG_NO_BUFFERING`. |
| `CudaNative` / `CuFileNative` | P/Invoke declarations for the CUDA Driver API and cuFile. Not a model runner. |
| `CuFileReader` | Registers a cuFile handle when the library loads. Otherwise it reads through the host and copies. |
| `ShannonEntropyScorer` | Real AVX2 histogram entropy over a byte window. |
| `GnnRouter` | A small stack-allocated scorer over patch features. It is not a trained MoE router. |
| `LlamaTransformer` | RMSNorm runs. QKV and attention are empty. The output projection, SwiGLU, and LM head are comments. |
| `FractalServe` `/v1/chat/completions` | Does not return model text. It responds `501` unless `FRACTAL_EXPERIMENTAL_STUB_CHAT=1`, in which case the old fixed word list is prefixed with `[experimental-stub, not model output]`. Startup still initializes CUDA. |
| `ModelHarvester.SquashToTernary` | Threshold packing stub. It also deletes the source file. This is not 1.58-bit quantization. |
| `ModelSwapRouter` | A PTX string and a byte-range copy. Not expert paging. |
| `ZeroAllocHnswGraph` | `CosineSimilaritySIMD` is real. Link insertion is a no-op, so search walks an empty graph. |
| `HiveMindOrchestrator` | Scripted status lines and `Task.Delay`. Not a swarm. |
| `SpeculativeDecoder.GenerateDraftTokens` | Writes `(length * 7 + i * 13) % 32000`. |
| `Fractal.Squasher` | Not on this branch. See the roadmap. |

No latency, bandwidth, or memory-footprint number in older docs was measured for a correct forward pass. Those claims are withdrawn.

## Roadmap

1. Keep this float32 CPU path correct while making the GEMM faster (weight layout, fewer bytes moved per token).
2. Run the same math on the CUDA Driver API already declared in `FractalBridge`, on an RTX 5070 Ti. The checked-in PTX GEMV is not that runner.
3. Replace `SquashToTernary` with a real 1.58-bit conversion and a kernel that consumes it. A private C/Vulkan ternary engine exists for later phases and is not used here.
4. Stream MoE experts from NVMe. `local-wip/qwen3-moe-squasher` has `split_qwen3_moe.py`, which shards a full-precision Qwen3-30B-A3B checkpoint into embeddings, layer chunks, and the LM head. The `Fractal.Squasher` project on that branch is a hello-world console app, and its `FractalServe` rewrite calls `CoreLoop` types that are not in the repo, so it does not build a model. This branch does not merge that server.
5. A "Near Content Model" (a compact geometric sketch of a teacher) is a research goal. There is no implementation.

## CI

`.github/workflows/ci.yml` restores, builds, tests, and publishes NativeAOT. The Linux job downloads Qwen2.5-0.5B-Instruct, installs PyTorch CPU and transformers for the reference script, and requires the parity test. The Windows job downloads the tokenizer only, runs the unit tests, and publishes `win-x64`.

## License

See [LICENSE](LICENSE).
