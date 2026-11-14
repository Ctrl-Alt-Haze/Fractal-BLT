# Architecture

This note describes the code in the repository. It is not a claim that every subsystem below runs a model.

## CPU Qwen2 path

`FractalInfer` loads one safetensors file and runs a Qwen2 decoder in float32.

1. `SafetensorsFile` reads the 8-byte header length, the JSON header, and the raw little-endian payload. BF16 and FP16 values are widened to float32. FP32 is copied. The compute dtype stays float32.
2. `Qwen2Config` accepts `model_type: qwen2`, SiLU, and `use_sliding_window: false`. Qwen2.5-0.5B-Instruct matches that. The Qwen3-MoE config under `Models/qwen3_30b_a3b` does not, and the loader rejects it.
3. `BpeTokenizer` reads `tokenizer.json`: NFC when the file says so, longest-match added tokens, the Qwen pretokenizer regex, GPT-2 byte-to-unicode, then BPE merges in rank order. `ApplyQwen25Instruct` covers the no-tools chat template (default system prompt, one user turn, generation prefix). It is not a Jinja interpreter.
4. `Qwen2Model.Forward` embeds the new tokens, then for each layer: RMSNorm, Q/K/V projections (Q, K, and V have bias; the output projection and the MLP do not), NeoX RoPE (`inv_freq` duplicated across the two halves, `rope_theta` from config), writes K and V into a cache, runs causal grouped-query attention with scale `1/sqrt(head_dim)`, adds the residual, then SwiGLU (`silu(gate) * up`, then down), then the second residual. A final RMSNorm and the LM head produce logits for the last token only. If `lm_head.weight` is absent and `tie_word_embeddings` is true, the embedding matrix is the LM head.
5. Greedy decoding is the first maximum (ties keep the lower index, matching `torch.argmax`). Temperature sampling is a multinomial draw from `softmax(logits / temperature)`.

Matrix multiplies are row-wise dot products of a PyTorch linear weight `[out, in]` with each token. Rows run in parallel. Each dot is sequential, using AVX-512 FMA, AVX2, or a scalar tail. That keeps the result stable across thread counts on one CPU. It is not a GPU kernel. NativeAOT evaluates `IsSupported` when publishing. The CLI defaults to the build CPU (`IlcInstructionSet=native`). A baseline publish keeps SSE only and drops the AVX loops.

The KV cache is sized to the context passed to `Load` (the CLI default is 2048). One `Qwen2Model` is not safe for concurrent `Forward` calls. Activations are reused across steps; token encoding still allocates.

## Pieces that exist and do what their names say

- `FractalStreamer.SafetensorsHeaderParser` resolves a tensor name to a file, offset, and length. It does not decode dtypes. `FractalInfer.SafetensorsFile` does.
- `UnbufferedFile.OpenUnbuffered` is a direct open (`O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` on Windows).
- `FractalBridge.CudaNative` is a partial CUDA Driver API surface (`cuInit`, contexts, streams, `cuMemAlloc`, async memcpy, module load, `cuLaunchKernel`) plus a resolver that maps the import onto `libcuda.so` on Linux. Calling it requires a driver. Nothing in `FractalInfer` calls it.
- `ShannonEntropyScorer` computes a 256-bin Shannon entropy with a stack histogram.

## Experimental code

These types are still compiled. They are not the inference path.

- `LlamaTransformer` normalizes and can rotate pairs. `ComputeQKV` and `SelfAttention` return immediately, so the residual stream never receives attention or an MLP, and the method never writes LM-head logits.
- `FractalServe` builds a Kestrel app, patches bytes with the entropy scorer, and can launch a PTX GEMV when CUDA is present. The strings it used to return were a fixed list of about fifty words indexed by a hash. The route now returns HTTP 501. `FRACTAL_EXPERIMENTAL_STUB_CHAT=1` restores that list and prefixes it so it cannot be read as a completion. Process startup still calls `CudaNative.Init`.
- `ModelHarvester.HarvestModelAsync` downloads a single URL. `SquashToTernary` packs raw 16-bit words with a threshold and then deletes the input file. It does not compute absolute-mean scales, and the output is not a safetensors file despite the extension.
- `ModelSwapRouter` embeds a `cp.async.bulk.tensor` PTX string. `HotSwapToModel` copies a span of the file onto a device pointer. It does not parse experts or keep a KV cache warm.
- `ZeroAllocHnswGraph.CosineSimilaritySIMD` is an AVX2 cosine. `AddVector` copies a vector and does not create edges. Search starts at node 0 and follows zeros.
- `HiveMindOrchestrator` copies a prompt into an unmanaged struct and reports scripted stages separated by delays.
- `SpeculativeDecoder` fills draft ids from an arithmetic sequence. `VerifyDraft` can argmax a provided logit buffer; nothing in the real Qwen2 path calls it.
- `GnnRouter.ComputeRoutes` builds a tiny adjacency from patch length and entropy and writes an expert id. The registry it is given in the server is random.

## Earlier claims that the code does not support

- There is no measured 2.4 µs retrieval, 64 GB/s TMA stream, 10 MB server, or zero-allocation tokenizer in the hot path.
- Weights are not paged from NVMe into an attention or MoE forward.
- There is no 1.58-bit model file that this runtime can execute.
- There is no Near Content Model.

## Next

The CPU path is the base for a CUDA port on an RTX 5070 Ti, a real ternary format, and MoE streaming. Sharding experiments for Qwen3-30B-A3B live on `local-wip/qwen3-moe-squasher` (`split_qwen3_moe.py`) and are not wired into `Qwen2Model`. A separate private C/Vulkan ternary engine is a later source of kernels, not a dependency of this tree.
