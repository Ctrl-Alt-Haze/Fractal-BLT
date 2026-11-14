---
layout: default
title: Fractal-BLT Documentation
---

# Fractal-BLT

The working inference path is a float32 CPU Qwen2.5-0.5B-Instruct forward pass (`FractalInfer`). Older pages in this site describe MoE streaming, a hive of agents, and microsecond retrieval that the code does not do. Read [the README](https://github.com/Ctrl-Alt-Haze/Fractal-BLT/blob/master/README.md) and [Architecture](ARCHITECTURE.md) for the current split between real code and experimental stubs.

## What is actually in the tree
1. **Qwen2 CPU runtime.** Safetensors in, BPE tokenization, RMSNorm, RoPE, grouped-query attention, SwiGLU, greedy or temperature sampling.
2. **Parsers and P/Invoke.** Safetensors headers, unbuffered file opens, and CUDA Driver API declarations. They are not a full GPU model.
3. **Experimental stubs.** The HTTP chat route, HNSW graph, 1.58-bit squasher, TMA router, hive orchestrator, and speculative draft are labeled as such. The chat route does not return model text.

## Guides
- [Getting Started](getting_started.md)
- [SDK Reference](sdk_reference.md)
- [Agentic Configuration](agents.md)
