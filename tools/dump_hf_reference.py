#!/usr/bin/env python3
"""Dump float32 eager logits and greedy token ids from Hugging Face transformers.

This is a test reference only. It is not part of the NativeAOT runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Directory with config.json and model.safetensors")
    parser.add_argument("--out", required=True, help="Output directory for meta.json and logits.f32")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not input_ids:
        raise SystemExit("tokenizer produced no ids")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    model = model.to(dtype=torch.float32)
    model.eval()
    param_dtype = next(model.parameters()).dtype
    if param_dtype != torch.float32:
        raise SystemExit(f"expected float32 weights, got {param_dtype}")

    if model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 0
        model.generation_config.repetition_penalty = 1.0

    generated: list[int] = []
    past = None
    prefill_logits = None
    current = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            output = model(current, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[0, -1, :].float().cpu()
            if step == 0:
                prefill_logits = logits.contiguous().numpy()
            nxt = int(torch.argmax(logits).item())
            generated.append(nxt)
            current = torch.tensor([[nxt]], dtype=torch.long)

    if prefill_logits is None:
        raise SystemExit("no logits")

    prefill_logits.tofile(out / "logits.f32")
    probe_id = input_ids[0]
    embed_probe = model.model.embed_tokens.weight[probe_id, :8].detach().float().cpu().tolist()
    meta = {
        "prompt": args.prompt,
        "input_ids": input_ids,
        "greedy_ids": generated,
        "embed_token": probe_id,
        "embed_probe": embed_probe,
        "vocab_size": int(model.config.vocab_size),
        "dtype": "float32",
        "attn_implementation": "eager",
    }
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"wrote {out}")
    print(f"input_ids {input_ids}")
    print(f"greedy {generated}")
    print(
        f"prefill argmax {int(prefill_logits.argmax())} "
        f"max {float(prefill_logits.max())} min {float(prefill_logits.min())}"
    )


if __name__ == "__main__":
    main()
