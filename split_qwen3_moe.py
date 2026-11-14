import os
import json
import torch
from safetensors.torch import load_file, save_file

def split_qwen3_moe(model_dir, output_dir, chunk_size_layers=6):
    """
    Splits Qwen3-30B-A3B at full precision into granular, hot-swappable sub-models:
    - Embeddings & Norms
    - Layer chunks (containing self-attn and MoE expert blocks)
    - Output lm_head
    """
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    state_dict = {}
    
    if os.path.exists(index_path):
        print(f"[-] Reading weight map from index: {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        
        shard_files = sorted(list(set(weight_map.values())))
        print(f"[-] Loading {len(shard_files)} shards into memory...")
        for shard in shard_files:
            shard_path = os.path.join(model_dir, shard)
            if os.path.exists(shard_path):
                print(f"    -> Loading shard: {shard}")
                state_dict.update(load_file(shard_path))
            else:
                print(f"[!] Warning: Shard {shard} listed in index but missing on disk.")
    else:
        # Fallback if unindexed single file
        single_file = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_file):
            print(f"[-] Loading monolithic file: {single_file}")
            state_dict = load_file(single_file)
        else:
            raise FileNotFoundError(f"No valid model structure found in {model_dir}")

    print(f"[+] Total tensors loaded: {len(state_dict)}")

    embeddings_dict = {}
    output_head_dict = {}
    layer_groups = {}

    for key, tensor in state_dict.items():
        # 1. Capture Embeddings & Initial Norms
        if "embed_tokens" in key or "embedding" in key:
            embeddings_dict[key] = tensor
        # 2. Capture Final Output Head & Final Norm
        elif "lm_head" in key or "model.norm" in key:
            output_head_dict[key] = tensor
        # 3. Parse Transformer Blocks (0 to 47 for Qwen3-30B)
        elif "model.layers." in key:
            parts = key.split(".")
            try:
                layer_idx_pos = parts.index("layers") + 1
                layer_idx = int(parts[layer_idx_pos])
                group_id = layer_idx // chunk_size_layers
                
                if group_id not in layer_groups:
                    layer_groups[group_id] = {}
                layer_groups[group_id][key] = tensor
            except (ValueError, IndexError):
                embeddings_dict[key] = tensor
        else:
            embeddings_dict[key] = tensor

    # Save Embeddings
    if embeddings_dict:
        out_path = os.path.join(output_dir, "submodel_00_embeddings.safetensors")
        save_file(embeddings_dict, out_path)
        print(f"[>] Saved Embeddings Block -> {out_path} ({len(embeddings_dict)} tensors)")

    # Save Layer Chunks (e.g., 8 chunks of 6 layers each to cover 48 layers)
    for group_id, tensors in sorted(layer_groups.items()):
        out_path = os.path.join(output_dir, f"submodel_layers_chunk_{group_id:02d}.safetensors")
        save_file(tensors, out_path)
        print(f"[>] Saved Layer Chunk {group_id} -> {out_path} ({len(tensors)} tensors)")

    # Save Output Head
    if output_head_dict:
        out_path = os.path.join(output_dir, "submodel_99_output_head.safetensors")
        save_file(output_head_dict, out_path)
        print(f"[>] Saved Output Head -> {out_path} ({len(output_head_dict)} tensors)")

    print("[*] Qwen3-30B-A3B modular split completed successfully!")

if __name__ == "__main__":
    SOURCE_MODEL = r"C:\Fractal-BLT\Models\qwen3_30b_a3b"
    OUTPUT_TARGET = r"C:\Fractal-BLT\Models\qwen3_30b_modular"
    
    # Group layers into chunks of 6 (48 total layers / 6 = 8 sub-files)
    split_qwen3_moe(SOURCE_MODEL, OUTPUT_TARGET, chunk_size_layers=6)