"""
Merge verl FSDP checkpoint into a single HuggingFace model.

Usage:
    python OPMD/scripts/merge_checkpoint.py \
        --checkpoint_path OPMD/checkpoints/<run>/global_step_400/actor \
        --output_path /tmp/opmd_ckpt_400_hf

The output can be loaded directly by vLLM or transformers.
"""
import argparse
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _to_plain_tensor(t):
    """Convert DTensor to plain torch.Tensor (local shard)."""
    if type(t).__name__ == "DTensor":
        return t._local_tensor
    return t


def merge_fsdp_checkpoint(checkpoint_path: str, output_path: str):
    """
    Load verl FSDP checkpoint (sharded across ranks) and save as HuggingFace model.

    The checkpoint directory should contain:
        model_world_size_{N}_rank_{0..N-1}.pt  — sharded model weights (DTensor)
        huggingface/                             — config & tokenizer files
    """
    ckpt_dir = Path(checkpoint_path)
    hf_dir = ckpt_dir / "huggingface"

    if not hf_dir.exists():
        raise FileNotFoundError(f"huggingface/ not found in {checkpoint_path}")

    # Find all rank files
    model_files = sorted(ckpt_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_world_size_*_rank_*.pt found in {checkpoint_path}")

    # Extract world_size from first filename
    import re
    match = re.match(r"model_world_size_(\d+)_rank_(\d+)\.pt", model_files[0].name)
    world_size = int(match.group(1))
    num_ranks = len(model_files)
    print(f"Found {num_ranks} rank files, world_size={world_size}")

    # Load all shards
    print("Loading rank shards...")
    shards = []
    for rank in range(world_size):
        shard_path = ckpt_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing rank file: {shard_path}")
        shard = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        shards.append(shard)
        print(f"  Rank {rank}: {len(shard)} keys")

    # Load model config and build model FIRST to get expected shapes
    config = AutoConfig.from_pretrained(str(hf_dir), trust_remote_code=True)
    print("Building model from config...")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Build expected shape mapping from model state dict
    model_state = model.state_dict()
    expected_shapes = {k: v.shape for k, v in model_state.items()}

    # Merge: for each key, compare against model expected shape to decide
    # whether individual shards are local FSDP chunks (need concatenation) or
    # replicated full tensors (take the first).
    print("Merging shards...")
    all_keys = sorted(set().union(*[set(s.keys()) for s in shards]))

    merged_state = {}
    for key in all_keys:
        # Collect local tensors from all ranks
        locals_list = []
        for shard in shards:
            if key in shard:
                locals_list.append(_to_plain_tensor(shard[key]))
            else:
                locals_list.append(None)

        # Filter out None
        valid = [t for t in locals_list if t is not None]

        if len(valid) == 0:
            continue

        if len(valid) == 1:
            merged_state[key] = valid[0].clone()
            continue

        # Compare shard shape against expected model shape
        expected_shape = expected_shapes.get(key)
        if expected_shape is not None and valid[0].shape != expected_shape:
            # Shard shape != model shape → FSDP sharding, concatenate
            merged_state[key] = torch.cat([t.clone() for t in valid], dim=0)
        else:
            # Shard shape matches model shape → replicated, take the first
            merged_state[key] = valid[0].clone()

    print(f"Merged state dict: {len(merged_state)} keys")

    print("Loading weights into model...")
    missing, unexpected = model.load_state_dict(merged_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    - {k}")

    # Save as HuggingFace format
    print(f"Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, max_shard_size="5GB", safe_serialization=True)

    # Copy tokenizer/config files from huggingface directory
    for fname in os.listdir(str(hf_dir)):
        src = hf_dir / fname
        dst = Path(output_path) / fname
        if src.is_file() and not dst.exists():
            shutil.copy2(str(src), str(dst))

    print(f"Done! Merged model saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge verl checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the actor checkpoint directory (contains model_world_size_*_rank_*.pt files)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory for the merged HuggingFace model")
    args = parser.parse_args()

    merge_fsdp_checkpoint(args.checkpoint_path, args.output_path)


if __name__ == "__main__":
    main()