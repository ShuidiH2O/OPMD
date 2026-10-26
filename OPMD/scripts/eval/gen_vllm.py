"""
OPMD/NTP Evaluation — vLLM Response Generation
===============================================
为指定模型在 AIME25 / DAPO / MATH-500 上生成推理结果。

用法:
    python gen_vllm.py --model_path /path/to/model --output_dir results/ \
        --tasks AIME25,DAPO,MATH-500 --gpu_ids 0,1,2,3,4,5,6,7
"""
import argparse
import json
import os
import subprocess
import sys
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

try:
    from vllm.distributed.parallel_state import destroy_model_parallel
except ImportError:
    destroy_model_parallel = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment
except ImportError:
    destroy_distributed_environment = None

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = Path("/mnt/cpfs/epic-user/zhengruize-20260612")
TEST_DATA_DIR = PROJECT_ROOT / "OPMD/datasets/test_data"

TASK_CONFIG = {
    "AIME25": {
        "path": str(TEST_DATA_DIR / "AIME25/test.parquet"),
        "n_responses": 16,
        "max_tokens": 31744,  # AIME 竞赛题需要长推理链
    },
    "DAPO": {
        "path": str(TEST_DATA_DIR / "DAPO/train.parquet"),
        "n_responses": 16,
        "max_samples": 500,
        "max_tokens": 16384,
    },
    "MATH-500": {
        "path": str(TEST_DATA_DIR / "MATH-500/test.parquet"),
        "n_responses": 16,
        "max_tokens": 4096,   # MATH-500 题目相对简单，4K tokens 足够
    },
    "Minerva": {
        "path": str(TEST_DATA_DIR / "Minerva/test.parquet"),
        "n_responses": 16,
        "max_tokens": 8192,   # Minerva 数理题，8K tokens
    },
    # coding benchmarks：greedy (temperature=0) / pass@1，用 raw 补全（不走 boxed 指令）
    "HumanEvalPlus": {
        "path": str(TEST_DATA_DIR / "HumanEvalPlus/test.parquet"),
        "n_responses": 1,
        "max_tokens": 1024,
        "code": True,
        "temperature": 0.0,
    },
    "MBPPPlus": {
        "path": str(TEST_DATA_DIR / "MBPPPlus/test.parquet"),
        "n_responses": 1,
        "max_tokens": 1024,
        "code": True,
        "temperature": 0.0,
    },
    "LiveCodeBench": {
        "path": str(TEST_DATA_DIR / "LiveCodeBench/test.parquet"),
        "n_responses": 1,
        "max_tokens": 2048,
        "code": True,
        "temperature": 0.0,
    },
    "BigCodeBench": {
        "path": str(TEST_DATA_DIR / "BigCodeBench/test.parquet"),
        "n_responses": 1,
        "max_tokens": 2048,
        "code": True,
        "temperature": 0.0,
    },
    "SantaCoderFIM": {
        "path": str(TEST_DATA_DIR / "SantaCoderFIM/test.parquet"),
        "n_responses": 1,
        "max_tokens": 256,
        "code": True,
        "temperature": 0.0,
    },
}

PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."
MAX_TOKENS = 16384  # default, overridable via --max_tokens
TEMPERATURE = 0.7
TOP_P = 0.95


def load_samples(filepath: str, max_samples: int = None):
    """读取 parquet 文件，返回 sample 列表。"""
    df = pd.read_parquet(filepath)
    samples = []
    for i in range(len(df)):
        prompt_content = df.at[i, "prompt"][0]["content"].strip()
        ground_truth = df.at[i, "reward_model"]["ground_truth"].strip()
        index = ""
        if "extra_info" in df.columns:
            try:
                index = str(df.at[i, "extra_info"]["index"])
            except Exception:
                index = str(i)
        samples.append({
            "example_id": i,
            "prompt": prompt_content,
            "answer": ground_truth,
            "index": index,
        })
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    return samples


def _normalize_cuda_visible_devices():
    """将 CUDA_VISIBLE_DEVICES 中的 GPU UUID 转换为索引格式。
    
    某些平台（如 EPIC）通过 UUID 设置 CUDA_VISIBLE_DEVICES，但 vLLM 0.23.0
    要求使用整数索引。此函数检测 UUID 格式并自动转换为索引。
    
    Returns:
        int: 可用 GPU 数量
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return torch.cuda.device_count()
    
    devices = [d.strip() for d in cvd.split(",")]
    
    # 检查是否已经是整数索引格式
    if all(d.isdigit() for d in devices):
        return torch.cuda.device_count()
    
    # 检查是否包含 UUID (格式: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    if not any(d.startswith("GPU-") for d in devices):
        return torch.cuda.device_count()
    
    # 使用 nvidia-smi 获取 UUID -> 索引映射
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        uuid_to_idx = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                uuid_to_idx[parts[0]] = parts[1]
        
        # 转换 UUID 为索引
        indices = []
        for dev in devices:
            if dev in uuid_to_idx:
                indices.append(uuid_to_idx[dev])
            elif dev.isdigit():
                indices.append(dev)
            else:
                print(f"WARNING: Cannot resolve GPU identifier: {dev}, skipping.")
        
        if indices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
            print(f"Converted GPU UUIDs to indices: {','.join(indices)} "
                  f"(original: {','.join(devices[:2])}...)")
        return len(indices)
    except Exception as e:
        print(f"WARNING: Failed to convert GPU UUIDs: {e}")
        return torch.cuda.device_count()


def main():
    global MAX_TOKENS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="AIME25,DAPO,MATH-500")
    parser.add_argument("--gpu_ids", type=str, default="auto",
                        help="Comma-separated GPU indices, or 'auto' to auto-detect all available GPUs")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Max model context length (default: auto-detect from model config)")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--no_chat_template", action="store_true", default=False,
                        help="Use raw text prompts instead of chat template (for NTP models)")
    parser.add_argument("--replace", action="store_true", default=False)
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                        help=f"Max tokens to generate (default: {MAX_TOKENS})")
    parser.add_argument("--n_responses", type=int, default=None,
                        help="Override number of responses per question (default: task-specific value)")
    args = parser.parse_args()

    MAX_TOKENS = args.max_tokens

    tasks = [t.strip() for t in args.tasks.split(",")]

    # GPU 检测: 优先使用平台已设置的环境变量，不覆盖
    if args.gpu_ids == "auto":
        # 转换 UUID 格式为索引（修复 vLLM 0.23.0 兼容性）
        gpu_count = _normalize_cuda_visible_devices()
        if gpu_count == 0:
            gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print("ERROR: No GPUs detected. Ensure CUDA_VISIBLE_DEVICES is set correctly.")
            sys.exit(1)

        # 自动选取合法的 TP_SIZE：num_attention_heads 必须能被 TP_SIZE 整除
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        num_heads = getattr(model_config, "num_attention_heads", None) or getattr(model_config, "num_key_value_heads", None)
        if num_heads is not None:
            valid_tp_sizes = [tp for tp in range(1, gpu_count + 1) if num_heads % tp == 0]
            if not valid_tp_sizes:
                print(f"ERROR: No valid tensor_parallel_size for model with {num_heads} heads "
                      f"using {gpu_count} GPUs. Supported sizes: {[s for s in [1,2,4,8,16] if s <= gpu_count]}")
                sys.exit(1)
            tensor_parallel_size = valid_tp_sizes[-1]  # 取最大的合法值
            if tensor_parallel_size < gpu_count:
                print(f"Auto-detected {gpu_count} GPUs, but model has {num_heads} heads. "
                      f"Using TP_SIZE={tensor_parallel_size} (max valid ≤ {gpu_count})")
            else:
                print(f"Auto-detected {tensor_parallel_size} GPUs (platform allocation)")
        else:
            tensor_parallel_size = gpu_count
            print(f"Auto-detected {tensor_parallel_size} GPUs (platform allocation)")
    else:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",")]
        tensor_parallel_size = len(gpu_ids)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        print(f"Using {tensor_parallel_size} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型 (单实例, tensor parallel)
    print(f"Loading model: {args.model_path}")
    llm_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
        print(f"Max model length: {args.max_model_len}")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    stop_token_ids = []
    # 使用 tokenizer 的实际 EOS token ID（而非硬编码字符串匹配）
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    # 额外处理 Qwen 的 <|im_end|>（其可能不等同于 eos_token_id）
    for stop_token in ["<|im_end|>"]:
        try:
            encoded = tokenizer.encode(stop_token, add_special_tokens=False)
            if encoded and len(encoded) == 1:
                stop_token_ids.append(encoded[0])
        except Exception:
            pass

    try:
        for task_name in tasks:
            if task_name not in TASK_CONFIG:
                print(f"Unknown task: {task_name}, skipping.")
                continue

            cfg = TASK_CONFIG[task_name]
            n_responses = args.n_responses if args.n_responses is not None else cfg["n_responses"]
            task_max_tokens = cfg.get("max_tokens", MAX_TOKENS)
            is_code = cfg.get("code", False)
            temperature = cfg.get("temperature", TEMPERATURE)
            top_p = cfg.get("top_p", TOP_P)
            print(f"\n{'='*60}\nTask: {task_name} (N={n_responses}, max_tokens={task_max_tokens}, code={is_code})")
            print(f"Model: {args.model_path}")
            print(f"{'='*60}")

            suffix = "raw" if (is_code or args.no_chat_template) else "chat"
            out_path = out_dir / f"{task_name.lower()}_t{temperature}_p{top_p}_n{n_responses}-MNT{task_max_tokens}_{suffix}.jsonl"

            if not args.replace and out_path.exists():
                # 检查是否已全部完成（断点续传）
                completed_seeds = set()
                with open(out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            completed_seeds.add(json.loads(line)["seed"])
                        except Exception:
                            pass
                n_expected = n_responses
                if len(completed_seeds) >= n_expected:
                    print(f"Result exists and complete: {out_path}, skipping.")
                    continue
                else:
                    print(f"Result exists but incomplete ({len(completed_seeds)}/{n_expected} rollouts done), resuming...")

            samples = load_samples(cfg["path"], max_samples=cfg.get("max_samples"))

            if is_code or args.no_chat_template:
                # coding / NTP：原始文本补全，不套 math 的 boxed 指令，也不走 chat template
                formatted_prompts = [s["prompt"] for s in samples]
            else:
                # math chat：追加 instruction + chat template
                for s in samples:
                    s["prompt"] = PROMPT_TEMPLATE.format(problem=s["prompt"])
                formatted_prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": s["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                    for s in samples
                ]

            if samples:
                print(f"Example prompt: {samples[0]['prompt'][:200]}...")

            # 批量生成: 分批提交，避免 KV cache 溢出
            task_max_tokens = cfg.get("max_tokens", MAX_TOKENS)  # 任务特定的 max_tokens
            batch_size = max(1, 120 // len(samples))  # 每批最多 120 条请求
            num_batches = (n_responses + batch_size - 1) // batch_size

            # 断点续传：检查已有结果，跳过已完成的 rollout
            completed_rollouts = set()
            if not args.replace and out_path.exists():
                with open(out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            completed_rollouts.add(data["seed"])
                        except Exception:
                            pass
                if completed_rollouts:
                    print(f"Resuming from existing results: {len(completed_rollouts)}/{n_responses} rollouts already done")

            all_results = []
            # 先加载已有结果
            if completed_rollouts:
                with open(out_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            all_results.append(json.loads(line))
                        except Exception:
                            pass

            for batch_idx in range(num_batches):
                start_rollout = batch_idx * batch_size
                end_rollout = min(n_responses, (batch_idx + 1) * batch_size)
                batch_rollouts = [r for r in range(start_rollout, end_rollout) if r not in completed_rollouts]

                if not batch_rollouts:
                    continue

                batch_prompts = []
                batch_params = []
                for rollout_id in batch_rollouts:
                    batch_prompts.extend(formatted_prompts)
                    batch_params.extend([
                        SamplingParams(
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=task_max_tokens,
                            stop_token_ids=stop_token_ids if stop_token_ids else None,
                        )
                        for _ in samples
                    ])

                batch_total = len(batch_prompts)
                print(f"Batch {batch_idx+1}/{num_batches}: {batch_total} requests (rollouts {batch_rollouts[0]}-{batch_rollouts[-1]})...")
                outputs = llm.generate(batch_prompts, batch_params, use_tqdm=True)

                idx = 0
                for rollout_id in batch_rollouts:
                    for sample in samples:
                        all_results.append({
                            "example_id": sample["example_id"],
                            "prompt": sample["prompt"],
                            "answer": sample["answer"],
                            "index": sample.get("index", ""),
                            "seed": rollout_id,
                            "response": outputs[idx].outputs[0].text,
                        })
                        idx += 1

                # 每完成一个 batch 就写入文件（断点续传保护）
                with open(out_path, "w", encoding="utf-8") as f:
                    for item in all_results:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Generated {len(all_results)} responses for {task_name}")
            print(f"Saved: {out_path}")

    finally:
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()