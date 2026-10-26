"""
Anchor Token Selection via Proxy Model Entropy
==============================================
使用 DeepSeek-R1-Distill-Qwen-1.5B 作为代理模型，扫描 MegaMath-Web-Pro 数据集，
计算每个 token 的预测熵，选取高熵 token 作为 Anchor Token。

用法:
    # 单卡
    python OPMD/scripts/preprocess/anchor_selection.py
    # 多卡（6 卡并行）
    python OPMD/scripts/preprocess/anchor_selection.py --num_gpus 6

输出:
    OPMD/datasets/megamath_web_pro_1B_anchors.parquet
    列: text (原始文本), anchor_positions (token位置列表), entropy_values (熵值列表)
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 配置
# ============================================================
# 代理模型路径
PROXY_MODEL_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/DeepSeek-R1-Distill-Qwen-1.5B"

# 数据集路径
DATASET_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B.jsonl"

# 输出路径
OUTPUT_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B_anchors.parquet"

# Anchor 选取比例（top-N% 高熵 token）
ANCHOR_RATIO = 0.15  # 默认 top-15%

# 代理模型推理参数
MAX_SEQ_LENGTH = 4096  # 单次处理的最大序列长度
BATCH_SIZE = 1  # 代理模型推理 batch size（1.5B 模型单卡可处理）

# 最大处理文档数（None = 全部处理）
MAX_DOCS: Optional[int] = None

# 临时熵值缓存文件（用于断点续传）
ENTROPY_CACHE_DIR = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/entropy_cache"

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
# 辅助函数
# ============================================================
def compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    从 logits 计算每个 token 位置的预测熵。
    
    Args:
        logits: (batch_size, seq_len, vocab_size) 或 (seq_len, vocab_size)
    
    Returns:
        entropy: (batch_size, seq_len) 或 (seq_len,)
    """
    # Softmax 得到概率分布
    probs = torch.softmax(logits, dim=-1)
    # 计算熵: H = -sum(p * log(p))
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def compute_entropy_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_length: int = 4096,
) -> list[list[float]]:
    """
    对一批文本计算每个 token 的预测熵。
    
    对于超过 max_seq_length 的文本，采用滑动窗口分块处理。
    每个 token 的熵取其所在窗口的预测值。
    
    Args:
        model: 代理模型
        tokenizer: tokenizer
        texts: 文本列表
        device: 设备
        max_seq_length: 最大序列长度
    
    Returns:
        entropies: 每个文本的 token 级熵值列表 (list of list of float)
    """
    model.eval()
    results = []

    for text in texts:
        # Tokenize 整个文本
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"][0]  # (seq_len,)
        total_len = len(input_ids)

        if total_len == 0:
            results.append([])
            continue

        # 初始化熵值数组
        token_entropies = torch.full((total_len,), float("nan"), device=device)

        # 分块处理
        stride = max_seq_length // 2  # 50% 重叠
        for start in range(0, total_len, stride):
            end = min(start + max_seq_length, total_len)
            chunk = input_ids[start:end].unsqueeze(0).to(device)  # (1, chunk_len)
            chunk_len = end - start

            with torch.no_grad():
                outputs = model(chunk, use_cache=False)
                # logits: (1, chunk_len, vocab_size)
                logits = outputs.logits[0]  # (chunk_len, vocab_size)

            # 计算熵
            chunk_entropy = compute_entropy_from_logits(logits)  # (chunk_len,)

            # 填充到对应的位置
            # 对于重叠区域，取平均值（或第一次出现的值）
            chunk_start_in_full = start
            chunk_end_in_full = start + chunk_len

            for i in range(chunk_len):
                pos = chunk_start_in_full + i
                if torch.isnan(token_entropies[pos]):
                    token_entropies[pos] = chunk_entropy[i]
                else:
                    # 重叠区域取平均
                    token_entropies[pos] = (token_entropies[pos] + chunk_entropy[i]) / 2.0

        results.append(token_entropies.cpu().tolist())

    return results


def select_anchor_positions(
    entropies: list[float],
    anchor_ratio: float = 0.2,
) -> tuple[list[int], list[float]]:
    """
    根据熵值选取 anchor token 位置。
    
    Args:
        entropies: token 级熵值列表
        anchor_ratio: 选取比例
    
    Returns:
        anchor_positions: anchor token 的 token 位置列表
        anchor_entropies: 对应的熵值列表
    """
    if len(entropies) == 0:
        return [], []

    # 计算阈值：top anchor_ratio 的熵值
    n_anchors = max(1, int(len(entropies) * anchor_ratio))
    entropy_array = np.array(entropies)
    threshold = np.partition(entropy_array, -n_anchors)[-n_anchors]

    # 选取熵值 >= threshold 的位置
    positions = []
    values = []
    for i, ent in enumerate(entropies):
        if ent >= threshold:
            positions.append(i)
            values.append(ent)

    # 按位置排序
    sorted_pairs = sorted(zip(positions, values), key=lambda x: x[0])
    positions = [p for p, _ in sorted_pairs]
    values = [v for _, v in sorted_pairs]

    return positions, values


def save_entropy_cache(
    doc_idx: int,
    text: str,
    entropies: list[float],
    cache_dir: str,
):
    """保存单个文档的熵值缓存（用于断点续传）。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"doc_{doc_idx:07d}.pt")
    torch.save(
        {
            "doc_idx": doc_idx,
            "text": text,
            "entropies": entropies,
        },
        cache_path,
    )


def load_entropy_cache(doc_idx: int, cache_dir: str) -> Optional[dict]:
    """加载单个文档的熵值缓存（跳过损坏文件）。"""
    cache_path = os.path.join(cache_dir, f"doc_{doc_idx:07d}.pt")
    if os.path.exists(cache_path):
        # 跳过空文件
        if os.path.getsize(cache_path) == 0:
            return None
        try:
            return torch.load(cache_path, weights_only=False)
        except Exception:
            return None
    return None


# ============================================================
# 多 GPU 并行 Worker
# ============================================================
def _get_available_gpu_ids(num_gpus: int) -> list[int]:
    """
    从环境变量 CUDA_VISIBLE_DEVICES 读取实际可用的 GPU ID 列表。

    租约平台通过 CUDA_VISIBLE_DEVICES 注入 GPU 信息。支持两种格式：
    - 数字索引: "0,1,2,3,4,5"
    - GPU UUID: "GPU-xxx,GPU-yyy,..."

    Returns:
        gpu_ids: 物理 GPU 索引列表 (如 [2, 3, 4, 5, 6, 7])
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        if "GPU-" in cuda_visible:
            # UUID 格式：通过 nvidia-smi 获取物理索引
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                uuid_to_idx = {}
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2:
                        uuid_to_idx[parts[1]] = int(parts[0])
                gpu_ids = []
                for uuid_str in cuda_visible.split(","):
                    uuid_str = uuid_str.strip()
                    if uuid_str in uuid_to_idx:
                        gpu_ids.append(uuid_to_idx[uuid_str])
                if not gpu_ids:
                    gpu_ids = list(range(num_gpus))
            except Exception:
                gpu_ids = list(range(num_gpus))
        else:
            gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
    else:
        gpu_ids = list(range(num_gpus))

    if len(gpu_ids) < num_gpus:
        logger.warning(
            f"Requested {num_gpus} GPUs but CUDA_VISIBLE_DEVICES only has "
            f"{len(gpu_ids)}: {gpu_ids}. Using {len(gpu_ids)} GPUs."
        )
        return gpu_ids

    return gpu_ids[:num_gpus]


def _gpu_worker(
    gpu_id: int,
    shard_texts: list[str],
    shard_indices: list[int],
    proxy_model_path: str,
    max_seq_length: int,
    anchor_ratio: float,
    output_dir: str,
    resume: bool,
) -> str:
    """
    单个 GPU 的 worker 进程：加载模型，处理分片文档，输出分片 parquet。

    Args:
        gpu_id: 实际物理 GPU 编号（从 CUDA_VISIBLE_DEVICES 解析，如 2, 3, 5, ...）
        shard_texts: 分配给该 GPU 的文本列表
        shard_indices: 对应的原始文档索引
        proxy_model_path: 代理模型路径
        max_seq_length: 最大序列长度
        anchor_ratio: anchor 选取比例
        output_dir: 临时输出目录
        resume: 是否从缓存恢复

    Returns:
        shard_parquet_path: 分片 parquet 文件路径
    """
    # 只暴露指定的物理 GPU 给当前 worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 重新导入以确保 CUDA 可见性生效
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModel, AutoTokenizer as _AutoTokenizer

    device = _torch.device("cuda:0")

    worker_logger = logging.getLogger(f"GPU-{gpu_id}")
    worker_logger.info(f"Worker {gpu_id}: processing {len(shard_texts)} docs on {device}")

    # 加载模型
    tokenizer = _AutoTokenizer.from_pretrained(proxy_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _AutoModel.from_pretrained(
        proxy_model_path,
        dtype=_torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    worker_logger.info(f"Worker {gpu_id}: model loaded, starting entropy computation...")

    # 处理文档
    all_entropies = []
    start_time = time.time()

    for local_i, (text, orig_idx) in enumerate(zip(shard_texts, shard_indices)):
        # 断点续传：检查缓存
        if resume:
            cache = load_entropy_cache(orig_idx, ENTROPY_CACHE_DIR)
            if cache is not None:
                all_entropies.append(cache["entropies"])
                if (local_i + 1) % 100 == 0 or local_i == 0:
                    worker_logger.info(
                        f"Worker {gpu_id}: [{local_i+1}/{len(shard_texts)}] "
                        f"(cached) doc {orig_idx}"
                    )
                continue

        try:
            entropies = compute_entropy_batch(
                model=model,
                tokenizer=tokenizer,
                texts=[text],
                device=device,
                max_seq_length=max_seq_length,
            )[0]
            all_entropies.append(entropies)
            save_entropy_cache(orig_idx, text, entropies, ENTROPY_CACHE_DIR)
        except Exception as e:
            worker_logger.error(f"Worker {gpu_id}: error on doc {orig_idx}: {e}")
            all_entropies.append([])

        if (local_i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            worker_logger.info(
                f"Worker {gpu_id}: [{local_i+1}/{len(shard_texts)}] "
                f"doc {orig_idx}, {elapsed:.0f}s elapsed"
            )

    elapsed = time.time() - start_time
    worker_logger.info(
        f"Worker {gpu_id}: entropy computation done in {elapsed:.0f}s "
        f"({elapsed/len(shard_texts):.2f}s/doc)"
    )

    # 选取 anchor
    worker_logger.info(f"Worker {gpu_id}: selecting anchors (ratio={anchor_ratio})...")
    records = []
    for local_i, (text, entropies) in enumerate(zip(shard_texts, all_entropies)):
        if len(entropies) == 0:
            continue
        anchor_positions, anchor_entropies = select_anchor_positions(entropies, anchor_ratio)
        if len(anchor_positions) > 0:
            records.append({
                "text": text,
                "anchor_positions": anchor_positions,
                "entropy_values": anchor_entropies,
            })

    # 保存分片结果
    os.makedirs(output_dir, exist_ok=True)
    shard_path = os.path.join(output_dir, f"anchors_shard_{gpu_id:02d}.parquet")
    df = pd.DataFrame(records)
    df.to_parquet(shard_path, index=False)

    total_anchors = sum(len(r["anchor_positions"]) for r in records)
    worker_logger.info(
        f"Worker {gpu_id}: saved {len(records)} docs, {total_anchors} anchors → {shard_path}"
    )
    return shard_path


def _gpu_worker_wrapper(
    gpu_id: int,
    shard_texts: list[str],
    shard_indices: list[int],
    proxy_model_path: str,
    max_seq_length: int,
    anchor_ratio: float,
    output_dir: str,
    resume: bool,
    result_queue,
):
    """模块级 wrapper，将结果放入队列（用于 multiprocessing spawn）。"""
    try:
        shard_path = _gpu_worker(
            gpu_id=gpu_id,
            shard_texts=shard_texts,
            shard_indices=shard_indices,
            proxy_model_path=proxy_model_path,
            max_seq_length=max_seq_length,
            anchor_ratio=anchor_ratio,
            output_dir=output_dir,
            resume=resume,
        )
        result_queue.put((gpu_id, shard_path, None))
    except Exception as e:
        logging.getLogger(__name__).error(f"GPU {gpu_id} worker failed: {e}", exc_info=True)
        result_queue.put((gpu_id, None, str(e)))


def _launch_multi_gpu(
    texts: list[str],
    num_gpus: int,
    proxy_model_path: str,
    max_seq_length: int,
    anchor_ratio: float,
    output_path: str,
    resume: bool,
) -> None:
    """
    启动多 GPU 并行处理，合并分片结果。

    Args:
        texts: 全部文本列表
        num_gpus: GPU 数量
        proxy_model_path: 代理模型路径
        max_seq_length: 最大序列长度
        anchor_ratio: anchor 选取比例
        output_path: 最终输出路径
        resume: 是否从缓存恢复
    """
    # 获取实际可用的 GPU ID（从 CUDA_VISIBLE_DEVICES 解析）
    gpu_ids = _get_available_gpu_ids(num_gpus)
    num_gpus = len(gpu_ids)
    logger.info(f"Available GPU IDs: {gpu_ids}")

    # 分片
    shard_size = (len(texts) + num_gpus - 1) // num_gpus
    shards = []
    for i, gpu_id in enumerate(gpu_ids):
        start = i * shard_size
        end = min(start + shard_size, len(texts))
        if start >= len(texts):
            break
        shards.append((
            gpu_id,
            texts[start:end],
            list(range(start, end)),
        ))

    logger.info(f"Splitting {len(texts)} docs across {len(shards)} GPUs "
                f"(~{shard_size} docs each)")

    # 使用临时目录存放分片结果
    output_dir = os.path.dirname(output_path)
    shard_dir = os.path.join(output_dir, "anchor_shards")
    os.makedirs(shard_dir, exist_ok=True)

    # 启动 worker 进程
    # 使用 spawn 方式确保每个进程独立初始化 CUDA
    ctx = mp.get_context("spawn")
    processes = []
    results = ctx.Queue()

    for gpu_id, shard_texts, shard_indices in shards:
        p = ctx.Process(
            target=_gpu_worker_wrapper,
            args=(gpu_id, shard_texts, shard_indices, proxy_model_path,
                  max_seq_length, anchor_ratio, shard_dir, resume, results),
        )
        p.start()
        processes.append(p)

    # 等待所有 worker 完成
    shard_paths = []
    for _ in range(len(processes)):
        gpu_id, shard_path, error = results.get()
        if error:
            logger.error(f"GPU {gpu_id} failed: {error}")
        elif shard_path:
            shard_paths.append(shard_path)
            logger.info(f"GPU {gpu_id} completed → {shard_path}")

    for p in processes:
        p.join()

    if len(shard_paths) == 0:
        logger.error("All GPU workers failed!")
        sys.exit(1)

    # 合并分片
    logger.info(f"Merging {len(shard_paths)} shards → {output_path}...")
    dfs = []
    for sp in sorted(shard_paths):
        dfs.append(pd.read_parquet(sp))
    merged = pd.concat(dfs, ignore_index=True)

    # 统计
    total_anchors = sum(len(r["anchor_positions"]) for _, r in merged.iterrows())
    logger.info(f"Total docs with anchors: {len(merged)}")
    logger.info(f"Total anchor tokens: {total_anchors:,}")

    merged.to_parquet(output_path, index=False)
    logger.info(f"Saved merged result to {output_path}")

    # 清理临时文件
    import shutil
    shutil.rmtree(shard_dir, ignore_errors=True)
    logger.info("Cleaned up temporary shard files.")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Anchor Token Selection via Proxy Model")
    parser.add_argument(
        "--proxy_model_path",
        type=str,
        default=PROXY_MODEL_PATH,
        help="代理模型路径",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DATASET_PATH,
        help="数据集路径 (JSONL)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
        help="输出路径 (parquet)",
    )
    parser.add_argument(
        "--anchor_ratio",
        type=float,
        default=ANCHOR_RATIO,
        help="Anchor 选取比例 (0-1)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help="代理模型单次处理的最大序列长度",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=MAX_DOCS,
        help="最大处理文档数 (None=全部)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从缓存恢复（断点续传）",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="GPU 并行数量（默认 1，设 >1 启用多 GPU 并行）",
    )
    args = parser.parse_args()

    # ============================================================
    # 1. 加载数据集
    # ============================================================
    logger.info(f"Loading dataset from {args.dataset_path}...")
    texts = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data["text"])

    if args.max_docs is not None:
        texts = texts[: args.max_docs]

    logger.info(f"Loaded {len(texts)} documents.")

    # ============================================================
    # 1.5 多 GPU 模式：启动并行 worker
    # ============================================================
    if args.num_gpus > 1:
        logger.info(f"Launching multi-GPU mode with {args.num_gpus} GPUs...")
        _launch_multi_gpu(
            texts=texts,
            num_gpus=args.num_gpus,
            proxy_model_path=args.proxy_model_path,
            max_seq_length=args.max_seq_length,
            anchor_ratio=args.anchor_ratio,
            output_path=args.output_path,
            resume=args.resume,
        )
        logger.info("Multi-GPU anchor selection done!")
        return

    os.makedirs(ENTROPY_CACHE_DIR, exist_ok=True)

    # ============================================================
    # 2. Resume 模式：跳过模型加载，直接从缓存重筛 anchor
    # ============================================================
    if args.resume:
        logger.info(
            f"Resume mode: loading cached entropies, "
            f"re-selecting anchors (ratio={args.anchor_ratio})..."
        )
        records = []
        missing_count = 0

        for i in tqdm(range(len(texts)), desc="Loading cache & selecting anchors"):
            cache = load_entropy_cache(i, ENTROPY_CACHE_DIR)
            if cache is not None:
                entropies = cache["entropies"]
                text = cache.get("text", texts[i])
                if len(entropies) > 0:
                    anchor_positions, anchor_entropies = select_anchor_positions(
                        entropies, anchor_ratio=args.anchor_ratio,
                    )
                    if len(anchor_positions) > 0:
                        records.append({
                            "text": text,
                            "anchor_positions": anchor_positions,
                            "entropy_values": anchor_entropies,
                        })
            else:
                missing_count += 1

        logger.info(
            f"Resume complete: {len(records)} docs with anchors, "
            f"{missing_count} docs missing cache"
        )

        logger.info(f"Saving {len(records)} documents to {args.output_path}...")
        df = pd.DataFrame(records)
        df.to_parquet(args.output_path, index=False)
        total_anchors = sum(len(r["anchor_positions"]) for r in records)
        logger.info(f"Total anchor tokens selected: {total_anchors:,}")
        logger.info("Done!")
        return

    # ============================================================
    # 3. 非 resume 模式：加载代理模型，计算熵值
    # ============================================================
    logger.info(f"Loading proxy model from {args.proxy_model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.proxy_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.proxy_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Proxy model loaded successfully.")

    # ============================================================
    # 4. 计算熵值
    # ============================================================
    logger.info("Computing per-token entropies...")
    all_entropies = []

    for i in tqdm(range(len(texts)), desc="Computing entropy"):
        text = texts[i]

        try:
            entropies = compute_entropy_batch(
                model=model,
                tokenizer=tokenizer,
                texts=[text],
                device=device,
                max_seq_length=args.max_seq_length,
            )[0]

            all_entropies.append(entropies)
            save_entropy_cache(i, text, entropies, ENTROPY_CACHE_DIR)

        except Exception as e:
            logger.error(f"Error processing doc {i}: {e}")
            all_entropies.append([])

    logger.info("Entropy computation completed.")

    # ============================================================
    # 5. 选取 Anchor Tokens
    # ============================================================
    logger.info(f"Selecting anchor tokens (ratio={args.anchor_ratio})...")
    records = []

    for i, (text, entropies) in enumerate(tqdm(zip(texts, all_entropies), total=len(texts))):
        if len(entropies) == 0:
            continue

        anchor_positions, anchor_entropies = select_anchor_positions(
            entropies,
            anchor_ratio=args.anchor_ratio,
        )

        if len(anchor_positions) > 0:
            records.append({
                "text": text,
                "anchor_positions": anchor_positions,
                "entropy_values": anchor_entropies,
            })

    # ============================================================
    # 6. 保存结果
    # ============================================================
    logger.info(f"Saving {len(records)} documents with anchors to {args.output_path}...")
    df = pd.DataFrame(records)
    df.to_parquet(args.output_path, index=False)

    total_anchors = sum(len(r["anchor_positions"]) for r in records)
    total_tokens = sum(len(e) for e in all_entropies)
    logger.info(f"Total tokens processed: {total_tokens:,}")
    logger.info(f"Total anchor tokens selected: {total_anchors:,}")
    logger.info(f"Anchor ratio: {total_anchors / max(total_tokens, 1):.2%}")
    logger.info("Done!")


if __name__ == "__main__":
    main()