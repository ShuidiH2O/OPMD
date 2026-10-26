"""
OPMD Dataset (Hybrid: OPD + NTP)
=================================
继承 verl 的 RLHFDataset，适配 OPMD 混合训练数据格式。

- MegaMath (task_type="opd"): 标准 OPD 流程（CoT rollout + teacher eval）
- FineWeb (task_type="ntp"): NTP 标准 causal LM，传递 ntp_labels

用法:
    在训练配置中指定:
    data.custom_dataset.path="OPMD/custom/dataset.py"
    data.custom_dataset.name=OPMDDataset
"""

import copy
import json
import logging
import os
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def _parse_json_field(value):
    """Parse a JSON string field back to its original Python object.
    
    In flattened parquet format, nested columns (prompt, reward_model, extra_info)
    are stored as JSON strings to avoid pyarrow nested-data chunked-array issues.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of sample dicts into batched tensors and arrays."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class OPMDDataset(Dataset):
    """
    OPMD 混合训练数据集，兼容 verl 的 DataProto 格式。
    
    与 RLHFDataset 的主要区别：
    1. 支持 task_type 区分 OPD 和 NTP 样本
    2. NTP 样本传递 ntp_labels（与 input_ids 相同，DataCollator 自动 shift）
    3. 保留 raw_prompt 用于 teacher model
    
    Args:
        data_files (str or list): Parquet 文件路径
        tokenizer (PreTrainedTokenizer): tokenizer
        config (DictConfig): verl 配置
        processor (ProcessorMixin, optional): 多模态处理器
        max_samples (int): 最大样本数（-1 = 全部）
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        # Base models (e.g. Llama-3.2-3B) may not have pad_token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.return_raw_chat = config.get("return_raw_chat", False)

        self.num_workers = config.get(
            "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
        )
        self.num_workers = min(self.num_workers, os.cpu_count())

        self._download()
        self._read_files_and_tokenize()

        if max_samples > 0 and len(self.dataframe) > max_samples:
            self.dataframe = self.dataframe[:max_samples]

        ntp_count = sum(
            1 for t in self.dataframe["task_type"] if t == "ntp"
        )
        opd_count = sum(
            1 for t in self.dataframe["task_type"] if t == "opd"
        )
        logger.info(
            f"OPMDDataset: {len(self.dataframe)} samples "
            f"(OPD: {opd_count}, NTP: {ntp_count})"
        )

    def _download(self):
        from verl.utils.fs import copy_to_local

        for i, parquet_file in enumerate(self.data_files):
            self.data_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe = datasets.concatenate_datasets(dataframes)

        self.dataframe = self.dataframe.map(
            self._tokenize_prompt,
            num_proc=self.num_workers,
            desc="Tokenizing prompts",
        )

        if self.filter_overlong_prompts:
            self.dataframe = self.dataframe.filter(
                lambda x: x["prompt_length"] <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc="Filtering overlong prompts",
            )

        logger.info(
            f"After filtering: {len(self.dataframe)} samples "
            f"(max_prompt_length={self.max_prompt_length})"
        )

    def _tokenize_prompt(self, sample: dict) -> dict:
        prompt = sample[self.prompt_key]
        # Parse JSON string (nested columns are flattened in parquet)
        prompt = _parse_json_field(prompt)

        if isinstance(prompt, list):
            # Chat format: use apply_chat_template if available
            if self.tokenizer.chat_template is not None:
                raw_prompt = self.tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=False
                )
                model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            else:
                # Base model without chat template: extract content directly
                prompt_text = "".join(
                    msg.get("content", "") for msg in prompt
                    if isinstance(msg, dict) and msg.get("role") == "user"
                )
                model_inputs = self.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=True
                )
        elif isinstance(prompt, str):
            model_inputs = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )
        else:
            prompt_text = str(prompt)
            model_inputs = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=True
            )

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # Use verl's postprocess_data for consistent left-padding & truncation
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        input_ids_1d = input_ids[0]
        attention_mask_1d = attention_mask[0]

        result = {
            "input_ids": input_ids_1d,
            "attention_mask": attention_mask_1d,
            "prompt_length": int(attention_mask_1d.sum().item()),
            "raw_prompt": sample[self.prompt_key],  # keep as JSON string, parse in __getitem__
        }

        return result

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe[idx]
        # Convert to tensors if stored as lists by datasets.map()
        input_ids = row["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = row["attention_mask"]
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "raw_prompt": _parse_json_field(row.get("raw_prompt", None)),
        }

        if "data_source" in row:
            item["data_source"] = row["data_source"]
        if "reward_model" in row:
            item["reward_model"] = _parse_json_field(row["reward_model"])
        if "extra_info" in row:
            item["extra_info"] = _parse_json_field(row["extra_info"])

        return item