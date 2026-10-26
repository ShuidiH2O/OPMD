"""
LLM-as-a-Judge Answer Extraction
================================
使用 DeepSeek-R1-Distill-Qwen-1.5B 从模型回复中提取最终答案，
避免因格式问题（如 \boxed{} 缺失）导致评测偏差。

用法:
    from judge_extract import JudgeExtractor
    judge = JudgeExtractor()
    answer = judge.extract("The solution is ... therefore the answer is 42.")
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

JUDGE_MODEL_PATH = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/DeepSeek-R1-Distill-Qwen-1.5B"

JUDGE_PROMPT = """You are an answer extraction assistant. Given a mathematical solution, extract ONLY the final numerical or algebraic answer. Output nothing else.

Rules:
1. If the solution ends with \\boxed{{X}}, output X.
2. If the solution states "the answer is X" or "final answer: X", output X.
3. If the solution ends with a number or expression, output that.
4. Output ONLY the answer, no explanation, no extra text.
5. If you cannot determine the answer, output "UNKNOWN".

Solution:
__SOLUTION__

Final answer:"""


class JudgeExtractor:
    """使用本地小模型从回复中提取最终答案。"""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        self.model_path = model_path or JUDGE_MODEL_PATH
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )

    def extract(self, solution: str, max_new_tokens: int = 64) -> str:
        """从 solution 文本中提取最终答案。"""
        self._load()

        prompt = JUDGE_PROMPT.replace("__SOLUTION__", solution[-6000:])
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # 清理输出
        response = response.split("\n")[0].strip()
        return response

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()


def extract_answer_with_judge(judge, solution: str) -> str:
    """使用 judge 提取答案，失败时 fallback 到正则提取。"""
    try:
        ans = judge.extract(solution)
        if ans and ans.upper() != "UNKNOWN" and len(ans) <= 200:
            return ans
    except Exception:
        pass

    # Fallback 到正则提取
    from utils import extract_answer as regex_extract
    return regex_extract(solution)