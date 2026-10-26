"""
OPMD Training Entry Point
=========================
OPMD 使用 verl 的 PPO trainer 实现 Reverse KL distillation。
通过 token_reward_direct 优势估计器 + compute_distillation_reward
将 Reverse KL 梯度映射为 PPO 框架下的 token-level advantage 信号。

Usage: python OPMD/custom/train_entry.py [verl args...]
"""

import sys
import os

# Ensure OPMD is in path
_opmd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _opmd_dir not in sys.path:
    sys.path.insert(0, _opmd_dir)

# Use verl's PPO trainer with OPD's custom advantage estimator (token_reward_direct)
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()