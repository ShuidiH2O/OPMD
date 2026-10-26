"""
OPMD Reward Function (Monitoring-Only Stub)
============================================
OPMD 使用 Reverse KL 蒸馏 reward（由 compute_distillation_reward 产生的 rm_scores），
此 rule-based reward 函数仅用于训练监控（true_reward_score），不参与梯度更新。

NaiveRewardManager 检测到 batch 中存在 rm_scores 时会自动使用 rm_scores 作为训练信号，
此函数返回的 score 仅作为 true_reward_score 记录在日志中。

在新的 OPMD 设计中，anchor token x_t 已经包含在 context 窗口 (x_{<t}, x_t) 中，
Student 的 generation c 不再需要"预测" anchor token，因此此函数不执行任何有意义的
规则检查。

用法:
    在训练脚本中指定:
    custom_reward_function.path="OPMD/custom/reward.py"
    custom_reward_function.name=reward_func
"""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    OPMD 监控 stub（仅用于对 verl 接口的兼容，不提供有意义的规则奖励）。

    实际训练信号完全由 rm_scores（Reverse KL distillation reward）提供。
    真正的训练监控指标（entropy、advantage 等）在训练日志中已有完整输出。

    Args:
        data_source: 数据来源标识
        solution_str: 模型生成的 continuation 文本
        ground_truth: 未使用（anchor token 在 context 中，不在 generation 中）
        extra_info: 额外信息（anchor_position, anchor_entropy 等）

    Returns:
        dict: {"score": 0.0}（恒为 0，训练信号由 rm_scores 提供）
    """
    return {"score": 0.0}


def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    """
    OPMD reward 函数入口（verl 兼容接口）。

    此函数不参与梯度更新。NaiveRewardManager 检测到 batch 中存在 rm_scores 时
    会自动使用 rm_scores 作为 reward_tensor，此函数返回值仅记录为 true_reward_score。
    """
    return compute_score(data_source, solution_str, ground_truth, extra_info)
