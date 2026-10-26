# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_cuda_available, is_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_cuda_available:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except (ImportError, ModuleNotFoundError):
            import torch

            def index_first_axis(x, indices):
                return x[indices]

            def pad_input(hidden_states, indices, batch, seqlen):
                # indices are flattened into (batch * seqlen) space
                shape = hidden_states.shape[1:]  # trailing dims, e.g. (1,) or (vocab_size,)
                output = torch.zeros(batch * seqlen, *shape, dtype=hidden_states.dtype, device=hidden_states.device)
                output[indices] = hidden_states
                return output.reshape(batch, seqlen, *shape)

            def rearrange(x, pattern):
                try:
                    from einops import rearrange as einops_rearrange
                    return einops_rearrange(x, pattern)
                except ImportError:
                    # Fallback for common patterns used in verl
                    if pattern == "b s ... -> (b s) ...":
                        return x.reshape(-1, *x.shape[2:])
                    raise ImportError(
                        "einops is required for rearrange fallback with custom patterns"
                    )

            def unpad_input(x, attention_mask):
                seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
                indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                max_seqlen_in_batch = seqlens_in_batch.max().item()
                cu_seqlens = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                x_unpad = index_first_axis(x.reshape(-1, *x.shape[2:]), indices)
                return x_unpad, indices, cu_seqlens, max_seqlen_in_batch
    elif is_npu_available:
        from verl.utils.npu_utils import index_first_axis, pad_input, rearrange, unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
