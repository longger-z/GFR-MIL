from __future__ import annotations

import torch

try:
    from xformers.ops import fmha
except ModuleNotFoundError:
    class _SeqInfo:
        def __init__(self, lengths: list[int]) -> None:
            self.lengths = lengths

        def intervals(self) -> list[tuple[int, int]]:
            start = 0
            intervals = []
            for length in self.lengths:
                intervals.append((start, start + length))
                start += length
            return intervals

    class BlockDiagonalMask:
        def __init__(self, q_lengths: list[int], kv_lengths: list[int] | None = None) -> None:
            self.q_lengths = q_lengths
            self.kv_lengths = kv_lengths or q_lengths
            self._batch_sizes = q_lengths
            self.k_seqinfo = _SeqInfo(self.kv_lengths)

        @classmethod
        def from_tensor_list(cls, tensors: list[torch.Tensor]):
            lengths = [tensor.shape[1] for tensor in tensors]
            return cls(lengths), torch.cat(tensors, dim=1)

        @classmethod
        def from_seqlens(cls, q_seqlen: list[int], kv_seqlen: list[int]):
            return cls(q_seqlen, kv_seqlen)

        def materialize(self, shape, dtype=None, device=None):
            if hasattr(dtype, "dtype"):
                device = dtype.device
                dtype = dtype.dtype
            dtype = dtype or torch.float32
            mask = torch.full(shape, float("-inf"), dtype=dtype, device=device)
            q_start = 0
            kv_start = 0
            for q_len, kv_len in zip(self.q_lengths, self.kv_lengths):
                mask[q_start:q_start + q_len, kv_start:kv_start + kv_len] = 0
                q_start += q_len
                kv_start += kv_len
            return mask

    class _Fmha:
        BlockDiagonalMask = BlockDiagonalMask

        @staticmethod
        def memory_efficient_attention(q, k, v, attn_bias=None):
            mask = None
            if attn_bias is not None:
                mask = attn_bias.materialize((q.shape[1], k.shape[1]), dtype=q.dtype, device=q.device)
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask,
                dropout_p=0,
            )
            return out.transpose(1, 2)

    fmha = _Fmha()
