import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fmha_compat import fmha
from .model_utils import initialize_weights
from .roformer import RoFormerEncoder


def inefficient_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
):
    attn_weight = torch.softmax(
        (q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask,
        dim=-1,
    )
    return attn_weight @ v, attn_weight


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class AttentionGated(nn.Module):
    def __init__(self, dim, act="relu", bias=False, dropout=False, rrt=None):
        super().__init__()
        self.L = dim
        self.D = 128
        self.K = 1

        feature = [nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.25)]
        if rrt is not None:
            feature.append(rrt)
        self.feature = nn.Sequential(*feature)

        self.attention_a = [nn.Linear(self.L, self.D, bias=bias)]
        if act == "gelu":
            self.attention_a.append(nn.GELU())
        elif act == "relu":
            self.attention_a.append(nn.ReLU())
        elif act == "tanh":
            self.attention_a.append(nn.Tanh())

        self.attention_b = [
            nn.Linear(self.L, self.D, bias=bias),
            nn.Sigmoid(),
        ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(self.D, self.K, bias=bias)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.feature(x)
        a = self.attention_a(x)
        b = self.attention_b(x)
        attention = self.attention_c(a.mul(b))
        return torch.transpose(attention, -1, -2)


class LayerNorm(nn.Module):
    def __init__(self, n_embd, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class ClassAttention(nn.Module):
    def __init__(
        self,
        n_classes: int,
        hidden_dim: int,
        attention_dim: int,
        n_head: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        assert attention_dim % n_head == 0

        self.keys_projection = nn.Linear(hidden_dim, attention_dim, bias=bias)
        self.values_projection = nn.Linear(hidden_dim, attention_dim, bias=bias)
        self.output_projection = nn.Linear(attention_dim, hidden_dim, bias=bias)
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.dropout = dropout
        self.n_classes = n_classes
        self.class_tokens = nn.Parameter(
            nn.init.xavier_normal_(torch.rand((1, n_classes, attention_dim), requires_grad=True))
        )
        self.output_inference_weights = True

    def forward(self, features: torch.Tensor, attn_bias: fmha.BlockDiagonalMask) -> torch.Tensor:
        keys = self.keys_projection(features)
        values = self.values_projection(features)

        k = keys.view(1, features.shape[1], self.n_head, self.attention_dim // self.n_head)
        q = (
            self.class_tokens.view(1, self.n_classes, self.n_head, self.attention_dim // self.n_head)
            .repeat(1, len(attn_bias._batch_sizes), 1, 1)
            .to(k)
        )
        v = values.view(1, features.shape[1], self.n_head, self.attention_dim // self.n_head)

        class_attn_bias = fmha.BlockDiagonalMask.from_seqlens(
            q_seqlen=[self.n_classes] * len(attn_bias._batch_sizes),
            kv_seqlen=[el[1] - el[0] for el in attn_bias.k_seqinfo.intervals()],
        )

        attn_weights = torch.zeros([])
        if not q.is_cuda or self.output_inference_weights:
            out, attn_weights = inefficient_scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=class_attn_bias.materialize((q.shape[1], k.shape[1])).to(q),
            )
            out = out.transpose(1, 2).reshape(
                len(attn_bias._batch_sizes),
                self.n_classes,
                self.attention_dim,
            )
            attn_weights = einops.rearrange(attn_weights, "b h c n -> b c n h")
        else:
            out = fmha.memory_efficient_attention(q, k, v, attn_bias=class_attn_bias).view(
                len(attn_bias._batch_sizes),
                self.n_classes,
                self.attention_dim,
            )

        return self.output_projection(out), attn_weights


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class CrossAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.q_projection = nn.Linear(n_embd, n_embd, bias=bias)
        self.kv_projection = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.output_projection = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, q, features: torch.Tensor, attn_bias: fmha.BlockDiagonalMask | None) -> torch.Tensor:
        b = q.shape[0] if len(q.shape) == 2 else q.shape[1]
        bs = features.shape[0] if len(features.shape) == 2 else features.shape[1]

        q = self.q_projection(q).split(self.n_embd, dim=-1)[0]
        k, v = self.kv_projection(features).split(self.n_embd, dim=-1)
        k = k.view(1, bs, self.n_head, self.n_embd // self.n_head)
        q = q.view(1, b, self.n_head, self.n_embd // self.n_head)
        v = v.view(1, bs, self.n_head, self.n_embd // self.n_head)

        if not q.is_cuda:
            out = (
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=0,
                )
                .transpose(1, 2)
                .reshape(1, -1, self.n_embd)
            )
        else:
            out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(
                1,
                q.shape[1],
                self.n_embd,
            )
        return self.resid_dropout(self.output_projection(out))


class CrossBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool = True,
        rope: bool = False,
        rope_freqs: str = "pixel",
        resid_dropout: float = 0,
    ):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd=n_embd, bias=bias)
        self.attn = CrossAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=resid_dropout,
            bias=bias,
        )
        self.ln_2 = LayerNorm(n_embd=n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask | None,
    ) -> torch.Tensor:
        attention_output = self.attn(
            q=self.ln_1(features_1),
            features=self.ln_1(features_2),
            attn_bias=attn_bias,
        )
        features = features_1 + attention_output
        features = features + self.mlp(self.ln_2(features))
        return features


class PathoMIL_RE(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim=512,
        num_heads=8,
        n_classes=2,
        num_tokens=512,
        depth=2,
        register=None,
        qkv_bias=False,
        qkv_scale=None,
        drop=0.0,
        high_dim=None,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        task="subtype",
        topk=16,
        dropout=0.25,
    ):
        super().__init__()
        depth = 2
        high_dim = dim if high_dim is None else high_dim

        self.task = task
        self.depth = depth
        self.topk = [0.2, 0.1]
        self.T = [1.2, 1]
        self.n_expert = 1
        self.step_loss_weights = [0.1, 0.1]

        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
        self.fc2 = nn.Sequential(nn.Linear(high_dim, hidden_dim), nn.GELU())
        self.reflect = nn.ModuleList(
            [CrossBlock(hidden_dim, num_heads, 0.25, False) for _ in range(depth - 1)]
        )
        self.gated = nn.ModuleList(
            [AttentionGated(hidden_dim, bias=False) for _ in range(depth)]
        )
        self.r_attention = nn.ModuleList(
            [
                RoFormerEncoder(
                    n_attention_block=1,
                    n_embd=hidden_dim,
                    n_head=num_heads,
                    dropout=0.25,
                    bias=True,
                    rope=True,
                    rope_freqs="pixel",
                    resid_dropout=0.25,
                )
                for _ in range(depth)
            ]
        )
        self.cls_attention_fine = ClassAttention(
            hidden_dim=hidden_dim,
            attention_dim=256,
            dropout=0.25,
            n_classes=1,
            n_head=num_heads,
            bias=True,
        )
        self.classfier = nn.Sequential(nn.Linear(hidden_dim, n_classes))
        initialize_weights(self)

    def forward_features(self, x_high, coords_high, x_low, coords_low, region_labels, T=1):
        if region_labels is None:
            raise ValueError("GFR-MIL requires region_labels")
        if region_labels.dim() == 1:
            region_labels = region_labels.unsqueeze(0)
        valid_mask = region_labels[0] >= 0
        if not torch.any(valid_mask):
            raise ValueError("region_labels must contain at least one valid low-region index")
        x_high = x_high[valid_mask]
        coords_high = coords_high[valid_mask]
        region_labels = region_labels[:, valid_mask]
        max_idx = int(region_labels.max().item())
        if max_idx >= x_low.shape[0]:
            raise ValueError("region_labels contain an index outside the low-feature bag")
        x_low = x_low[: max_idx + 1, :]

        coords_high = coords_high.long() // 448
        coords_low = coords_low.long() // 3584
        x_high = self.fc2(x_high)
        x_low = self.fc1(x_low)

        step_features = []
        att_loss = 0
        chosen_region = None
        for step_idx in range(self.depth):
            (
                features_i,
                att_loss_i,
                last_choice,
                logits_c,
                x_low,
                x_high,
                coords_high,
                region_labels,
            ) = self.forward_one_step(
                step_idx,
                x_high,
                coords_high,
                x_low,
                coords_low,
                region_labels,
                self.T[step_idx],
                chosen_region,
            )
            if chosen_region is None:
                chosen_region = last_choice[0]
            else:
                chosen_region = torch.cat([chosen_region, last_choice[0]], dim=0)

            step_features.append(features_i)
            if step_idx < self.depth - 1:
                x_low = self.reflect[step_idx](x_low, features_i, None)[0]
            att_loss = att_loss + self.step_loss_weights[step_idx] * att_loss_i

        logits = torch.cat(step_features, dim=1)[0]
        attn_bias, logits = fmha.BlockDiagonalMask.from_tensor_list(
            [feature.unsqueeze(0) for feature in [logits]]
        )
        logits, _ = self.cls_attention_fine(logits, attn_bias)
        logits = logits.mean(dim=1, keepdims=True)
        logits = self.classfier(logits)[0]
        return logits, att_loss / self.depth

    def forward_one_step(
        self,
        dp,
        x_high,
        coords_high,
        x_low,
        coords_low,
        region_labels,
        T=1,
        chosen_region=None,
    ):
        gated_attention = self.gated[dp](x_low).reshape(-1, self.n_expert).permute(1, 0)
        logits_c = 0
        attn_weights = torch.sigmoid(gated_attention)
        valid_regions = region_labels[region_labels >= 0].unique()
        selectable = torch.zeros_like(gated_attention, dtype=torch.bool)
        selectable[:, valid_regions] = True
        if chosen_region is not None:
            selectable[:, chosen_region] = False
            if selectable.sum() == 0:
                selectable[:, valid_regions] = True

        available_count = int(selectable.sum().item())
        sel_k = min(max(1, int(self.topk[dp] * available_count)), available_count)

        if self.training:
            weights = F.softmax(gated_attention / T, dim=-1)
            att_loss = (-weights * torch.log(weights + 1e-9)).sum()
            weights = weights * selectable.float()
            weight_sum = weights.sum()
            if weight_sum < 1e-8:
                weights = selectable.float() / selectable.float().sum().clamp_min(1.0)
            else:
                weights = weights / weight_sum
            topk_indices = torch.multinomial(weights, sel_k, replacement=False)
            topk_values = torch.gather(input=attn_weights, dim=1, index=topk_indices)
        else:
            masked_attn = attn_weights.masked_fill(~selectable, -1.0)
            topk_values, topk_indices = torch.topk(masked_attn, k=sel_k, dim=1)
            att_loss = torch.tensor(0).to(x_high.device)

        region_labels, sorted_idx = torch.sort(region_labels)
        x_high = x_high[sorted_idx[0]]
        coords_high = coords_high[sorted_idx[0]]
        topk_indices, sorted_topk_idx = torch.sort(topk_indices, dim=-1)
        topk_values = torch.gather(input=topk_values, dim=1, index=sorted_topk_idx)

        expert_idx = 0
        selected_mask = torch.isin(region_labels[0], topk_indices[expert_idx])
        if selected_mask.sum() == 0:
            selected_mask[0] = True
        x_high_i = x_high[selected_mask, :]
        coords_high_i = coords_high[selected_mask, :]

        counts = torch.bincount(region_labels[region_labels >= 0])
        targets = topk_indices[expert_idx]
        sel_counts = counts[targets]
        att = torch.repeat_interleave(topk_values[expert_idx], sel_counts, dim=-1).reshape(1, -1, 1)

        all_idx = valid_regions
        if self.training:
            remain_idx = all_idx[~torch.isin(all_idx, topk_indices[expert_idx])]
            if remain_idx.numel() > 0:
                att_remain = attn_weights[0, remain_idx]
                remain_mask = torch.isin(region_labels[0], remain_idx)
                x_high_remain = x_high[remain_mask, :]
                coords_high_remain = coords_high[remain_mask, :]
                remain_counts = counts[remain_idx]
                att_recycle = torch.repeat_interleave(att_remain, remain_counts, dim=-1).reshape(1, -1)
                denom = att_recycle.sum().clamp_min(1e-6)
                x_high_re = torch.matmul(att_recycle, x_high_remain) / denom
                coords_high_re = torch.matmul(att_recycle, coords_high_remain.float()) / denom
            else:
                x_high_re = torch.matmul(att.reshape(1, -1), x_high_i) / att.reshape(1, -1).sum().clamp_min(1e-6)
                coords_high_re = torch.matmul(att.reshape(1, -1), coords_high_i.float()) / att.reshape(1, -1).sum().clamp_min(1e-6)
            coords_high_re = coords_high_re.long()
            x_high = torch.cat([x_high, x_high_re], dim=0)
            coords_high = torch.cat([coords_high, coords_high_re], dim=0)
            x_high_i = torch.cat([x_high_i, x_high_re], dim=0)
            coords_high_i = torch.cat([coords_high_i, coords_high_re], dim=0)

        attn_bias, features_high_i = fmha.BlockDiagonalMask.from_tensor_list(
            [feature.unsqueeze(0) for feature in [x_high_i]]
        )
        _, coords_high_i = fmha.BlockDiagonalMask.from_tensor_list(
            [coord.unsqueeze(0) for coord in [coords_high_i]]
        )

        if self.training:
            att_high_i = torch.concat([att, torch.ones([1, 1, 1], device=x_low.device)], dim=1)
        else:
            att_high_i = att
        features_high_i = att_high_i * features_high_i
        features_high_i = self.r_attention[dp](features_high_i, coords_high_i, attn_bias)

        if self.training:
            region_labels = torch.concat(
                [region_labels, torch.tensor(-1, device=x_low.device).reshape(-1, 1)],
                dim=1,
            )

        return features_high_i, att_loss, topk_indices, logits_c, x_low, x_high, coords_high, region_labels

    def forward(self, x_high, coords_high, x_low, coords_low, region_labels, return_attn=False):
        logits, att_loss = self.forward_features(x_high, coords_high, x_low, coords_low, region_labels)
        if self.task == "subtype":
            return *self.forward_subtyping(logits), att_loss
        if self.task == "survival":
            return *self.forward_prognosis(logits), att_loss
        raise ValueError(f"unsupported task: {self.task}")

    def forward_subtyping(self, logits):
        prob = F.softmax(logits, dim=-1)
        y_hat = torch.topk(prob, 1, dim=1)[1]
        return logits, prob, y_hat

    def forward_prognosis(self, logits):
        y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return hazards, survival, y_hat
