import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from hilbertcurve.hilbertcurve import HilbertCurve
from timm.models.layers import DropPath
import numpy as np

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()


        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj


        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()


        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)

        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)


        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 输出四个横向扫描序列
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SelectiveScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            k=2,
            stack=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = 1
        self.d_model = d_model
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = k

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(k)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(k)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.stack = stack


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()

        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)

        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        B, K, C, L = x.shape
        assert K == self.K

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x.view(B, self.K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, self.K, -1, L), self.dt_projs_weight)

        xs = x.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, self.K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, self.K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, self.K, -1, L)
        assert out_y.dtype == torch.float

        return out_y if self.stack else torch.unbind(out_y, dim=1)

class DynamicTilesScan(nn.Module):
    def __init__(
            self,
            d_model,
            input_size,
            d_state=16,
            d_conv=3,
            expand=2,
            dropout=0.,
            bottleneck=4,
            patch=2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.bottleneck = bottleneck
        self.g_inner = self.d_inner // bottleneck
        self.patch = patch

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.register_buffer('intra_tiles_indices', self.get_hilbert_indices(self.patch))
        self.register_buffer('intra_tiles_indices_reverse', self.indices_reverse_intra(self.get_hilbert_indices(self.patch)))
        self.tiles_proj = nn.Sequential(
            nn.Conv2d(in_channels=self.d_inner,
                      out_channels=self.d_inner,
                      kernel_size=self.patch,
                      stride=self.patch,
                      groups=self.d_inner,
                      bias=False),
            nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner // bottleneck, kernel_size=1, bias=False),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=3, padding=1, bias=False, groups=self.d_inner),
            nn.SiLU(),
        )
        self.act = nn.SiLU()
        self.g_act = nn.Sigmoid()

        self.forward_core0 = SelectiveScan(d_model=self.d_inner, d_state=d_state, k=1, stack=True)
        self.forward_core1 = SelectiveScan(d_model=self.g_inner, d_state=d_state, k=2, stack=False)

        kernel_size = int(abs((math.log(self.g_inner, 2) + 1) / 2))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding),
            torch.nn.Sigmoid()
        )
        self.apply(self._init_module_weights)

    def _init_module_weights(self, module):
        if isinstance(module, nn.Conv2d):
            self._init_conv(module)
        elif isinstance(module, nn.BatchNorm2d):
            self._init_bn(module)

    def _init_conv(self, module):
        nn.init.kaiming_normal_(module.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if module.in_channels == module.out_channels:
            nn.init.orthogonal_(module.weight)

    def _init_bn(self, module):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)
        module.reset_running_stats()

    def bidirectional_gather(self, tensor: torch.Tensor):
        return torch.stack([tensor, torch.flip(tensor, dims=[-1])], dim=1)

    def indices_reverse_intra(self, indices: torch.Tensor):
        n = len(indices)
        device = indices.device

        inv_indices = torch.zeros(n, dtype=torch.long, device=device)
        positions = torch.arange(n, device=device)

        inv_indices[indices] = positions

        centrifugal_indices = inv_indices

        return centrifugal_indices

    def indices_reverse_inter(self, indices: torch.Tensor):
        B, C, N = indices.shape

        device = indices.device
        positions = torch.arange(N, device=device).view(1, 1, N).expand(B, C, N)

        centrifugal_indices = torch.zeros_like(indices)
        centrifugal_indices.scatter_(2, indices, positions)

        return centrifugal_indices

    def get_hilbert_indices(self, patch_size: int):
        n = int(np.log2(patch_size))
        if 2 ** n != patch_size:
            raise ValueError("patch_size must be a power of 2")

        hilbert_curve = HilbertCurve(n, 2)  # 2表示二维
        points = np.array([(x, y) for y in range(patch_size) for x in range(patch_size)])
        distances = hilbert_curve.distances_from_points(points)
        coord_to_index = {tuple(points[i]): distances[i] for i in range(len(points))}

        hilbert_indices = np.zeros((patch_size, patch_size), dtype=int)
        for y in range(patch_size):
            for x in range(patch_size):
                hilbert_indices[y, x] = coord_to_index[(x, y)]

        return torch.tensor(hilbert_indices.flatten(), dtype=torch.long)

    def inter_gather(self, x: torch.Tensor, inter_idx: torch.Tensor):
        B, C, N0, L = x.shape
        B, C, N1 = inter_idx.shape
        assert N0 == N1
        inter_idx_expanded = inter_idx.unsqueeze(-1).expand(-1, -1, -1, L)

        sorted_inter = torch.gather(x, dim=2, index=inter_idx_expanded)

        return sorted_inter

    def intra_gather(self, x: torch.Tensor, intra_idx: torch.Tensor):
        B, C, N, L0 = x.shape
        L1 = intra_idx.size(0)
        assert L0 == L1

        intra_idx_expanded = intra_idx.view(1, 1, 1, -1).expand_as(x)

        sorted_intra = torch.gather(x, dim=3, index=intra_idx_expanded)

        return sorted_intra

    def forward(self, x: torch.Tensor):
        # x = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x.shape
        L = self.patch**2
        N = (H // self.patch) * (W // self.patch)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x_r_p = self.act(self.tiles_proj(x))
        x_l = self.bidirectional_gather(tensor=x_r_p.view(B, self.g_inner, -1))
        g0, g1 = self.forward_core1(x_l)
        g = g0 + torch.flip(g1, dims=[-1])
        g_a = self.g_act(g)
        w = torch.repeat_interleave(input=g_a, repeats=self.bottleneck, dim=1) # [B C N]
        _ , inter_tiles_indices = torch.sort(w, dim=-1, stable=False)
        inter_tiles_indices_reverse = self.indices_reverse_inter(inter_tiles_indices)

        g_p = self.gap(g_a).view(B, 1, self.g_inner)
        w_cr = self.conv1d(g_p).view(B, self.g_inner, 1, 1)
        w_c = torch.repeat_interleave(input=w_cr, repeats=self.bottleneck, dim=1)

        x_unfold = F.unfold(x, kernel_size=(self.patch, self.patch), stride=(self.patch, self.patch)).view(B, self.d_inner, N, L)
        x_BCNL_inter = self.inter_gather(x=x_unfold, inter_idx=inter_tiles_indices)
        x_BCNL_intra = self.intra_gather(x=x_BCNL_inter, intra_idx=self.intra_tiles_indices).view(B, self.d_inner, -1).unsqueeze(1)
        y_BCD = self.forward_core0(x_BCNL_intra).squeeze(1).view(B, self.d_inner, N, L)
        y_BCNL_deintra = self.intra_gather(x=y_BCD, intra_idx=self.intra_tiles_indices_reverse)
        y_BCNL_deinter = self.inter_gather(x=y_BCNL_deintra, inter_idx=inter_tiles_indices_reverse)
        y_BZN = y_BCNL_deinter.permute(0, 1, 3, 2).contiguous().view(B, self.d_inner * L, -1)
        y_BCHW = F.fold(y_BZN, output_size=(H, W),kernel_size=(self.patch, self.patch), stride=(self.patch, self.patch))

        y_BCHW = y_BCHW * w_c.expand_as(y_BCHW) + y_BCHW

        y = y_BCHW.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class DTSBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_channels: int = 0,
                 drop_path: float = 0.1,
                 attn_drop_rate: float = 0,
                 d_state: int = 8,
                 ss2d_expand_ratio: int = 2,
                 patch = 2,
                 bottleneck = 4
                 ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(input_channels)
        self.self_attention = DynamicTilesScan(d_model=input_channels, input_size=input_size,dropout=attn_drop_rate, d_state=d_state, expand=ss2d_expand_ratio, patch=patch, bottleneck=bottleneck)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x

def calculating_params_flops(model, channel, size):
    from thop import profile
    input = torch.randn(1, channel, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print("FLOPs: %.3f G" % (flops / 1e9))
    print("Params: %.3f M" % (params / 1e6))

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3f M" % (total / 1e6))


if __name__ == "__main__":
    input_channels = 256
    input_size = 28

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DynamicTilesScan(d_model=input_channels, patch=4, input_size=input_size).to(device)
    calculating_params_flops(model, input_channels, input_size)
