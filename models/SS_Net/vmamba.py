import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from mamba_ssm import selective_scan_fn as selective_scan_fn_v1
    from mamba_ssm import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import numpy as np
np.ones(1)


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class PatchEmbed2D(nn.Module):
    # 实现PatchEmbedding
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        # 若patch_size是整数
        if isinstance(patch_size, int):
            # 每个patch的大小为patch_size*patch_size
            patch_size = (patch_size, patch_size)
        # 定义 卷积，将patch映射到embed_dim维空间
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 若norm_layer标志位挂起
        if norm_layer is not None:
            # 启用norm_layer
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 将patch映射到embed_dim维空间后调整张量格式
        # 将原来序号为0,1,2,3的通道调整为0,2,3,1，即B,H,W,C
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            # 若启用层归一化
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    # 实现PatchMerging
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # 定义 输入的维度/输入的通道数
        self.dim = dim
        # 定义 线性层
        # 该线性层是实现两倍的维度/通道缩放
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 定义 线性层前的层归一化
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        # 获取输入的shape
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        # 若宽和高不是偶数，则输出警告信息
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            # SHAPE_FIX记录为PatchMerging后的H, W
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        # 将图片分为4部分
        # 取所有批次；从索引0开始，在高度和宽度上每隔一个元素取一个；取全部维度
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # 取所有批次；从索引1开始，在高度每隔一个元素取一个；从索引0开始，在宽度每隔一个元素取一个；取全部维度
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # 取所有批次；从索引0开始，在高度每隔一个元素取一个；从索引1开始，在宽度每隔一个元素取一个；取全部维度
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # 取所有批次；从索引1开始，在高度每隔一个元素取一个；从索引1开始，在宽度每隔一个元素取一个；取全部维度
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # 若可以进行PatchMerging
        if SHAPE_FIX[0] > 0:
            # 确保每个部分都是 SHAPE_FIX[0]*SHAPE_FIX[1]的大小
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        # 将四个部分进行维度平接/通道平接
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # 将x展平，(B, H/2,W/2, 4*C)->(B, H/2*W/2, 4*C)
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        # 进行归一化
        x = self.norm(x)
        # 通过线性层进行缩放
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # 定义 输入的维度/输入的通道数
        # 因为VSSM-dims_decoder[]定义的是每个VSS Layer Up的输出通道的缘故，而第一层VSS Layer Up不存在PatchExpand2D
        # 所以PatchExpand2D接收的是上一层VSS Layer Up的输出通道，即本层VSS Layer Up的输出通道两倍
        self.dim = dim * 2
        # 定义 扩大倍率
        self.dim_scale = dim_scale
        # 定义 线性层
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        # 定义 层归一化层
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        # 获取输入的shape
        B, H, W, C = x.shape
        # 通过线性层，使 维度/通道 变为原来的{扩大倍率}倍
        # B, H, W, C -> B, H, W, dim_scale*C
        x = self.expand(x)

        # 重排和重塑x张量
        # B, H, W, dim_scale*C = B, H, W, dim_scale*dim_scale*C/dim_scale -> B, H*dim_scale, W*dim_scale, C/dim_scale
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        # 通过层归一化
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # 定义 输入的维度/输入的通道数
        self.dim = dim
        # 定义 扩大倍率
        self.dim_scale = dim_scale
        # 定义 线性层
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        # 定义 层归一化层
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        # 获取输入的shape
        B, H, W, C = x.shape
        # 通过线性层，使 维度/通道 变为原来的{扩大倍率}倍
        x = self.expand(x)

        # 重排和重塑x张量
        # B, H, W, dim_scale*C = B, H, W, dim_scale*dim_scale*C/dim_scale -> B, H*dim_scale, W*dim_scale, C/dim_scale
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        # 通过层归一化
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    # 图形化 四向扫描 S6主模块
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
        # 定义 输入的维度
        self.d_model = d_model
        # 隐状态维度
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        # 定义 深度卷积/DWConv/逐通道卷积的卷积核大小
        self.d_conv = d_conv
        # 定义 线性层升降维倍率
        self.expand = expand
        # 定义 输入升维后的维度
        self.d_inner = int(self.expand * self.d_model)
        # 定义 低秩SΔ(x)的秩维度，若"auto"则根据输入维度 自动设定
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 定义 输入侧线性层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 定义 深度卷积/DWConv/逐通道卷积
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # 定义 SiLU激活函数
        self.act = nn.SiLU()

        # 定义 一个临时元组
        # 初始化 线性层，最后取其权重
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # x_proj_weight由x_proj的权重组成，是可学习的参数
        # x_proj_weight形状为(扫描路径数, 综合维度, 秩维度)
        # 综合维度 = 秩维度+隐状态维度+隐状态维度
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 定义 一个临时元组
        # 初始化 线性层，最后取其权重
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
        # dt_projs_weight由dt_projs的权重组成，是可学习的参数
        # dt_projs_weight形状为(扫描路径数, x输入的维度, 秩维度)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        # 定义 初始化的可学习A矩阵
        # shape为(扫描路径数, x输入的维度, 隐状态维度)
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # 定义 初始化的可学习D矩阵
        # shape为(扫描路径数, x输入的维度, 隐状态维度)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        # 定义 最后的层归一化
        self.out_norm = nn.LayerNorm(self.d_inner)
        # 定义 输出侧线性层
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # 定义 随机失活
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        # SΔ(x)线性层的 权重 的初始化，

        # 定义 线性层
        # 输入为秩维度，输出为SS2D内x维度
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # # Initialize special dt projection to preserve variance at initialization
        # 初始化特殊的dt projection，以在初始化时保持方差
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            # 如果 dt_init 为 "constant"，即恒定初始化，则将权重初始化为常数 dt_init_std。
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            # 如果 dt_init 为 "random"，即随机初始化，则将权重均匀初始化在 [-dt_init_std, dt_init_std] 范围内。
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        # 初始化dt偏差，使F.softplus（dt_bias）在dt_min和dt_max之间
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
            # 限制其最小值为dt_init_floor
        ).clamp(min=dt_init_floor)
        # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # 我们的初始化会将所有Linear.bias设置为零，需要将此标记为_no_reinit
        dt_proj.bias._no_reinit = True

        # 返回一个(B,L,N)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        # S4D real initialization
        # 生成一个(x输入的维度, 隐状态维度)的矩阵
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()

        # log化使A矩阵去极端化
        A_log = torch.log(A)  # Keep A_log in fp32
        # 复制 {扫描路径数} 份
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        # 不进行L2范数正则化
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        # 生成一个(x输入的维度, 隐状态维度)的全一矩阵
        D = torch.ones(d_inner, device=device)

        # 复制 {扫描路径数} 份
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        # 不进行L2范数正则化
        D._no_weight_decay = True
        return D


    def selective_scan_ref(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                           return_last_state=False):
        # B:批次大小； D: 输入S6维度； L: 序列长度； N: 隐状态维度； G: 扫描路径数

        # 在VM-UNet中：
        # ——————————输入侧——————————
        # 输入序列u 张量格式：(B D L)
        # 预-时间窗口时变矩阵SΔ(x) 张量格式：(B D L)
        # 遗忘门时不变矩阵A 张量格式：(D N)
        # 输入门时变矩阵B 张量格式：(B G N L)
        # 输出门时变矩阵C 张量格式：(B G N L)
        # 残差连接时不变矩阵D 张量格式：(D)
        # 支路输入序列z 张量格式：(B D L)
        # 预-时间窗口时变矩阵SΔ(x)的偏置 张量格式：(D)
        # ——————————输出侧——————————
        # 输出序列Out 张量格式：(B D L)
        # 可选输出 隐状态 张量格式：(B D N)
        """
        u: r(B D L)
        delta: r(B D L)
        A: c(D N) or r(D N)
        B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
        D: r(D)
        z: r(B D L)
        delta_bias: r(D), fp32

        out: r(B D L)
        last_state (optional): r(B D dstate) or c(B D dstate)
        """
        # ——————————预处理——————————
        # 获取输入序列u的数据类型
        dtype_in = u.dtype
        # 将输入序列u 内元素 变为浮点型
        u = u.float()
        # 将时间窗口时变矩阵SΔ(x)-delta 内元素 变为浮点型
        delta = delta.float()

        # 若 预-时间窗口时变矩阵SΔ(x)的偏置delta_bias 存在：
        if delta_bias is not None:
            # 将 时间窗口时变矩阵SΔ(x)-delta的内元素 与 浮点化的预-时间窗口时变矩阵SΔ(x)的偏置delta_bias 数值相加
            delta = delta + delta_bias[..., None].float()

        # 若 标志位--预-时间窗口时变矩阵SΔ(x)的应用softplus 挂起
        if delta_softplus:
            # 预-时间窗口时变矩阵SΔ(x)应用softplus，变为Δ
            # 注意：以下的delta全部变更说明为：时间窗口时变矩阵Δ。无论有没有应用softplus
            delta = F.softplus(delta)

        # 获取 批次大小；输入S6的维度；隐状态维度
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

        # 若 输入门时变矩阵B 与 输出门时变矩阵C 的张量维度大于等于3，即为S6类型的B C。则相应标志位挂起。
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3

        # 若 遗忘门时不变矩阵A 内元素是复数
        if A.is_complex():
            # 若 输入门时变矩阵B 是S6类型的
            if is_variable_B:
                # B.float()： 首先将 输入门时变矩阵B 转换为 浮点型
                # rearrange(B.float(), "... (L two) -> ... L two", two=2)： 然后将 输入门时变矩阵B 最后一个维度(序列长度)分为两部分。并将新部分作为新的维度，以满足实部与虚部。注意，虚部是空的。
                # torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2)) ：最后将 重排的输入门时变矩阵B 转换为复数张量
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))

            # 若 输出门时变矩阵C 是S6类型的
            # 同理
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        # 若 遗忘门时不变矩阵A 内元素不是复数
        else:
            # 将 输入门时变矩阵B 转换为 浮点型
            B = B.float()
            # 将 输出门时变矩阵C 转换为 浮点型
            C = C.float()

        # 新建一个全零张量X，与A的张量格式与数据类型相同,表示 当前时间步的隐状态张量x
        x = A.new_zeros((batch, dim, dstate))
        # 新建一个列表ys，记录每个时间步的输出
        ys = []

        # ——————————离散化——————————
        # 将 遗忘门时不变矩阵A 进行离散化，公式如下：
        # →A = exp(Δ A)
        # 时间窗口时变矩阵Δ 与 遗忘门时不变矩阵A 进行张量相乘，结果进行逐个元素的e指数运算，得到 离散化遗忘门时不变矩阵deltaA
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

        # 将 输入门时变矩阵B 与 进行离散化，公式如下：
        # 若 输入门时变矩阵B 是S4类型
        if not is_variable_B:
            # 时间窗口时变矩阵Δ 与 输入门时变矩阵B 与 输入序列u 进行张量相乘
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        # 若 输入门时变矩阵B 是S6类型的
        else:
            # 若 输入门时变矩阵B 是标准S6类型
            if B.dim() == 3:
                # 时间窗口时变矩阵Δ 与 输入门时变矩阵B 与 输入序列u 进行张量相乘
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            # 若 输入门时变矩阵B 是高维S6类型
            else:
                # 输入门时变矩阵B 沿 扫描路径维度 重复 {输入S6维度数 // 扫描路径维度数}遍
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                # 时间窗口时变矩阵Δ 与 输入门时变矩阵B 与 输入序列u 进行张量相乘
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

        # 若 输出门时变矩阵C 是S6类型的 并且还是高维S6类型
        if is_variable_C and C.dim() == 4:
            # 输出门时变矩阵C 沿 扫描路径维度 重复 {输入S6维度数 // 扫描路径维度数}遍
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None

        # ——————————时间步迭代——————————
        # Ht = →A * Ht-1 + →B * x
        # y = C * Ht
        # 根据 序列长度 来迭代
        for i in range(u.shape[2]):
            # 计算 当前时间步的隐状态张量x
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]

            # 若 输出门时变矩阵C 是S4类型
            if not is_variable_C:
                # 当前时间步的隐状态张量x 与 输出门时变矩阵C 张量相乘 ，得到 当前时间步输出序列y
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                # 若 输出门时变矩阵C 是标准S6类型
                if C.dim() == 3:
                    # 当前时间步的隐状态张量x 与 输出门时变矩阵C在序列长度维度的第i个切片 进行张量相乘 ，得到 当前时间步输出序列y
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                # 若 输出门时变矩阵C 是高维S6类型
                else:
                    # 当前时间步的隐状态张量x 与 输出门时变矩阵C在序列长度维度的第i个切片 进行张量相乘 ，得到 当前时间步输出序列y
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            # 若 当前迭代次数 达到 隐状态维度大小-1，即最后一次迭代。
            if i == u.shape[2] - 1:
                # 记录 当前时间步的隐状态张量x 为 最后一次迭代的隐状态张量last_state
                last_state = x
            # 当前时间步输出序列y 是复数序列
            if y.is_complex():
                # 当前时间步输出序列y 的实部乘以2
                y = y.real * 2
            # 将 当前时间步输出序列y 加入到 列表ys 中
            ys.append(y)
        # 将 全部时间步的输出序列列表ys 以 按元素堆叠的形式 堆叠为 全部时间步的输出序列y
        y = torch.stack(ys, dim=2)  # (batch dim L)

        # ——————————输出侧——————————
        # 若 残差连接时不变矩阵D 不存在，输出序列Out 即为 全部时间步的输出序列y
        # 若 残差连接时不变矩阵D 存在，输出序列Out 即为 全部时间步的输出序列y 数值相加 重排为(输入s6维度, 1)形状的残差连接时不变矩阵D
        out = y if D is None else y + u * rearrange(D, "d -> d 1")

        # 支路输入序列z 存在
        if z is not None:
            # 输出序列Out 即 输出序列Out 乘以 经过silu激活函数的支路输入序列z
            out = out * F.silu(z)

        # 输出序列Out 转换为与 输入序列u 一样的数据格式
        out = out.to(dtype=dtype_in)
        # 若 return_last_state 挂起，返回 元组(输出序列Out, 最后一次迭代的隐状态张量last_state)
        # 若 return_last_state 置零，返回 输出序列Out
        return out if not return_last_state else (out, last_state)

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        # 获取x的张量形状参数
        B, C, H, W = x.shape
        # H*W得到序列长度L
        L = H * W
        K = 4

        # 将x(B,C,H,W)展平为x1(B,C,H*W(L)),-1表示自适应维度.这里表示横向扫描
        # x1 = x.view(B, -1, L)
        # 将x(B,C,H,W)的H,W维度转置。因为transpose后张量在内存中不连续，使用contiguous()使其连续。随后展平为x2(B,C,H*W(L)),-1表示自适应维度。这里表示纵向扫描
        # x2 = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        # 将x1 x2以torch.stack形式堆叠为(B, 2, C, L)
        # x_hwwh = torch.stack([x1,x2], dim=1).view(B, 2, -1, L)
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)

        # 将x_hwwh沿着(B, 2, -1, L)中得L维度翻转，构造 反向横向扫描与反向纵向扫描 的组合张量。x3仍为(B, 2, -1, L)
        # x3 = torch.flip(x_hwwh, dims=[-1])
        # x3与x_hwwh在通道维度上拼接，构造[正向横向扫描, 正向纵向扫描, 反向横向扫描, 反向纵向扫描]张量。xs为(B, 4, -1, L)
        # xs = torch.cat([x_hwwh, x3], dim=1)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        # xs与S6的project模块的权值进行矩阵相乘。
        # project模块的权值是低秩SΔ(x)、Sb(x)、Sc(x)的线性总权值，project模块权值的维度为 秩维度+隐状态维度+隐状态维度
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # #x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        # 将x_dbl分割为(0-self.dt_rank)->dts; (self.dt_rank-self.d_state)->Bs; (self.d_state-2*self.d_state)->Cs的三部分
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # 将dts与dt_projs_weight进行矩阵相乘，完成从低秩向隐状态维度的投影
        # dt_projs_weight是维度是xt的输入维度，可以将dts升维到xst的输入维度。自此dts才是真正的SΔ(x)。
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # #dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        # 将初始化的D矩阵浮点化
        Ds = self.Ds.float().view(-1)  # (k * d)
        # 将初始化的A矩阵进行离散化，并调整形状为(B, xt输入维度, 隐状态维度)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # 将xs四向扫描输入 与 时变权重SΔ(x)、Bt、Ct 与 时不变权值 As,Ds 传入s6主模块中
        # out_y为四向扫描的堆叠输出

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        r'''
        out_y = self.selective_scan_ref(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)'''
        assert out_y.dtype == torch.float
        # 现在定义out_y的张量格式为（批次大小, 扫描路径, 输入的维度, 序列长度）

        # out_y[:, 2:4]就是取out_y 的扫描路径的后两个切片([反向横向扫描, 反向纵向扫描])。
        # 在序列长度的维度上进行翻转，就是将[反向横向扫描, 反向纵向扫描]取消反向。
        # .view(B, 2, -1, L)就是重新将其变形为(B,2,输入的维度,序列长度)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        # out_y[:, 1]就是取out_y 的扫描路径的第二个切片(正向纵向扫描)
        # 对out_y[:, 1]转置就是化纵向扫描为横向扫描
        # .view(B, -1, L)就是重新将其变形为(B,输入的维度,序列长度)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # inv_y[:, 1]就是取inv_y 的扫描路径的第二个切片(去反的反向纵向扫描)
        # 对inv_y[:, 1]转置就是化纵向扫描为横向扫描
        # .view(B, -1, L)就是重新将其变形为(B,输入的维度,序列长度)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 输出四个横向扫描序列
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        # 进入线性层（双倍）升维
        xz = self.in_proj(x)
        # 将线性层（双倍）升维后的输出 ，将x，z分离
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        # 重更改为(B, C, H, W),因为permute后张量在内存中不连续，使用contiguous()使其连续。
        x = x.permute(0, 3, 1, 2).contiguous()
        # 经过DW卷积 通道数不变
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        # 进入SS2D，输出[四个原序列]的结果
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        # 将四个原序列的输出数值相加
        y = y1 + y2 + y3 + y4

        # 将y从(B,C,L)转置为(B,L,C)。因为transpose后张量在内存中不连续，使用contiguous()使其连续。然后重新展开为(B, H, W, C)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # 进行层归一化
        y = self.out_norm(y)
        # 进行激活函数非线性化
        y = y * F.silu(z)
        # 进行线性层重新降维
        out = self.out_proj(y)
        # 若随机失活标志位挂起，则进行随机失活
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self,
                 # 本层VSSLayer输出维度
                 hidden_dim: int = 0,
                 # VSSBlock随机失活比率
                 drop_path: float = 0,
                 # 层归一化
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
                 # SS2D随机失活比率
                 attn_drop_rate: float = 0,
                 # SSM的状态维度
                 d_state: int = 16,
                 # SS2D内部数据扩张维度
                 ss2d_expand_ratio: int = 2,
                 **kwargs,
                 ):
        super().__init__()
        # 定义 层归一化
        self.ln_1 = norm_layer(hidden_dim)
        # 定义 SS2D模块
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, expand=ss2d_expand_ratio, **kwargs)
        # 定义 随机失活
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # 等效于：
        # 首先经过层归一化，然后输入SS2D模块，最后进行VSSBlock的随机失活
        # x = self.drop_path(self.self_attention(self.ln_1(input)))
        # 与原始输入进行残差连接
        # x = input + x
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    # VSSLayer包含 VSSBlock与PatchMerging2D
    # PatchMerging2D随downsample标志位是否启用
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            # 本层VSSLayer输出维度
            dim,
            # VSSBlock级联深度
            depth,
            # SS2D模块失活率
            attn_drop=0.,
            # VSSBlock随机失活率
            drop_path=0.,
            # 层归一化
            norm_layer=nn.LayerNorm,
            # 是否包含下采样
            downsample=None,
            # 是否使用checkpoint
            use_checkpoint=False,
            # 隐状态维度
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        # 定义 本层VSSLayer输出维度
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        # 定义 本层的VSSBlock
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            # 以 VSSBlock级联深度 迭代
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            # 若启用downsample，则定义 PatchMerging2D
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # 以 VSSBlock级联深度 迭代
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                # 输入进VSSBlock
                x = blk(x)

        # 若 本层VSSLayer 存在 PatchMerging2D
        if self.downsample is not None:
            x_downsample = self.downsample(x)
            return x, x_downsample
        else:
            return x


class VSSLayer_up(nn.Module):
    # VSSLayer_up包含 PatchExpand2D与VSSBlock
    # PatchExpand2D随upsample标志位是否启用
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            # 本层VSSLayer_up的输出维度
            dim,
            # VSSBlock级联深度
            depth,
            # SS2D模块失活率
            attn_drop=0.,
            # VSSBlock随机失活率
            drop_path=0.,
            # 层归一化
            norm_layer=nn.LayerNorm,
            # 是否包含上采样
            upsample=None,
            use_checkpoint=False,
            # 隐状态维度
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        # 定义 本层VSSLayer_up的输出维度
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # 定义 本层的VSSBlock
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            # 以 VSSBlock级联深度 迭代
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            # 若启用upsample，则定义 PatchExpand2D
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        # 若 本层VSSLayer_up 存在 PatchExpand2D
        if self.upsample is not None:
            x = self.upsample(x)

        # 以 VSSBlock级联深度 迭代
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self,
                 # Patch大小
                 patch_size=4,
                 # 输入图像的通道数
                 in_chans=3,
                 # 分类类别数量
                 num_classes=1,
                 # 每Encoder层的VSS Block数
                 depths=[2, 2, 2, 2],
                 # 每Decoder层的VSS Block数
                 depths_decoder=[2, 9, 2, 2],
                 # 每Encoder层的输出通道数
                 dims=[96, 192, 384, 768],
                 # 每Decoder层的输出通道数
                 dims_decoder=[768, 384, 192, 96],
                 # ssm隐状态的维度
                 d_state=16,
                 # VSSM随机失活率
                 drop_rate=0.,
                 # SS2D随机失活率
                 attn_drop_rate=0.,
                 # VSSLayer(_up)随机失活率
                 drop_path_rate=0.1,
                 # 层批量归一化
                 norm_layer=nn.LayerNorm,
                 # Patch归一化
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs
                 ):
        super().__init__()
        # 定义 分类类别数量
        self.num_classes = num_classes
        # 定义 网络深度
        self.num_layers = len(depths)
        # 若dims是整数
        if isinstance(dims, int):
            # 定义 经过每层后的输出通道数
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        # 定义 Embedding层维度层数
        self.embed_dim = dims[0]
        # 没有用法
        self.num_features = dims[-1]
        # 没有用法
        self.dims = dims

        # 定义 patch-token：Patch Embedding参数
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        # 定义 positional embedding是否启用绝对位置编码
        '''self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        # 若启用绝对位置编码
        if self.ape:
            # 定义 patch-token的分辨率
            self.patches_resolution = self.patch_embed.patches_resolution
            # 定义 与patch-token同shape的全零填充的可学习嵌入参数
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            # 定义 以高斯分布 初始化全零填充的可学习嵌入参数
            trunc_normal_(self.absolute_pos_embed, std=.02)'''
        # 定义 绝对位置编码随机失活
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 定义 各层VSSLayer/VSSLayer_up随机层间丢弃比率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        '''dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]'''

        # 定义模块列表以批量定义
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 定义 一个VSSLayer
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        '''self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 定义 一个VSSLayer_up
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        # 定义 最后的Final Projection
        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        # 定义 分类逐点卷积
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)'''

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder
    def forward_features(self, x):
        # 创建 记录各层进入VSSLayer前的输入 的列表
        skip_list = []
        # 进行 Patch Embedding
        x = self.patch_embed(x)
        # 若启用绝对位置编码
        '''if self.ape:
            x = x + self.absolute_pos_embed'''
        # 进行 绝对位置编码随机失活
        x = self.pos_drop(x)
        # skip_list.append(x)

        # 根据 VSSLayer数量 迭代
        for layer in self.layers:
            # 进入VSSLayer
            x = layer(x)
            # 自适应解包返回结果
            if isinstance(x, tuple):
                x_skipconnection, x = x  # 存在下采样时的返回
                # 记录各层进入VSSLayer前的输入
                skip_list.append(x_skipconnection)
            else:
                x = x  # 无下采样时的返回
        return x, skip_list

    # Decoder
    def forward_features_up(self, x, skip_list):
        # 根据 VSSLayer_up 迭代
        for inx, layer_up in enumerate(self.layers_up):
            # 若是 第一个 VSSLayer_up
            if inx == 0:
                x = layer_up(x)
            # 若不是 第一个 VSSLayer_up
            else:
                # VSSLayer_up的输入为x 与 倒序取出的skip_list
                x = layer_up(x + skip_list[-inx])
        return x

    # Final
    def forward_final(self, x):
        # 进入 Final Projection
        x = self.final_up(x)
        # B H W C → B C H W
        x = x.permute(0, 3, 1, 2)
        # 进入 分类逐点卷积
        x = self.final_conv(x)
        return x

    # 没有用法
    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    # VSSM全过程
    def forward(self, x):
        x, skip_list = self.forward_features(x)
        #x = self.forward_features_up(x, skip_list)
        #x = self.forward_final(x)
        skip_list.append(x)
        return skip_list
