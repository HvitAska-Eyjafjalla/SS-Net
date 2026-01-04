import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from entmax import sparsemax
import math

class DCTLowPassChannelAttention(nn.Module):
    def __init__(self, input_channels, input_size, reduction_ratio=8):
        super().__init__()
        kernel_size = int(abs((math.log(input_channels, 2) + 1) / 2))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2


        self.frequency_coefficient = torch.nn.Parameter(torch.rand(4))
        self.register_buffer('precomputed_matrix',
                             self.construct_all_frequency_matrix(input_size, input_size, 7, 7))
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding),
            torch.nn.Sigmoid()
        )
        self.norm = torch.nn.BatchNorm2d(input_channels)
        self.act = torch.nn.Sigmoid()


    def forward(self, x):
        B, C, H, W = x.shape
        x_original = x

        x_dct = torch.einsum('b c h w, v u h w -> b c v u', x, self.precomputed_matrix)
        x_BC = self.construct_2DDCT_frequency_feature_value(B, C, x_dct)
        x_BC11 = self.conv1d(x_BC).view(B, C, 1, 1)
        x_CA = x * x_BC11.expand_as(x)
        x_residual = x_original + x_CA
        x_n = self.norm(x_residual)
        x_sigmoid = self.act(x_n)
        return x_sigmoid

    def construct_2DDCT_frequency_feature_value(self, batch, channel, input_BCVU):
        lowest_point = input_BCVU[:, :, 0, 0]
        low_area = input_BCVU[:, :, 0:2, 0:2].sum(dim=-1).sum(dim=-1)
        medium_area = input_BCVU[:, :, 0:4, 0:4].sum(dim=-1).sum(dim=-1)
        high_area = input_BCVU[:, :, 0:7, 0:7].sum(dim=-1).sum(dim=-1)

        low_area_avg = (low_area - lowest_point) / 3
        medium_area_avg = (medium_area - low_area) / 12
        high_area_avg = (high_area - medium_area) / 33

        BC = lowest_point * self.frequency_coefficient[0] + low_area_avg * self.frequency_coefficient[1] + \
             medium_area_avg * self.frequency_coefficient[2] + high_area_avg * self.frequency_coefficient[3]

        output_BC = BC.view(batch, 1, channel)
        return output_BC

    def DCT2D_basis_formula_with_normalization(self, time_domain_position_x, time_domain_position_y,
                                               time_domain_size_x, time_domain_size_y,
                                               frequency_domain_position_u, frequency_domain_position_v):
        basis = np.cos(
            ((2 * time_domain_position_x + 1) * frequency_domain_position_u * np.pi) / (2 * time_domain_size_x)) * \
                np.cos(
                    ((2 * time_domain_position_y + 1) * frequency_domain_position_v * np.pi) / (2 * time_domain_size_y))

        if frequency_domain_position_u == 0:
            normalization_coefficient_u = 1 / np.sqrt(time_domain_size_x)
        else:
            normalization_coefficient_u = np.sqrt(2) / np.sqrt(time_domain_size_x)

        if frequency_domain_position_v == 0:
            normalization_coefficient_v = 1 / np.sqrt(time_domain_size_y)
        else:
            normalization_coefficient_v = np.sqrt(2) / np.sqrt(time_domain_size_y)

        normalized_basis = basis * normalization_coefficient_u * normalization_coefficient_v
        return normalized_basis

    def construct_single_frequency_matrix(self, time_domain_size_x, time_domain_size_y, frequency_domain_position_u,
                                          frequency_domain_position_v):
        single_frequency_matrix = torch.zeros(time_domain_size_x, time_domain_size_y)

        for time_domain_position_y in range(0, time_domain_size_y):
            for time_domain_position_x in range(0, time_domain_size_x):
                single_frequency_matrix[time_domain_position_y, time_domain_position_x] = \
                    self.DCT2D_basis_formula_with_normalization(time_domain_position_x, time_domain_position_y,
                                                                time_domain_size_x, time_domain_size_y,
                                                                frequency_domain_position_u,
                                                                frequency_domain_position_v)

        return single_frequency_matrix

    def construct_all_frequency_matrix(self, time_domain_size_x, time_domain_size_y, frequency_domain_size_u,
                                       frequency_domain_size_v):
        all_frequency_matrix = torch.zeros(frequency_domain_size_u, frequency_domain_size_v, time_domain_size_x,
                                           time_domain_size_y)

        for frequency_domain_position_v in range(0, frequency_domain_size_v):
            for frequency_domain_position_u in range(0, frequency_domain_size_u):
                single_frequency_matrix = self.construct_single_frequency_matrix(time_domain_size_x, time_domain_size_y,
                                                                                 frequency_domain_position_u,
                                                                                 frequency_domain_position_v)

                all_frequency_matrix[frequency_domain_position_v, frequency_domain_position_u, :,
                :] = single_frequency_matrix

        return all_frequency_matrix

class DWTFusionModule(nn.Module):
    def __init__(self,
                 hr_input_channels: int,
                 hr_input_size: int,
                 lr_input_channels: int,
                 compress_ratio:int,
                 groups: int = 1,
                 hr_up_scale: int = 2,
                 kernel_size_lpf: int = 5,
                 kernel_size_hpf: int = 3,
                 ):
        super().__init__()
        self.groups = groups
        self.hr_up_scale = hr_up_scale
        self.kernel_size_lpf = kernel_size_lpf
        self.kernel_size_hpf = kernel_size_hpf
        self.compress_channel = lr_input_channels // compress_ratio

        self.dct_attention = DCTLowPassChannelAttention(input_channels=hr_input_channels,
                                                        input_size=hr_input_size)
        self.point_wise_conv_init_lr = nn.Sequential(
            nn.Conv2d(in_channels=hr_input_channels, out_channels=lr_input_channels, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=2)

        self.register_buffer('haar_low', torch.tensor([1., 1.]) / np.sqrt(2))
        self.register_buffer('haar_high', torch.tensor([1., -1.]) / np.sqrt(2))

        self.mask_conv_padding_LPF = nn.ReflectionPad2d(padding=kernel_size_lpf//2)
        if self.compress_channel != lr_input_channels:
            self.mask_conv_compress_LPF = nn.Sequential(
                nn.Conv2d(in_channels=lr_input_channels, out_channels=self.compress_channel, kernel_size=1),
            )
        else:
            self.mask_conv_compress_LPF = None
        self.mask_conv_LPF = nn.Sequential(
            nn.Conv2d(in_channels=self.compress_channel, out_channels=self.compress_channel, kernel_size=kernel_size_lpf, groups=self.compress_channel),
            nn.Conv2d(in_channels=self.compress_channel, out_channels=groups * (kernel_size_lpf ** 2), kernel_size=1),
            nn.ReLU()
        )
        self.batch_norm_LPF = nn.BatchNorm2d(groups*(kernel_size_lpf**2))
        self.register_buffer('hamming_lowpass', torch.FloatTensor(self.hamming2D(kernel_size_lpf, kernel_size_lpf))[None, None,])


        self.mask_conv_padding_HPF = nn.ReflectionPad2d(padding=kernel_size_hpf // 2)
        if self.compress_channel != lr_input_channels:
            self.mask_conv_compress_HPF = nn.Sequential(
                nn.Conv2d(in_channels=lr_input_channels, out_channels=self.compress_channel, kernel_size=1),
            )
        else:
            self.mask_conv_compress_HPF = None
        self.mask_conv_HPF = nn.Sequential(
            nn.Conv2d(in_channels=self.compress_channel, out_channels=self.compress_channel, kernel_size=kernel_size_hpf, groups=self.compress_channel),
            nn.Conv2d(in_channels=self.compress_channel, out_channels=groups * (kernel_size_hpf ** 2), kernel_size=1),
            nn.ReLU()
        )
        self.batch_norm_HPF = nn.BatchNorm2d(groups * (kernel_size_hpf ** 2))

        self.hr_vector_upsampling = VectorSampling(in_channels=lr_input_channels, groups=self.groups)

        self.tau = LearnableTemperature(init_tau=1, tau_min=0.5, tau_max=2)

        self.CBR = nn.Sequential(
            nn.Conv2d(in_channels=lr_input_channels, out_channels=lr_input_channels, kernel_size=kernel_size_hpf, padding=kernel_size_hpf//2, groups=lr_input_channels),
            nn.BatchNorm2d(num_features=lr_input_channels),
        )

        self.apply(self._init_module_weights)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)


    def forward(self, hr_feature, lr_feature):
        B, C_hr, H_hr, W_hr = hr_feature.shape
        B, C_lr, H_lr, W_lr = lr_feature.shape

        # Primary Fusion Stage
        hr_Attention = self.dct_attention(hr_feature)
        hr_channel_changed = self.point_wise_conv_init_lr(hr_Attention)
        hr_channel_changed_2X = F.interpolate(input=hr_channel_changed,
                                              size=None,
                                              scale_factor=self.hr_up_scale,
                                              mode='bilinear',
                                              align_corners=None)
        primary_fusion = hr_channel_changed_2X + lr_feature
        LL, LH, HL, HH = self.dwt_haar_custom(primary_fusion, haar_low=self.haar_low, haar_high=self.haar_high)

        # Final Fusion Stage
        # HF Part
        comprehensive_HF = self.comprehensive_high_frequency_response_map_generator(LH, HL, HH)
        comprehensive_HF_2X = F.pixel_shuffle(comprehensive_HF, 2)
        mask_HPF = self.filter_mask_generator(input=comprehensive_HF_2X,
                                              kernel_size=self.kernel_size_hpf,
                                              groups=self.groups,
                                              padding_module=self.mask_conv_padding_HPF,
                                              compress_conv_module=self.mask_conv_compress_HPF,
                                              mask_conv_module=self.mask_conv_HPF,
                                              batch_norm_module=self.batch_norm_HPF,)
        lr_HPF = self.apply_filter(input=lr_feature,
                                   mask=mask_HPF,
                                   kernel_size=self.kernel_size_hpf,
                                   padding_module=self.mask_conv_padding_HPF)
        lr_residual = lr_feature + lr_HPF

        # LF - Kernel Mask Part
        mask_LPF = self.filter_mask_generator(input=LL,
                                              kernel_size=self.kernel_size_lpf,
                                              groups=self.groups,
                                              padding_module=self.mask_conv_padding_LPF,
                                              compress_conv_module=self.mask_conv_compress_LPF,
                                              mask_conv_module=self.mask_conv_LPF,
                                              batch_norm_module=self.batch_norm_LPF,
                                              hamming_window=self.hamming_lowpass)
        hr_channel_changed_LPF = self.apply_filter(input=hr_channel_changed,
                                                   mask=mask_LPF,
                                                   kernel_size=self.kernel_size_lpf,
                                                   padding_module=self.mask_conv_padding_LPF)
        hr_channel_changed_LPF = hr_channel_changed_LPF + hr_channel_changed
        # LF - Cos-Similarity Offset Part
        hr_channel_changed_LPF_2X = self.hr_vector_upsampling(self.dropout1(hr_channel_changed_LPF))

        result = self.channel_shuffle(hr_channel_changed_LPF_2X, self.groups) + self.channel_shuffle(self.dropout2(lr_residual), self.groups)
        # result = hr_channel_changed_LPF_2X + lr_residual
        return self.CBR(result)

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # Reshape -> Transpose -> Flatten
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Swap groups and channels
        x = x.view(batch_size, -1, height, width)  # Flatten
        return x

    def _init_module_weights(self, module):
        if isinstance(module, nn.Conv2d):
            self._init_conv(module)
        elif isinstance(module, nn.BatchNorm2d):
            self._init_bn(module)

    def _init_conv(self, module):
        is_hpf = any([module in seq for seq in [
            self.mask_conv_HPF
        ]])

        if is_hpf:
            nn.init.xavier_uniform_(module.weight,
                                    gain=nn.init.calculate_gain('relu', param=0.2))
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, 0.01)
        else:
            nn.init.kaiming_normal_(module.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        if module.in_channels == module.out_channels:
            nn.init.orthogonal_(module.weight)

        if module == self.mask_conv_HPF[-1]:
            nn.init.normal_(module.weight, 0, 0.001)
            nn.init.constant_(module.bias, 0)

    def _init_bn(self, module):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0)
        module.reset_running_stats()

    def dwt_haar_custom(self, x: torch.Tensor, haar_low, haar_high):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0

        x = rearrange(x, 'b c (h s_h) w -> b c h w s_h', s_h=2, h=H // 2)
        L_rows = (x * haar_low.view(1, 1, 1, 1, 2)).sum(dim=-1)
        H_rows = (x * haar_high.view(1, 1, 1, 1, 2)).sum(dim=-1)

        L_cols = rearrange(L_rows, 'b c h (w s_w) -> b c h w s_w', s_w=2, w=W // 2)
        LL = (L_cols * haar_low.view(1, 1, 1, 1, 2)).sum(dim=-1)
        HL = (L_cols * haar_high.view(1, 1, 1, 1, 2)).sum(dim=-1)

        H_cols = rearrange(H_rows, 'b c h (w s_w) -> b c h w s_w', s_w=2, w=W // 2)
        HH = (H_cols * haar_high.view(1, 1, 1, 1, 2)).sum(dim=-1)
        LH = (H_cols * haar_low.view(1, 1, 1, 1, 2)).sum(dim=-1)

        return LL, LH, HL, HH

    def comprehensive_high_frequency_response_map_generator(self, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor):
        B, C, H, W = LH.shape

        response_abs = torch.stack([LH.abs(), HL.abs(), HH.abs()], dim=1)
        max_val, _ = response_abs.view(B, 3, C, -1).max(dim=-1, keepdim=True)
        normalized = response_abs / (max_val.unsqueeze(-1) + 1e-8)

        def sample_gumbel(shape):
            U = torch.rand(shape).clamp(1e-6, 1 - 1e-6)
            return -torch.log(-torch.log(U)).to(LH.device)

        gumbel_noise = sample_gumbel(normalized.shape)

        perturbed_logits = (normalized + gumbel_noise) / self.tau.tau

        sparse_weights = sparsemax(perturbed_logits, dim=1)

        mask = (sparse_weights * torch.stack([LH, HL, HH], dim=1))

        duplicated = mask.sum(dim=1)

        combined = torch.stack([(duplicated+LH+HL+HH)/4, LH, HL, HH], dim=2)  # [B,C,4,H,W]

        combined = F.layer_norm(combined.view(B, 4 * C, H, W), [H, W])
        return combined

    def hamming2D(self, M: int, N: int):
        hamming_x = np.hamming(M)
        hamming_y = np.hamming(N)
        hamming_2d = np.outer(hamming_x, hamming_y)
        return hamming_2d

    def filter_mask_generator(self, input, kernel_size, groups, padding_module, compress_conv_module, mask_conv_module, batch_norm_module, hamming_window=None):
        B, C, H, W = input.shape
        if compress_conv_module is not None:
            input = compress_conv_module(input)
        mask_BGKHW = mask_conv_module(padding_module(input)) # B G*K^2 H W
        mask_norm_BGKHW = batch_norm_module(mask_BGKHW).view(B, groups, kernel_size**2, H, W) # B G K^2 H W
        mask_softmax_BGHWK = self.softmax(mask_norm_BGKHW).permute(0, 1, 3, 4, 2).contiguous() # B G H W K^2
        mask_BGLKK = mask_softmax_BGHWK.view(B, groups, H, W, kernel_size, kernel_size).view(B, -1, kernel_size, kernel_size) # B G H W K K -> B G*H*W K K

        # if apply hamming_window, the mask will be LPF
        if hamming_window is not None:
            mask_hamming = mask_BGLKK * hamming_window
            mask = mask_hamming / mask_hamming.sum(dim=(-1, -2), keepdims=True) # mask : B G*H*W K K
        # HPF Filter
        else:
            # mask = mask_BGLKK - mask_BGLKK.mean(dim=(-1, -2), keepdim=True)
            mask = mask_BGLKK / mask_BGLKK.sum(dim=(-1, -2), keepdims=True)  # mask : B G*H*W K K

        mask = mask.contiguous()
        # B G*H*W K K -> B G*H*W K^2 -> B G H*W K^2 -> B G H W K^2 -> B G K^2 H W
        mask_final = mask.view(B, -1, kernel_size**2).view(B, groups, -1, kernel_size**2).view(B, groups, H, W, kernel_size**2).permute(0, 1, 4, 2, 3).contiguous()
        return mask_final.unsqueeze(2) # B G 1 K^2 H W

    def apply_filter(self, input, mask, kernel_size, padding_module):
        B, C, H, W = input.shape
        B, G, _, K, __, ___ = mask.shape
        assert H==__ and W==___
        input_unfold = F.unfold(padding_module(input), kernel_size=(kernel_size, kernel_size), stride=1, padding=0).reshape(B, C, kernel_size**2, H, W)
        input_group = input_unfold.view(B, G, -1, kernel_size**2, H, W) # B, G, C//G, K^2, H, W

        apply_kernel = torch.sum(input_group * mask, dim=3)
        return apply_kernel.view(B, -1, H, W) # B C H W


class VectorSampling(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4, dyscope=True):
        super().__init__()
        self.scale = scale
        self.groups = groups

        assert in_channels >= groups and in_channels % groups == 0

        out_channels = 2 * groups * scale ** 2

        self.direction_offset = nn.Conv2d(2 * groups, out_channels, kernel_size=3, padding=1, stride=1)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())
        self.direction_sampler = CosineSimilarityDirectionVectorgenerator(group=groups, window_size=3, top_k=3)

        self.group_norm_hr = nn.GroupNorm(num_groups=groups, num_channels=in_channels)

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        result =  torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)
        return result

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.group_norm_hr(x)
        direction_vector = self.direction_sampler(x).view(B, -1, H, W) # B, 2, G, H, W -> B, 2*G, H, W
        if hasattr(self, 'scope'):
            offset = self.direction_offset(direction_vector) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.direction_offset(direction_vector) * 0.25 + self.init_pos
        return self.sample(x, offset)


class CosineSimilarityDirectionVectorgenerator(nn.Module):
    def __init__(self, group, window_size, top_k=3, init_tau=5):
        super().__init__()
        self.group = group
        self.window_size = window_size
        self.top_k = top_k
        self.register_buffer('d8_neighborhood_vector', F.normalize(torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                                                                                 [0, -1], [0, 1],
                                                                                 [1, -1], [1, 0], [1, 1]],
                                                                                dtype=torch.float32), p=2, dim=1))
        self.tau_module = LearnableTemperature(init_tau=5, tau_min=1, tau_max=10)

    def compute_group_similarity(self, input, groups=2, window_size=3, dilation=1):
        B, C, H, W = input.shape
        assert C % groups == 0
        group_channels = C // groups

        grouped_input = input.view(B, groups, group_channels, H, W)

        unfolded = F.unfold(grouped_input.view(B * groups, group_channels, H, W),
                            kernel_size=window_size,
                            padding=(window_size // 2) * dilation,
                            dilation=dilation)

        unfolded = unfolded.view(B, groups, group_channels, window_size ** 2, H, W)

        center_idx = window_size ** 2 // 2
        center = unfolded[..., center_idx:center_idx + 1, :, :]  # [B, G, Cg, 1, H, W]
        neighbors = torch.cat([unfolded[..., :center_idx, :, :],
                               unfolded[..., center_idx + 1:, :, :]], dim=3)  # [B, G, Cg, K^2-1, H, W]

        center_expanded = center.expand(-1, -1, -1, window_size ** 2 - 1, -1, -1)

        similarity = F.cosine_similarity(center_expanded, neighbors, dim=2)  # [B, G, K^2-1, H, W]
        return similarity

    def _differentiable_topk_mask(self, weights: torch.Tensor) -> torch.Tensor:
        B, G, N, H, W = weights.shape
        top_k = self.top_k

        with torch.no_grad():
            sorted_vals, sorted_indices = torch.sort(weights, dim=2, descending=True)
            kth_vals = sorted_vals[:, :, top_k - 1:top_k]  # [B,G,1,H,W]

        diff = weights - kth_vals.detach()
        mask = torch.sigmoid(diff / self.tau_module.tau * 20)

        return mask


    def forward(self, input):
        similarity = self.compute_group_similarity(input=input, groups=self.group, window_size=self.window_size)

        topk_mask = self._differentiable_topk_mask(similarity)
        sparse_weights = similarity * topk_mask

        weighted_dir = torch.einsum('b g n h w,n d->b g h w d', sparse_weights, self.d8_neighborhood_vector)
        # B 2 G H W
        return weighted_dir.permute(0, 4, 1, 2, 3).contiguous()


class LearnableTemperature(nn.Module):
    def __init__(self, init_tau=0.5, tau_min=0.3, tau_max=1):
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        scale = (init_tau - tau_min) / (tau_max - tau_min)
        self.raw_tau = nn.Parameter(torch.logit(torch.tensor(scale)))

    @property
    def tau(self):
        return torch.sigmoid(self.raw_tau) * (self.tau_max - self.tau_min) + self.tau_min


def calculating_params_flops(model, hr_channel, hr_size, lr_channel, lr_size):
    from thop import profile
    hr = torch.randn(1, hr_channel, hr_size, hr_size).cuda()
    lr = torch.randn(1, lr_channel, lr_size, lr_size).cuda()
    flops, params = profile(model, inputs=(hr,lr))
    print("FLOPs: %.4fG" % (flops / 1e9))
    print("Params: %.4fM" % (params / 1e6))
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.4fM" % (total / 1e6))

def calculating_params_summary(model, hr_channel, hr_size, lr_channel, lr_size):
    from torchsummary import summary
    summary(model,
            input_size=[(hr_channel, hr_size, hr_size), (lr_channel, lr_size, lr_size)],
            device='cuda')

if __name__ == "__main__":
    hr_input_channel = 768
    lr_input_channel = hr_input_channel // 2
    hr_iput_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DWTFusionModule(hr_input_channels=hr_input_channel,
                            hr_input_size=hr_iput_size,
                            lr_input_channels=lr_input_channel,
                            compress_ratio=2,
                            groups= 4,
                            hr_up_scale= 2,
                            kernel_size_lpf= 5,
                            kernel_size_hpf= 3,).to(device)

    #out = model(data)
    calculating_params_flops(model, hr_input_channel, hr_iput_size, lr_input_channel, hr_iput_size*2)
    # calculating_params_summary(model, hr_input_channel, hr_iput_size, lr_input_channel, hr_iput_size*2)















