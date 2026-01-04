import math
import numpy as np
import torch
import einops
import torch.nn.functional as F

class HybridFrequencyAttentionBlock(torch.nn.Module):
    def __init__(self, input_channels, input_size, reduction_ratio=8, conv_ratio=1):
        super().__init__()
        self.conv_ratio = conv_ratio
        if input_size % 7 == 0:
            self.patch_size = 7
            self.patch_number = input_size // 7
        elif input_size % 8 == 0:
            self.patch_size = 8
            self.patch_number = input_size // 8
        else: assert 'Error'

        kernel_size = int(abs((math.log(input_channels, 2) + 1) / 2))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1
        padding = kernel_size // 2


        self.frequency_coefficient0 = torch.nn.Parameter(torch.rand(4))
        self.frequency_coefficient1 = torch.nn.Parameter(torch.rand(2))
        self.offset_coefficient = torch.nn.Parameter(torch.rand(1))
        self.register_buffer('precomputed_matrix0', self.construct_all_frequency_matrix(self.patch_size, self.patch_size, self.patch_size, self.patch_size))
        self.register_buffer('precomputed_matrix1',
                             self.construct_all_frequency_matrix(input_size, input_size, self.patch_size, self.patch_size))
        self.register_buffer('zero_offset', torch.zeros(1, 1, 1, 1))
        self.conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding),
            torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding=padding),
            torch.nn.Sigmoid()
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels//self.conv_ratio, out_channels=input_channels//self.conv_ratio, kernel_size=3, padding=1, stride=1, groups=input_channels//self.conv_ratio),
            torch.nn.Sigmoid()
        )
        self.norm = torch.nn.BatchNorm2d(input_channels)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_original = x

        x_dct = torch.einsum('b c h w, v u h w -> b c v u', x, self.precomputed_matrix1)
        x_BC = self.construct_2DDCT_frequency_feature_value(B, C, x_dct)
        x_BC11 = self.conv1d(x_BC).view(B, C, 1, 1)

        x_patch = einops.rearrange(x, 'b c (m y) (n x) -> b c m n y x', x=self.patch_size, y=self.patch_size)
        x_patch_dct = torch.einsum('b c m n x y, v u x y -> b c m n v u', x_patch, self.precomputed_matrix0)
        x_BCMN = self.construct_patch_feature_value(x_patch_dct)
        x_BCMN_0 = x_BCMN[:, :C//self.conv_ratio, :, :]
        x_BCMN_weighted = self.conv(x_BCMN_0)
        x_sw_BCHW = F.interpolate(torch.cat([x_BCMN_weighted, self.zero_offset.expand(B,C-C//self.conv_ratio,self.patch_number,self.patch_number)], dim=1), size=(H, W), mode='nearest')

        x_CA = x * (x_BC11.expand_as(x) + self.offset_coefficient*x_sw_BCHW)
        x_residual = x_original + x_CA

        x_n = self.norm(x_residual)
        x_sigmoid = self.act(x_n)
        return x_sigmoid

    def construct_patch_feature_value(self, input_BCMNVU):
        low_band = input_BCMNVU[:, :, :, :, 0: 2, 0: 2].sum(dim=-1).sum(dim=-1)
        medium_band = input_BCMNVU[:, :, :, :, 0: 4, 0: 4].sum(dim=-1).sum(dim=-1)
        high_band = input_BCMNVU[:, :, :, :, 0: self.patch_size, 0: self.patch_size].sum(dim=-1).sum(dim=-1)

        medium_band_avg = (medium_band - low_band) / 12
        high_band_avg = (high_band - medium_band) / ((self.patch_size * self.patch_size) - 16)

        BCMN = medium_band_avg * self.frequency_coefficient1[0] + high_band_avg * self.frequency_coefficient1[1]
        return BCMN


    def construct_2DDCT_frequency_feature_value(self, batch, channel, input_BCVU):
        lowest_point = input_BCVU[:, :, 0, 0]
        low_area = input_BCVU[:, :, 0:2, 0:2].sum(dim=-1).sum(dim=-1)
        medium_area = input_BCVU[:, :, 0:4, 0:4].sum(dim=-1).sum(dim=-1)
        high_area = input_BCVU[:, :, 0:self.patch_size, 0:self.patch_size].sum(dim=-1).sum(dim=-1)

        low_area_avg = (low_area - lowest_point) / 3
        medium_area_avg = (medium_area - low_area) / 12
        high_area_avg = (high_area - medium_area) / (self.patch_size*self.patch_size-16)

        BC = lowest_point * self.frequency_coefficient0[0] + low_area_avg * self.frequency_coefficient0[1] + \
             medium_area_avg * self.frequency_coefficient0[2] + high_area_avg * self.frequency_coefficient0[3]

        output_BC = BC.view(batch, 1, channel)
        return output_BC

    def DCT2D_basis_formula_with_normalization(self, time_domain_position_x, time_domain_position_y, time_domain_size_x, time_domain_size_y, frequency_domain_position_u, frequency_domain_position_v):
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


def calculating_params_flops(model, channel, size):
    from thop import profile
    input = torch.randn(1, channel, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print("FLOPs: %.3fK" % (flops / 1e3))
    print("Params: %.3fK" % (params / 1e3))
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fK" % (total / 1e3))

if __name__ == "__main__":
    input_channels = 1024
    input_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = torch.rand([1, 64, 56, 56]).float().to(device)
    model = HybridFrequencyAttentionBlock(input_channels=input_channels, input_size=input_size, ).to(device)
    # out = model(data)
    calculating_params_flops(model, input_channels, input_size)
    # print('out.shape:', out.shape)




















