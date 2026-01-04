from .DTSBlock import DTSBlock
from .DWTFusion import DWTFusionModule
from .pvtv2 import PyramidVisionTransformerImpr
from functools import partial
from .vmamba import PatchEmbed2D, Final_PatchExpand2D
from .HFALayer import HybridFrequencyAttentionBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


class SS_UNet(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_size=224,
                 load_ckpt_path=True):
        super().__init__()
        self.load_ckpt_path = load_ckpt_path
        # 64 32
        self.backbone = PyramidVisionTransformerImpr(patch_size=4, embed_dims=[64, 128], num_heads=[1, 2], mlp_ratios=[8, 8],
                                                     qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4], sr_ratios=[8, 4],
                                                     drop_rate=0.0, drop_path_rate=0.1)


        self.pwc2 = nn.Sequential(
            PatchEmbed2D(patch_size=2, in_chans=128, embed_dim=256, norm_layer=nn.LayerNorm),
            BHWC2BCHW(),
        )
        self.encoder2 = nn.Sequential(
            BCHW2BHWC(),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),

            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),

            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=2, input_size=input_size // 16),
            BHWC2BCHW(),
        )  # 16

        self.pwc3 = nn.Sequential(
            PatchEmbed2D(patch_size=2, in_chans=256, embed_dim=512, norm_layer=nn.LayerNorm),
            BHWC2BCHW(),
        )
        self.encoder3 = nn.Sequential(
            BCHW2BHWC(),
            DTSBlock(input_channels=512, patch=2, input_size=input_size // 32),
            DTSBlock(input_channels=512, patch=2, input_size=input_size // 32),
            DTSBlock(input_channels=512, patch=2, input_size=input_size // 32),
            BHWC2BCHW(),
        )  # 8

        self.decoder2 = nn.Sequential(
            BCHW2BHWC(),
            DTSBlock(input_channels=256, patch=4, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=4, input_size=input_size // 16),
            DTSBlock(input_channels=256, patch=4, input_size=input_size // 16),
            BHWC2BCHW(),
        )

        self.decoder1 = nn.Sequential(
            BCHW2BHWC(),
            DTSBlock(input_channels=128, patch=4, input_size=input_size // 8),
            DTSBlock(input_channels=128, patch=4, input_size=input_size // 8),
            DTSBlock(input_channels=128, patch=4, input_size=input_size // 8),
            BHWC2BCHW(),
        )

        self.decoder0 = nn.Sequential(
            BCHW2BHWC(),
            DTSBlock(input_channels=64, patch=8, input_size=input_size // 4, bottleneck=4),
            DTSBlock(input_channels=64, patch=8, input_size=input_size // 4, bottleneck=4),
            DTSBlock(input_channels=64, patch=8, input_size=input_size // 4, bottleneck=4),
        )

        self.feature_fusion_0 = DWTFusionModule(hr_input_channels=128,
                                                hr_input_size=input_size // 8,  # 32
                                                lr_input_channels=64,
                                                compress_ratio=1,
                                                groups=4,
                                                hr_up_scale=2,
                                                kernel_size_lpf=5,
                                                kernel_size_hpf=3, )

        self.feature_fusion_1 = DWTFusionModule(hr_input_channels=256,
                                                hr_input_size=input_size // 16,  # 16
                                                lr_input_channels=128,
                                                compress_ratio=2,
                                                groups=4,
                                                hr_up_scale=2,
                                                kernel_size_lpf=5,
                                                kernel_size_hpf=3, )

        self.feature_fusion_2 = DWTFusionModule(hr_input_channels=512,
                                                hr_input_size=input_size // 32,  # 8
                                                lr_input_channels=256,
                                                compress_ratio=2,
                                                groups=4,
                                                hr_up_scale=2,
                                                kernel_size_lpf=5,
                                                kernel_size_hpf=3, )


        # self.HFA0 = nn.Identity()
        # self.HFA1 = nn.Identity()
        # self.HFA2 = nn.Identity()
        self.HFA2 = HybridFrequencyAttentionBlock(input_channels=256, input_size=input_size // 16)
        self.HFA1 = HybridFrequencyAttentionBlock(input_channels=128, input_size=input_size // 8)
        self.HFA0 = HybridFrequencyAttentionBlock(input_channels=64, input_size=input_size // 4)


        self.final_expanding = nn.Sequential(
            Final_PatchExpand2D(dim=64, dim_scale=4),
            BHWC2BCHW(),
        )
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)
        self.activation = torch.nn.Sigmoid()

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.backbone.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path, weights_only=False)
            new_dict = {k: v for k, v in modelCheckpoint.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(modelCheckpoint),
                                                                                       len(new_dict)))
            self.backbone.load_state_dict(model_dict)

            not_loaded_keys = [k for k in modelCheckpoint.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

    def forward(self, x):
        e0, e1= self.backbone(x)
        e1_p = self.pwc2(e1)
        e2 = self.encoder2(e1_p) + e1_p
        e2_p = self.pwc3(e2)
        e3 = self.encoder3(e2_p) + e2_p

        b_d2 = self.HFA2(self.feature_fusion_2(e3, e2))
        d2 = self.decoder2(b_d2) #+ b_d2

        b_d1 = self.HFA1(self.feature_fusion_1(d2, e1))
        d1 = self.decoder1(b_d1) #+ b_d1

        b_d0 = self.HFA0(self.feature_fusion_0(d1, e0))
        d0 = self.decoder0(b_d0) #+ b_d0

        d0_4x = self.final_expanding(d0)
        return self.activation(self.final_conv(d0_4x))
        #return 0

class BCHW2BHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


class BHWC2BCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()

class Rolling_BHWC(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -3))


def calculating_params_flops(model, channel, size):
    from thop import profile
    input = torch.randn(1, channel, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print("FLOPs: %.3f G" % (flops / 1e9))
    print("Params: %.3f M" % (params / 1e6))

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3f M" % (total / 1e6))


def calculate_inference_memory_fps(model, input_channels, size, warmup=10, repeat=100):
    import time
    model.eval()
    input = torch.randn(1, input_channels, size, size).cuda()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input)

    peak_memory = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_memory / (1024 ** 2)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(repeat):
            _ = model(input)
            torch.cuda.synchronize()  # 每次迭代后同步

    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_ms = (total_time / repeat) * 1000
    fps = repeat / total_time

    print("\n====== Results ======")
    print(f"Peak GPU Memory: {peak_memory_mb:.2f} MB")
    print(f"Avg. inference time: {avg_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f} (frame/s)")

    return peak_memory_mb, avg_time_ms, fps


def try_backpropagation(model, input_channels, input_size, device):
    import torch.optim as optim
    print("\n" + "=" * 50)
    print("Starting Backpropagation Test")
    print("=" * 50)

    batch_size = 4
    x = torch.randn(batch_size, input_channels, input_size, input_size,
                    requires_grad=True).to(device)

    target = torch.randn_like(x)

    print("\nRunning forward pass...")
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == x.shape, \
        f"Output shape {output.shape} doesn't match input shape {x.shape}"

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(f"Initial loss: {loss.item():.4f}")

    print("\nRunning backward pass...")
    model.zero_grad()
    loss.backward()

    has_gradients = False
    print("\nChecking gradients:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_mean = param.grad.abs().mean().item()
            print(f"{name}: gradient exists (mean abs: {grad_mean:.6f})")
        else:
            print(f"{name}: NO GRADIENT")

    if has_gradients:
        print("\nBackpropagation successful! Gradients are flowing.")
    else:
        print("\nBackpropagation failed! No gradients detected.")

    print("\nUpdating parameters with dummy optimizer...")
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer.step()

    new_output = model(x)
    new_loss = criterion(new_output, target)
    print(f"New loss: {new_loss.item():.4f}")

    if not torch.allclose(output, new_output, atol=1e-5):
        print("Model output changed after parameter update.")
    else:
        print("Model output did not change after update. Check if gradients are zero.")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    input_channels = 3
    input_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SS_UNet(num_classes=1, input_size=input_size,).to(device)
    model.load_from()
    calculating_params_flops(model, input_channels, input_size)
    calculate_inference_memory_fps(model, input_channels, input_size, warmup=50, repeat=50)

