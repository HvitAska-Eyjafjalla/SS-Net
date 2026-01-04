from configs.config_universal import UniversalConfig

class SS_NetConfig:
    def __init__(self, UniversalConfig):
        self.load_ckpt_path = UniversalConfig.project_directory / 'pretrain' / 'SS_Net' / 'pvt_v2_b2.pth'
        self.num_classes = UniversalConfig.num_classes
        self.input_size = UniversalConfig.input_size