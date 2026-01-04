from configs.config_universal import *
from project_utils import *
from torch.utils.data import DataLoader
from engine import evaluate_one_epoch
from configs.config_model import *

from multi_dataset import *

from models.SS_Net.DynamicTilesMamba import SS_UNet

def test(universal_config, model_config):
    for directory in [universal_config.log_directory, universal_config.output_directory,
                      universal_config.best_models_directory]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    test_dataset = MultiDataset(universal_config=universal_config, Mylogger=None, test=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=universal_config.num_workers,
                                 drop_last=True)

    model_config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'SS_Net':
        model = SS_UNet(**model_config_dict)
    # ↑—————————————————————————————————————————————————————↑
    else:
        raise Exception('model in not right!')
    model.to(device=universal_config.device)

    criterion = universal_config.criterion
    optimizer = get_optimizer(universal_config, model)
    scheduler = get_scheduler(universal_config, optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=universal_config.automatic_mixed_precision)

    if universal_config.pretrain_model_path is not None:
        pretrain_model = torch.load(universal_config.pretrain_model_path, map_location=torch.device(universal_config.device), weights_only=False)
        model.load_state_dict(pretrain_model['model_state_dict'], strict=False)
    loss = evaluate_one_epoch(test_dataloader,
                              model,
                              criterion,
                              scheduler,
                              0,
                              None,
                              universal_config,
                              universal_config.output_directory,
                              validation=False,
                              test_data_name=None
                              )

if __name__ == '__main__':
    universal_config = UniversalConfig(execute_model_index=0, execute_pretrain_model_index=1, execute_dataset_index=0, criterion='BceDiceLoss', optimizer=None, scheduler=None, num_workers=None)
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'SS_Net':
        model_config = SS_NetConfig(universal_config)
    # ↑—————————————————————————————————————————————————————↑
    else:
        raise Exception('model in not right!')
    test(universal_config, model_config)
