
from configs.config_model import *
from project_utils import *
import torch

from multi_dataset import MultiDataset
from torch.utils.data import DataLoader

from engine import *
import time

from models.SS_Net.DynamicTilesMamba import SS_UNet


def execute(universal_config, model_config):
    print('\nStep1: Creating result directory')
    try:
        for directory in [universal_config.log_directory, universal_config.output_directory,
                          universal_config.best_models_directory]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
        print('\tResult directory created!')
    except:
        print('\tCaught ERROR, result directory not created.')
        exit()

    print('Step2: Creating Logger')
    logger_name = universal_config.execute_model + '_' + universal_config.execute_dataset
    Mylogger_instance = MyLogger(logger_name, universal_config.log_directory, universal_config, model_config)
    Mylogger_instance.creat_info_file()
    Mylogger_instance.log_UniversalConfig_info()
    Mylogger_instance.log_ModelConfig_info()
    print('\tLogger instance created')

    print('Step3: Device Initialization')
    set_seed(universal_config.seed)
    torch.cuda.empty_cache()

    print('Step4: Preparing Dataset')
    Mylogger_instance.logger.info('#----------Dataset Info----------#')

    train_dataset = MultiDataset(universal_config=universal_config, Mylogger=Mylogger_instance, train=True)
    val_dataset = MultiDataset(universal_config=universal_config, Mylogger=Mylogger_instance, validation=True)
    test_dataset = MultiDataset(universal_config=universal_config, Mylogger=Mylogger_instance, test=True)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=universal_config.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=universal_config.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=universal_config.num_workers,
                                drop_last=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=universal_config.num_workers,
                                 drop_last=True)
    print(f'\t Dataloaders of {universal_config.execute_dataset} are ready')

    print('Step5: Preparing Model')
    model_config_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith('__') and not callable(v)}
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'SS_Net':
        model = SS_UNet(**model_config_dict)
    # ↑—————————————————————————————————————————————————————↑
    else:
        raise Exception('model in not right!')

    model.to(device=universal_config.device)
    print(f'\t{universal_config.execute_model} is ready, calculating metrics...')
    calculating_params_flops(model, universal_config.input_size_h, Mylogger_instance)

    print('Step6: Prepareing Criterion, Optimizer, Scheduler and Amp')
    criterion = universal_config.criterion
    optimizer = get_optimizer(universal_config, model)
    scheduler = get_scheduler(universal_config, optimizer)

    # 打印 日志分割信息
    Mylogger_instance.logger.info('#----------Amp Info----------#')
    if universal_config.automatic_mixed_precision:
        log_info = '\tEnabled Automatic Mixed Precision'
    else:
        log_info = '\tDisabled Automatic Mixed Precision'
    # 打印 日志信息
    Mylogger_instance.logger.info(log_info)
    print(log_info)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=universal_config.automatic_mixed_precision)

    print('Step7: Set Pretrain Model')
    Mylogger_instance.logger.info('#----------Set Pretrain Model Info----------#')
    start_epoch = 1
    total_time = 0
    step = 0
    min_loss = 9999
    max_dice = 0
    max_mIOU = 0
    min_loss_epoch = 0
    max_dice_epoch = 0
    max_score_epoch = 0
    max_score_loss = 9999
    max_score_bias = 9999

    # 占位符
    max_dice_mIOU = 0
    max_dice_loss = 9999
    max_score_dice = 0
    max_score_mIOU = 0

    if universal_config.pretrain_model_path is not None:
        pretrain_model = torch.load(universal_config.pretrain_model_path,
                                    map_location=torch.device(universal_config.device), weights_only=False)
        pretrain_model_total_time = pretrain_model['current_total_time']
        current_epoch = pretrain_model['stepped_size']
        start_epoch += current_epoch
        min_loss, min_loss_epoch, loss = pretrain_model['min_loss'], pretrain_model['min_loss_epoch'], pretrain_model[
            'current_loss']
        max_dice, max_mIOU, max_dice_epoch = pretrain_model['max_dice'], pretrain_model['max_mIOU'], pretrain_model[
            'max_dice_epoch']
        max_score_epoch, max_score_loss, max_score_bias = pretrain_model['max_score_epoch'], pretrain_model[
            'max_score_loss'], pretrain_model['max_score_bias']
        model.load_state_dict(pretrain_model['model_state_dict'], strict=False)
        optimizer.load_state_dict(pretrain_model['optimizer_state_dict'])
        scheduler.load_state_dict(pretrain_model['scheduler_state_dict'])

        Mylogger_instance.log_and_print_custom_info(
            f'\tPretrain model loaded from {universal_config.pretrain_model_path}.\n'
            f'\tPretrain model had used time: {pretrain_model_total_time / 60:.2f} minutes\n'
            f'\t\tmin loss epoch:{min_loss_epoch}, min loss:{min_loss:.4f}\n'
            f'\t\tmax dice epoch:{max_dice_epoch}, max dice:{max_dice:.4f}, max mIOU:{max_mIOU:.4f}\n'
            f'\t\tmax score epoch:{max_score_epoch}, max score loss:{max_score_loss:.4f}, max score bias: {max_score_bias:.4f}\n')
    else:
        pretrain_model_total_time = None
        Mylogger_instance.log_and_print_custom_info('No pretrain model loaded', indent=True)

    print('Step8: Training')
    Mylogger_instance.logger.info('#----------Training and Validating Info----------#')
    refresh_ALASR = False
    torch.cuda.reset_peak_memory_stats(universal_config.device)
    for epoch in range(start_epoch, universal_config.total_epochs + 1):
        start_time = time.time()
        torch.cuda.empty_cache()

        step = train_one_epoch(train_dataloader,
                               model,
                               criterion,
                               optimizer,
                               scheduler,
                               grad_scaler,
                               epoch,
                               step,
                               Mylogger_instance,
                               universal_config)

        loss, mIOU, dice = evaluate_one_epoch(val_dataloader,
                                              model,
                                              criterion,
                                              scheduler,
                                              epoch,
                                              Mylogger_instance,
                                              universal_config,
                                              universal_config.output_directory,
                                              validation=True
                                              )

        if loss < min_loss:
            refresh_ALASR = True
            min_loss = loss
            min_loss_epoch = epoch
            max_score_bias = max_score_loss - min_loss
            save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch, max_score_epoch,
                       max_score_loss, max_score_bias, loss, model, optimizer, scheduler, 'min_loss.pth',
                       universal_config)
            if (epoch / universal_config.total_epochs) >= universal_config.result_interval:
                Mylogger_instance.log_and_print_custom_info(
                    f'min_loss.pth Updated: epoch:{min_loss_epoch}, loss:{min_loss:.4f}', indent=True)

        # 若 mIOU与dice 不为空
        if (mIOU is not None) and (dice is not None):
            if dice > max_dice:
                refresh_ALASR = True
                max_dice = dice
                max_dice_epoch = epoch
                max_dice_loss = loss
                max_dice_mIOU = mIOU
                save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch,
                           max_score_epoch, max_score_loss, max_score_bias, loss, model, optimizer, scheduler,
                           'max_dice.pth', universal_config)
                if (epoch / universal_config.total_epochs) >= universal_config.result_interval:
                    # 打印 日志信息
                    Mylogger_instance.log_and_print_custom_info(
                        f'max_dice.pth Updated: epoch:{max_dice_epoch}, dice:{max_dice:.4f}, mIOU:{max_dice_mIOU:.4f}, loss:{max_dice_loss:.4f}', indent=True)

                # 计算 当前的loss偏差
                loss_bias = loss - min_loss
                if loss_bias < max_score_bias and loss_bias != 0:
                    max_score_loss = loss
                    max_score_dice = dice
                    max_score_bias = loss_bias
                    max_score_mIOU = mIOU
                    max_score_epoch = epoch
                    save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch,
                               max_score_epoch, max_score_loss, max_score_bias, loss, model, optimizer, scheduler,
                               'max_score.pth', universal_config)

                    if (epoch / universal_config.total_epochs) >= universal_config.result_interval:
                        # 打印 日志信息
                        Mylogger_instance.log_and_print_custom_info(
                            f'max_score.pth Updated: epoch:{max_score_epoch}, dice:{max_score_dice:.4f}, mIOU:{max_score_mIOU:.4f}, loss:{max_score_loss:.4f}, bias: {max_score_bias:.4f}', indent=True)

        if universal_config.scheduler == 'AdaptiveLinearAnnealingSoftRestarts':
            scheduler.step(loss, refresh_ALASR)
            refresh_ALASR = False

        save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch, max_score_epoch,
                   max_score_loss, max_score_bias, loss, model, optimizer, scheduler, 'latest.pth', universal_config)
        # torch.save(model.state_dict(), str(universal_config.best_models_directory / f'latest.pth'))

        early_stopping_remaining_epoch = max(min_loss_epoch, max_dice_epoch,
                                             max_score_epoch) + universal_config.early_stopping_patience - epoch

        if early_stopping_remaining_epoch <= 0:
            Mylogger_instance.log_and_print_custom_info(f'Early Stopped! Stop at epoch{epoch}.', indent=True)
            break

        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time
        remaining_epochs = universal_config.total_epochs - epoch
        if pretrain_model_total_time is not None:
            total_estimated_remaining_time = (total_time / (epoch - start_epoch + 1)) * remaining_epochs
        else:
            total_estimated_remaining_time = (total_time / epoch) * remaining_epochs
        if pretrain_model_total_time is not None:
            early_stop_estimated_remaining_time = (total_time / (
                    epoch - start_epoch + 1)) * early_stopping_remaining_epoch
        else:
            early_stop_estimated_remaining_time = (total_time / epoch) * early_stopping_remaining_epoch

        if epoch % universal_config.estimate_interval == 0:
            Mylogger_instance.log_and_print_custom_info(
                f'Before early Stop point remain {early_stopping_remaining_epoch} epoch(s)', indent=True)

            if early_stopping_remaining_epoch <= 70:
                Mylogger_instance.log_and_print_custom_info(
                    f'Supplementary Information: Max dice epoch: {max_dice_epoch}, dice:{max_dice:.4f}', indent=True)
            if pretrain_model_total_time is not None:
                print(f'\tPretrain model used time: {pretrain_model_total_time / 60:.2f}minutes. '
                      f'Current training used time: {total_time / 60:.2f} minutes. Estimated '
                      f'total remaining time: {total_estimated_remaining_time / 60:.2f} minutes. Estimated remaining '
                      f'time at early stop point: {early_stop_estimated_remaining_time / 60:.2f} minutes.')
            else:
                print(
                    f'\tCurrent training used time: {total_time / 60:.2f} minutes. Estimated total remaining time: {total_estimated_remaining_time / 60:.2f} minutes. Estimated remaining time at early stop point: {early_stop_estimated_remaining_time / 60:.2f} minutes.')

    max_memory_allocated = torch.cuda.max_memory_allocated(universal_config.device) / 1024 ** 2  # 以MB为单位
    Mylogger_instance.log_and_print_custom_info(f'\nTraining Ends: \n' \
                                                f'\t\t Maximum CUDA memory usage:{max_memory_allocated:.2f} MB \n'
                                                f'\t\t min_loss.pth info : epoch:{min_loss_epoch}, loss:{min_loss:.4f}\n' \
                                                f'\t\t max_dice.pth info : epoch:{max_dice_epoch}, dice:{max_dice:.4f}, mIOU:{max_dice_mIOU:.4f}, loss:{max_dice_loss:.4f}\n' \
                                                f'\t\t max_score.pth info: epoch:{max_score_epoch}, dice:{max_score_dice:.4f}, mIOU:{max_score_mIOU:.4f}, loss:{max_score_loss:.4f}, bias: {max_score_bias:.4f}\n')

    print('Step9: Testing')
    Mylogger_instance.logger.info('#----------Testing Info----------#')
    if max_score_epoch == 0:
        test_model_list = ['min_loss.pth', 'max_dice.pth']
    else:
        test_model_list = ['min_loss.pth', 'max_dice.pth', 'max_score.pth']
    for best_model_file_name in test_model_list:
        best_model_file = universal_config.best_models_directory.glob(best_model_file_name)
        best_model_file = next(best_model_file, None)
        best_model = torch.load(best_model_file, map_location=torch.device(universal_config.device), weights_only=False)
        model.load_state_dict(best_model['model_state_dict'], strict=False)
        output_directory = universal_config.output_directory / best_model_file_name[:-4]
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)

        loss = evaluate_one_epoch(test_dataloader,
                                  model,
                                  criterion,
                                  scheduler,
                                  0,
                                  Mylogger_instance,
                                  universal_config,
                                  output_directory,
                                  validation=False,
                                  test_data_name=best_model_file_name[:-4]
                                  )
        if best_model_file_name == 'min_loss.pth' and min_loss_epoch == max_dice_epoch == max_score_epoch:
            break
        if best_model_file_name == 'max_dice.pth' and max_dice_epoch == max_score_epoch:
            break


def save_model(total_time, epoch, min_loss, max_mIOU, max_dice, min_loss_epoch, max_dice_epoch, max_score_epoch,
               max_score_loss, max_score_bias, loss, model, optimizer, scheduler, model_name, universal_config):
    torch.save(
        {
            'current_total_time': total_time,
            'stepped_size': epoch,
            'min_loss': min_loss,
            'max_mIOU': max_mIOU,
            'max_dice': max_dice,
            'min_loss_epoch': min_loss_epoch,
            'max_dice_epoch': max_dice_epoch,
            'max_score_epoch': max_score_epoch,
            'max_score_loss': max_score_loss,
            'max_score_bias': max_score_bias,
            'current_loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, str(universal_config.best_models_directory / model_name))


def main(execute_model_index, execute_pretrain_model_index, execute_dataset_index, criterion=None, optimizer=None,
         scheduler=None, num_workers=None):
    universal_config = UniversalConfig(execute_model_index, execute_pretrain_model_index, execute_dataset_index,
                                       criterion, optimizer, scheduler, num_workers)
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'SS_Net':
        model_config = SS_NetConfig(universal_config)
    # ↑—————————————————————————————————————————————————————↑
    else:
        raise Exception('model in not right!')
    execute(universal_config, model_config)


if __name__ == '__main__':
    universal_config = UniversalConfig(execute_model_index=0, execute_pretrain_model_index=0, execute_dataset_index=0,
                                       criterion='BceDiceLoss', optimizer=None, scheduler=None, num_workers=None)
    # ↓—————————————————————————————————————————————————————↓
    if universal_config.execute_model == 'SS_Net':
        model_config = SS_NetConfig(universal_config)
    # ↑—————————————————————————————————————————————————————↑
    else:
        raise Exception('model in not right!')
    execute(universal_config, model_config)
