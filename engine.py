import numpy as np
import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from skimage import measure
from medpy.metric.binary import hd95
import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def train_one_epoch(train_dataloader, model, criterion, optimizer, scheduler, grad_scaler, epoch, step, Mylogger, config):
    model.train()
    loss_list = []

    # 创建 tqdm进度条
    with tqdm.tqdm(total=len(train_dataloader), desc=f'Training[{epoch}/{config.total_epochs}]',
                   unit='Batch') as pbar:
        for iter, data in enumerate(train_dataloader):
            step += iter
            images, targets = data

            if config.device == 'cuda' and train_dataloader.pin_memory == 'True':
                images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
            else:
                images, targets = images.to(config.device).float(), targets.to(config.device).float()

            if config.automatic_mixed_precision:
                with autocast(device_type=config.device, dtype=torch.bfloat16):
                    out = model(images)
                    loss = criterion(out, targets)
            else:
                out = model(images)
                loss = criterion(out, targets)


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optimizer.step()
            pbar.update(1)

            loss_list.append(loss.item())
            now_lr = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.set_postfix(**{'Loss (batch)': loss.item(), 'LR': now_lr})
            if iter % config.print_interval == 0:
                log_info = f'Training[{epoch}/{config.total_epochs}]: Iter:{iter}; Loss: {loss.item():.4f}; LR: {now_lr}'
                Mylogger.logger.info(log_info)
    if config.scheduler != 'AdaptiveLinearAnnealingSoftRestarts':
        scheduler.step()
    return step


def evaluate_one_epoch(evaluate_dataloader, model, criterion, scheduler, epoch, Mylogger, config, output_directory, validation, test_data_name=None):
    model.eval()
    predictions = []
    targets = []


    loss_list = []
    dice_list = []
    miou_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    hd95_list = []

    split_save_directory = output_directory / 'splits'

    with torch.no_grad():
        if validation:
            tqdm_description = f'Validating[{epoch}/{config.total_epochs}]'
        else:
            tqdm_description = f'Testing {test_data_name}'
        with tqdm.tqdm(total=len(evaluate_dataloader), desc=tqdm_description,
                       unit='Batch') as pbar:
            for i, data in enumerate(evaluate_dataloader):
                img, msk = data

                if config.device == 'cuda' and evaluate_dataloader.pin_memory == 'True':
                    img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
                else:
                    img, msk = img.to(config.device).float(), msk.to(config.device).float()

                if config.automatic_mixed_precision:
                    with autocast(device_type=config.device, dtype=torch.bfloat16, enabled=False):
                        output = model(img)
                        loss = criterion(output, msk)
                else:
                    output = model(img)
                    loss = criterion(output, msk)

                loss_list.append(loss.item())
                pbar.update(1)
                pbar.set_postfix(**{'Loss (iter)': f'{loss.item():.4f}'})

                np_output = output.squeeze(0).cpu().detach().numpy()
                np_output = np_array_binarization(np_output, config.evaluate_threshold)
                np_msk = msk.squeeze(0).cpu().detach().numpy()
                np_msk = np_array_binarization(np_msk, config.evaluate_threshold)

                predictions.append(np_output)
                targets.append(np_msk)

                single_hd95 = calculate_hd95(np_prediction=np_output, np_target=np_msk, input_size=config.input_size)
                single_accuracy, single_sensitivity, single_specificity, single_Dice, single_miou, _ = calculate_metrics(np_output, np_msk)

                dice_list.append(single_Dice)
                miou_list.append(single_miou)
                accuracy_list.append(single_accuracy)
                sensitivity_list.append(single_sensitivity)
                specificity_list.append(single_specificity)
                hd95_list.append(single_hd95)

                if not validation:
                    save_imgs(img=img,
                              msk=msk,
                              msk_pred=np_output,
                              i=i,
                              dice=single_Dice,
                              hd95=single_hd95,
                              save_directory=output_directory,
                              split_save_directory=split_save_directory,
                              datasets=config.execute_dataset,
                              threshold=config.evaluate_threshold)

    confusion = calculate_metrics(predictions, targets, confusion_only=True)
    TN, FP, FN, TP = confusion.flatten()

    dice = np.mean(dice_list)
    mioU = np.mean(miou_list)
    accuracy = np.mean(accuracy_list)
    specificity = np.mean(specificity_list)
    sensitivity = np.mean(sensitivity_list)
    hd95 = np.mean(hd95_list)
    loss_A = np.mean(loss_list)

    if validation:
        log_info = f'Validation[{epoch}/{config.total_epochs}]: Loss: {loss_A:.4f}, mIOU: {mioU:.4f}, ' \
                   f'Dice: {dice:.4f}, Acc.: {accuracy:.4f}, ' \
                   f'Spec.: {specificity:.4f}, Recall: {sensitivity:.4f}, HD95: {hd95:.2f}, ' \
                   f'TN={TN:,}, FP={FP:,}, FN={FN:,}, TP={TP:,}.'
    else:
        log_info = f'Testing {test_data_name}: Loss: {loss_A:.4f}, mIOU: {mioU:.4f}, ' \
                   f'Dice: {dice:.4f}, Acc.: {accuracy:.4f}, ' \
                   f'Spec.: {specificity:.4f}, Recall: {sensitivity:.4f}, HD95: {hd95:.2f}, ' \
                   f'TN={TN:,}, FP={FP:,}, FN={FN:,}, TP={TP:,}.'
    print(log_info)
    if Mylogger is not None:
        Mylogger.logger.info(log_info)

    return loss_A, mioU, dice


def save_imgs(img, msk, msk_pred, i, dice, hd95, save_directory, split_save_directory, datasets, threshold=0.5):
    def save_single_image(data: np.ndarray, path, is_gray: bool):
        plt.figure(figsize=(data.shape[1] / 100, data.shape[0] / 100), dpi=100)
        plt.imshow(data, cmap='gray' if is_gray else None)
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi='figure')
        plt.close()

    def save_comparison(img, msk, msk_pred, path, dice, hd95):
        plt.figure(figsize=(7, 11))
        plt.suptitle(f'Dice: {dice:.4f}, HD95: {hd95:.4f}', fontsize=20)

        plt.subplot(3, 1, 1)
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(3, 1, 2)
        plt.imshow(msk, cmap='gray')
        plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.imshow(msk_pred, cmap='gray')
        plt.axis('off')

        plt.savefig(path, bbox_inches='tight', pad_inches=0.3)
        plt.close()

    def save_contour_overlay(base_img, contour_img, path):
        plt.figure(figsize=(base_img.shape[1] / 100, base_img.shape[0] / 100), dpi=100)

        plt.imshow(base_img, cmap='gray')
        contours = measure.find_contours(contour_img, 0.5)

        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color='red')

        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi='figure')
        plt.close()

    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    msk = msk.cpu()
    img = img / 255. if img.max() > 1.1 else img

    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    if msk.ndim == 3 and msk.shape[0] == 1:
        msk = msk[0]
    if msk_pred.ndim == 3 and msk_pred.shape[0] == 1:
        msk_pred = msk_pred[0]

    split_save_directory_i = split_save_directory / f'{i}.png'
    split_save_directory_i.mkdir(parents=True, exist_ok=True)

    save_single_image(img, split_save_directory_i / 'img.png', is_gray=False)
    save_single_image(msk, split_save_directory_i / 'mask.png', is_gray=True)
    save_single_image(msk_pred, split_save_directory_i / 'pred.png', is_gray=True)

    save_contour_overlay(msk_pred, msk, split_save_directory_i / 'pred_with_mask_contour.png')
    save_comparison(img, msk, msk_pred, save_directory / f"{i}.png", dice, hd95)


def calculate_metrics(np_prediction, np_target, confusion_only=False):
    y_pre = np.array(np_prediction).reshape(-1)
    y_true = np.array(np_target).reshape(-1)
    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion.ravel()
    if not confusion_only:
        accuracy = float(TN + TP) / float(TN + FP + FN + TP) if float(TN + FP + FN + TP) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        Dice = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        return accuracy, sensitivity, specificity, Dice, miou, confusion
    else:
        return confusion


def calculate_hd95(np_prediction, np_target, input_size):
    np_prediction = np_prediction.squeeze()
    np_target = np_target.squeeze()

    if np.all(np_target == 0) or np.all(np_prediction == 0):
        single_hd95 = math.sqrt(2 * (input_size ** 2))
    else:
        single_hd95 = hd95(np_prediction, np_target)
    return single_hd95


def np_array_binarization(np_array, threshold):
    return np.where(np_array >= threshold, 1, 0)

