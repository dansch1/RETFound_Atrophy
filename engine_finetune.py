import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, multilabel_confusion_matrix,
    mean_absolute_error
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched
from loss import ICLoss, ILoss


def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        mixup_fn: Optional[Mixup] = None,
        log_writer=None,
        args=None
):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if mixup_fn:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        output_ = nn.Softmax(dim=1)(output)
        output_label = output_.argmax(dim=1)
        output_onehot = F.one_hot(output_label.to(torch.int64), num_classes=num_class)

        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(output_onehot.detach().cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        pred_labels.extend(output_label.detach().cpu().numpy())
        pred_softmax.extend(output_.detach().cpu().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')

    score = (f1 + roc_auc + kappa) / 3
    if log_writer:
        for metric_name, value in zip(
                ['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa',
                 'score'],
                [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)

    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n'
          f' Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
          f' Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}')

    metric_logger.synchronize_between_processes()

    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall',
                         'average_precision', 'kappa'])
        wf.writerow(
            [metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall,
             average_precision, kappa])

    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score


@torch.no_grad()
def evaluate_I(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = ILoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    model.eval()
    true_intervals, pred_intervals = [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        metric_logger.update(loss=loss.item())
        true_intervals.extend(output.reshape(-1, 2).cpu().detach().numpy())
        pred_intervals.extend(target.reshape(-1, 2).cpu().detach().numpy())

    iou = iou_interval(true_intervals, pred_intervals)
    print(f'Interval IoU: {iou}')

    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, iou


def misc_measures(confusion_matrix):
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []

    for i in range(1, confusion_matrix.shape[0]):
        cm1 = confusion_matrix[i]
        acc.append(1. * (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1))
        sensitivity_ = 1. * cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        sensitivity.append(sensitivity_)
        specificity_ = 1. * cm1[0, 0] / (cm1[0, 1] + cm1[0, 0])
        specificity.append(specificity_)
        precision_ = 1. * cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_ * specificity_))
        F1_score_2.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_))
        mcc = (cm1[0, 0] * cm1[1, 1] - cm1[0, 1] * cm1[1, 0]) / np.sqrt(
            (cm1[0, 0] + cm1[0, 1]) * (cm1[0, 0] + cm1[1, 0]) * (cm1[1, 1] + cm1[1, 0]) * (cm1[1, 1] + cm1[0, 1]))
        mcc_.append(mcc)

    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()

    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_


@torch.no_grad()
def evaluate_IC(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    """Evaluate the model."""
    criterion = ICLoss(num_class)
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax, = [], [], [], [], []
    true_intervals, pred_intervals = [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        images, target = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)
        interval_target, class_target = target[..., :2].to(torch.int64), target[..., 2].flatten().to(torch.int64)
        target_onehot = F.one_hot(class_target, num_classes=num_class)

        with torch.cuda.amp.autocast():
            interval_pred, class_pred = model(images)
            class_pred = class_pred.reshape(-1, num_class)
            loss = criterion((interval_pred, class_pred), target)

        class_pred_ = nn.Softmax(dim=1)(class_pred)
        class_pred_label = class_pred_.argmax(dim=1)
        class_pred_onehot = F.one_hot(class_pred_label.to(torch.int64), num_classes=num_class)

        metric_logger.update(loss=loss.item())
        true_onehot.extend(target_onehot.cpu().numpy())
        pred_onehot.extend(class_pred_onehot.detach().cpu().numpy())
        true_labels.extend(class_target.cpu().numpy())
        pred_labels.extend(class_pred_label.detach().cpu().numpy())
        pred_softmax.extend(class_pred_.detach().cpu().numpy())

        true_intervals.extend(interval_target.reshape(-1, 2).cpu().detach().numpy())
        pred_intervals.extend(interval_pred.reshape(-1, 2).cpu().detach().numpy())

    accuracy = accuracy_score(true_labels, pred_labels)
    hamming = hamming_loss(true_onehot, pred_onehot)
    jaccard = jaccard_score(true_onehot, pred_onehot, average='macro')
    average_precision = average_precision_score(true_onehot, pred_softmax, average='macro')
    kappa = cohen_kappa_score(true_labels, pred_labels)
    f1 = f1_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    roc_auc = roc_auc_score(true_onehot, pred_softmax, multi_class='ovr', average='macro')
    precision = precision_score(true_onehot, pred_onehot, zero_division=0, average='macro')
    recall = recall_score(true_onehot, pred_onehot, zero_division=0, average='macro')

    iou = iou_interval(true_intervals, pred_intervals)
    print(f'Interval IoU: {iou}')

    class_score = (f1 + roc_auc + kappa) / 3
    intervall_score = iou
    score = args.class_weight * class_score + (1 - args.class_weight) * intervall_score
    if log_writer:
        for metric_name, value in zip(
                ['accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall', 'average_precision', 'kappa',
                 'score'],
                [accuracy, f1, roc_auc, hamming, jaccard, precision, recall, average_precision, kappa, score]):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)

    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, Hamming Loss: {hamming:.4f},\n'
          f' Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
          f' Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}')

    metric_logger.synchronize_between_processes()

    results_path = os.path.join(args.output_dir, args.task, f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if not file_exists:
            wf.writerow(['val_loss', 'accuracy', 'f1', 'roc_auc', 'hamming', 'jaccard', 'precision', 'recall',
                         'average_precision', 'kappa'])
        wf.writerow(
            [metric_logger.meters["loss"].global_avg, accuracy, f1, roc_auc, hamming, jaccard, precision, recall,
             average_precision, kappa])

    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels, predict_vector=pred_labels)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=True, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, args.task, 'confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, score


def iou_interval(true_intervals, pred_intervals):
    iou_scores = []

    for (x0_true, x1_true), (x0_pred, x1_pred) in zip(true_intervals, pred_intervals):
        if x0_true < 0 and x1_true < 0 and x0_pred < 0 and x1_pred < 0:
            iou_scores.append(1.0)
            continue

        if (x0_true < 0 or x1_true < 0) or (x0_pred < 0 or x1_pred < 0):
            iou_scores.append(0.0)
            continue

        intersection = max(0, min(x1_true, x1_pred) - max(x0_true, x0_pred))
        union = (x1_true - x0_true) + (x1_pred - x0_pred) - intersection
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)

    return np.mean(iou_scores)


def MAE_interval(true_interval_list, prediction_interval_list):
    true_interval_array = np.array(true_interval_list)
    prediction_interval_array = np.array(prediction_interval_list)

    mae_x0 = mean_absolute_error(true_interval_array[:, 0], prediction_interval_array[:, 0])
    mae_x1 = mean_absolute_error(true_interval_array[:, 1], prediction_interval_array[:, 1])

    return (mae_x0 + mae_x1) / 2
