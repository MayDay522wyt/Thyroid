# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
import numpy as np
from typing import Iterable, Optional

import torch
from sklearn.metrics import roc_auc_score

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
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
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def auc(target, output):
    # 将output中每个样本的两个类别的预测概率分别提取出来
    positive_probs = output[:, 1]  # 正类预测概率
    target = target.float()  # 将目标标签转换为浮点型
    # 使用sklearn的roc_auc_score计算AUC
    auc_score = roc_auc_score(target.cpu().numpy(), positive_probs.cpu().numpy())
    return auc_score


def recall(target, output):
    """
    recall: true positive rate。真阳性率，或灵敏度sensitivity。
    衡量模型正确预测为正类样本的比例
    召回率越高，表示模型能更好的捕捉到正类样本
    """
    _, predicted = torch.max(output, 1)  # 获取预测的类别
    positive_mask = (target == 1)  # 正类的掩码
    true_positives = torch.sum(torch.logical_and(predicted == 1, positive_mask))  # 正确预测的正类数量
    positives = torch.sum(positive_mask)  # 实际正类的数量
    recall_score = true_positives.float() / positives.float()  # 计算召回率
    return recall_score.item()


def f1(target, output):
    """
    召回率和精确率的均值
    """
    _, predicted = torch.max(output, 1)  # 获取预测的类别
    positive_mask = (target == 1)  # 正类的掩码
    true_positives = torch.sum(torch.logical_and(predicted == 1, positive_mask))  # 正确预测的正类数量
    predicted_positives = torch.sum(predicted == 1)  # 预测为正类的数量
    actual_positives = torch.sum(positive_mask)  # 实际正类的数量
    precision = true_positives.float() / predicted_positives.float()  # 计算精确率
    recall = true_positives.float() / actual_positives.float()  # 计算召回率
    f1_score = 2 * (precision * recall) / (precision + recall)  # 计算F1分数
    return f1_score.item()

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    targets = []
    outputs = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images) # output [batch_size, 2], target [batch_size]
            loss = criterion(output, target)

        targets.append(target)
        outputs.append(output)
        acc1 = accuracy(output, target)
        # auc_score = auc(target, output)
        # recall_score = recall(target, output)
        # f1_score = f1(target, output)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)
        
    metric_logger.synchronize_between_processes()
    
    targets = torch.cat(targets) #torch.Size([1034])
    outputs = torch.cat(outputs) 

    auc_score = auc(targets, outputs)
    recall_score = recall(targets, outputs)
    f1_score = f1(targets, outputs)


    metric_logger.meters['auc_score'].update(auc_score)
    metric_logger.meters['recall_score'].update(recall_score)
    metric_logger.meters['f1_score'].update(f1_score)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auc {auc_score.global_avg:.3f} recall {recall_score.global_avg:.3f} f1 {f1_score.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc_score=metric_logger.auc_score, recall_score=metric_logger.recall_score, f1_score=metric_logger.f1_score))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}