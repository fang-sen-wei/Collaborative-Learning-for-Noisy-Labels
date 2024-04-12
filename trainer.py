import time
import torch
import os
from util import log_display, accuracy, AverageMeter
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer():
    def __init__(self, data_loader, logger, config, name='Trainer', metrics='classfication'):
        self.data_loader = data_loader
        self.logger = logger
        self.name = name
        self.step = 0
        self.config = config
        self.log_frequency = config.log_frequency
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()
        self.report_metrics = self.classfication_metrics if metrics == 'classfication' else self.regression_metrics

    def train(self, epoch, GLOBAL_STEP, model1, criterion1, optimizer1, model2, criterion2, optimizer2, forget_rate):
        model1.train()
        model2.train()
        for images, labels in self.data_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            self.train_batch(images, labels, model1, criterion1, optimizer1, model2, criterion2, optimizer2, forget_rate)
            self.log(epoch, GLOBAL_STEP)
            GLOBAL_STEP += 1
        return GLOBAL_STEP

    def train_batch(self, x, y, model1, criterion1, optimizer1, model2, criterion2, optimizer2, forget_rate):
        start = time.time()

        model1.zero_grad()
        optimizer1.zero_grad()
        pred1 = model1(x)

        model2.zero_grad()
        optimizer2.zero_grad()
        pred2 = model2(x)

        # Calculate loss for sorting
        # The replacement of the sample selection loss function
        # use CE loss function for sorting
        loss1 = F.cross_entropy(pred1, y, reduction='none')
        loss2 = F.cross_entropy(pred2, y, reduction='none')

        # sort
        ind_1_sorted = torch.argsort(loss1.data).cuda()
        loss_1_sorted = loss1[ind_1_sorted]

        ind_2_sorted = torch.argsort(loss2.data).cuda()
        loss_2_sorted = loss2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        # exchange
        # use Robust loss functions to backward
        loss_1_update = criterion1(pred1[ind_2_update], y[ind_2_update])
        loss_2_update = criterion2(pred2[ind_1_update], y[ind_1_update])

        loss_1_update.backward()
        grad_norm_1 = torch.nn.utils.clip_grad_norm_(model1.parameters(), self.config.grad_bound)
        optimizer1.step()

        loss_2_update.backward()
        grad_norm_2 = torch.nn.utils.clip_grad_norm_(model2.parameters(), self.config.grad_bound)
        optimizer2.step()

        self.report_metrics(pred1, y, loss_1_update)
        self.report_metrics(pred2, y, loss_2_update)

        self.logger_payload['lr-1'] = optimizer1.param_groups[0]['lr'],
        self.logger_payload['|gn-1|'] = grad_norm_1

        self.logger_payload['lr-2'] = optimizer2.param_groups[0]['lr'],
        self.logger_payload['|gn-2|'] = grad_norm_2
        end = time.time()
        self.step += 1
        self.time_used = end - start


    def log(self, epoch, GLOBAL_STEP):
        if GLOBAL_STEP % self.log_frequency == 0:
            display = log_display(epoch=epoch,
                                  global_step=GLOBAL_STEP,
                                  time_elapse=self.time_used,
                                  **self.logger_payload)
            self.logger.info(display)

    def classfication_metrics(self, x, y, loss):
        acc, acc5 = accuracy(x, y, topk=(1, 5))
        self.loss_meters.update(loss.item(), y.shape[0])
        self.acc_meters.update(acc.item(), y.shape[0])
        self.acc5_meters.update(acc5.item(), y.shape[0])
        self.logger_payload = {"acc": acc,
                               "acc_avg": self.acc_meters.avg,
                               "loss": loss,
                               "loss_avg": self.loss_meters.avg}

    def regression_metrics(self, x, y, loss):
        diff = abs((x - y).mean().detach().item())
        self.loss_meters.update(loss.item(), y.shape[0])
        self.acc_meters.update(diff, y.shape[0])
        self.logger_payload = {"|diff|": diff,
                               "|diff_avg|": self.acc_meters.avg,
                               "loss": loss,
                               "loss_avg": self.loss_meters.avg}

    def _reset_stats(self):
        self.loss_meters.reset()
        self.acc_meters.reset()
        self.acc5_meters.reset()
