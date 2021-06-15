# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:20   xin      1.0         None
'''

from utils import AvgerageMeter, make_optimizer, make_lr_scheduler, calculate_score, mixup_data, rand_bbox, m_evaluate
from common.sync_bn import convert_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


import logging
import os
from tensorboardX import SummaryWriter
from torch import nn
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from prettytable import PrettyTable
from torch.cuda.amp import autocast as autocast
from swa import AveragedModel


class BaseTrainer(object):
    """

    """
    def __init__(self, cfg, model, train_dl, val_dl,
                                  loss_func, num_gpus, device):

        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_func = loss_func

        self.loss_avg = AvgerageMeter()
        self.acc_avg = AvgerageMeter()
        self.f1_avg = AvgerageMeter()

        self.val_loss_avg = AvgerageMeter()
        self.val_acc_avg = AvgerageMeter()
        self.device = device

        self.train_epoch = 1

        if cfg.SOLVER.USE_WARMUP and (self.cfg.SOLVER.LR_SCHEDULER == 'cosine_annealing'
                                      or self.cfg.SOLVER.LR_SCHEDULER == 'poly'):
            self.optim = make_optimizer(self.model, opt=self.cfg.SOLVER.OPTIMIZER_NAME, lr=0.1*cfg.SOLVER.BASE_LR,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)
         
        else:
            self.optim = make_optimizer(self.model, opt=self.cfg.SOLVER.OPTIMIZER_NAME, lr=cfg.SOLVER.BASE_LR,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)

        if cfg.SOLVER.RESUME:
            print("Resume from checkpoint...")
            param_dict = torch.load(cfg.SOLVER.RESUME_CHECKPOINT, map_location=lambda storage, loc: storage)
            if 'state_dict' in param_dict.keys():
                param_dict = param_dict['state_dict']

            start_with_module = False
            for k in param_dict.keys():
                if k.startswith('module.'):
                    start_with_module = True
                    break
            if start_with_module:
                param_dict = {k[7:] : v for k, v in param_dict.items() }
    
            print('ignore_param:')
            print([k for k, v in param_dict.items() if k not in self.model.state_dict() or
                   self.model.state_dict()[k].size() != v.size()])
            print('unload_param:')
            print([k for k, v in self.model.state_dict().items() if k not in param_dict.keys() or
                   param_dict[k].size() != v.size()] )

            param_dict = {k: v for k, v in param_dict.items() if k in self.model.state_dict() and
                          self.model.state_dict()[k].size() == v.size()}
            for i in param_dict:
                self.model.state_dict()[i].copy_(param_dict[i])

        self.batch_cnt = 0

        self.logger = logging.getLogger('baseline.train')
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR

        self.epochs = cfg.SOLVER.MAX_EPOCHS

        if cfg.SOLVER.TENSORBOARD.USE:
            summary_dir = os.path.join(cfg.OUTPUT_DIR, 'summaries/')
            os.makedirs(summary_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=summary_dir)
        self.current_iteration = 0

        self.logger.info(self.model)

        if cfg.SOLVER.SWA:
            # swa https://arxiv.org/abs/2012.12645
            self.scheduler = CosineAnnealingWarmRestarts(self.optim, T_0=len(self.train_dl), eta_min=cfg.SOLVER.MIN_LR)
        else:
            self.scheduler = make_lr_scheduler(self.optim, self.cfg.SOLVER.LR_SCHEDULER, self.cfg.SOLVER.USE_WARMUP,
                                               self.epochs, cfg.SOLVER.T_MAX, cfg.SOLVER.STEPS, cfg.SOLVER.MILESTONES,
                                               cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_EPOCH, cfg.SOLVER.MIN_LR)

        if num_gpus > 1:

            self.logger.info(self.optim)
            self.model = nn.DataParallel(self.model)
            if cfg.SOLVER.SYNCBN:
                self.model = convert_model(self.model)
                self.logger.info('More than one gpu used, convert model to use SyncBN.')
                self.logger.info('Using pytorch SyncBN implementation')
                self.logger.info(self.model)
            self.model = self.model.to(device)
            if cfg.SOLVER.SWA:
                self.swa_model = AveragedModel(self.model)
            self.logger.info('Trainer Built')

            return

        else:
            self.model = self.model.to(device)
            if cfg.SOLVER.SWA:
                self.swa_model = AveragedModel(self.model, device=self.device)
            self.logger.info('Trainer Built')

            return

    def handle_new_batch(self):
        lr = self.scheduler.get_lr()[0]

        if self.cfg.SOLVER.SWA:
            # iter scheduler
            self.scheduler.step()
        if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
            if self.summary_writer:
                self.summary_writer.add_scalar('Train/lr', lr, self.current_iteration)
                self.summary_writer.add_scalar('Train/loss', self.loss_avg.avg, self.current_iteration)
                self.summary_writer.add_scalar('Train/acc', self.acc_avg.avg, self.current_iteration)
                self.summary_writer.add_scalar('Train/f1', self.f1_avg.avg, self.current_iteration)

        self.batch_cnt += 1
        self.current_iteration += 1
        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
        
            self.logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f},'
                                'acc: {:.3f}, f1: {:.3f}, Base Lr: {:.2e}'
                                .format(self.train_epoch, self.batch_cnt,
                                        len(self.train_dl), self.loss_avg.avg,
                                        self.acc_avg.avg, self.f1_avg.avg, lr))

    def handle_new_epoch(self):

        self.batch_cnt = 1

        self.logger.info('Epoch {} done'.format(self.train_epoch))
        self.logger.info('-' * 20)

        if self.cfg.SOLVER.SWA:
            self.swa_model.update_parameters(self.model)
            
            self.swa_save()
            self.evaluate(self.model)
            self.evaluate(self.swa_model)
            
        else:
            self.scheduler.step()

            checkpoint = {
                'epoch': self.train_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
            }
            torch.save(checkpoint, osp.join(self.output_dir, self.cfg.MODEL.NAME + '_epoch_last.pth'))
        
            if self.train_epoch > self.cfg.SOLVER.START_SAVE_EPOCH and self.train_epoch % self.checkpoint_period == 0:
                self.save()
            if (self.train_epoch > self.cfg.SOLVER.START_EVAL_EPOCH and self.train_epoch % self.eval_period == 0) or \
                    self.train_epoch % 20 == 0 or self.train_epoch == 1:
                self.evaluate(self.model)

        self.acc_avg.reset()
        self.f1_avg.reset()
        self.loss_avg.reset()
        self.val_loss_avg.reset()

        self.train_epoch += 1

    def step(self, batch, scaler=None):
        
        self.model.train()
        self.optim.zero_grad()
        data, target = batch

        data, target = data.to(self.device), target.to(self.device)

        if self.cfg.INPUT.USE_CUT_MIX:
            # cut mix
            r = np.random.rand(1)
            if r < 0.5:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(data.size()[0]).to(self.device)

                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                target[:, bbx1:bbx2, bby1:bby2] = target[rand_index, bbx1:bbx2, bby1:bby2]

                data = torch.autograd.Variable(data, requires_grad=True)
                target = torch.autograd.Variable(target)

        if self.cfg.SOLVER.FP16:
            # support fp16
            with autocast():
                if self.cfg.INPUT.USE_MIX_UP:
                    data, target_a, target_b, lam = mixup_data(data, target, 0.4, True)

                outputs = self.model(data)

                if self.cfg.INPUT.USE_MIX_UP:
                    # mix up
                    loss1 = self.loss_func(outputs, target_a)
                    loss2 = self.loss_func(outputs, target_b)
                    loss = lam * loss1 + (1 - lam) * loss2
                else:
                    loss = self.loss_func(outputs, target)

                if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
                    if self.summary_writer:
                        self.summary_writer.add_scalar('Train/loss', loss, self.current_iteration)
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()
        else:
            if self.cfg.INPUT.USE_MIX_UP:
                data, target_a, target_b, lam = mixup_data(data, target, 0.4, True)

            outputs = self.model(data)

            if self.cfg.INPUT.USE_MIX_UP:
                loss1 = self.loss_func(outputs, target_a)
                loss2 = self.loss_func(outputs, target_b)
                loss = lam * loss1 + (1 - lam) * loss2
            else:
                loss = self.loss_func(outputs, target)

            if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
                if self.summary_writer:
                    self.summary_writer.add_scalar('Train/loss', loss, self.current_iteration)
            loss.backward()
            self.optim.step()

        if self.cfg.SOLVER.MULTI_SCALE:
            outputs = outputs[0]

        if isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs = outputs[0]

        _, preds = torch.max(outputs, 1)

        target = target.data.cpu().numpy().squeeze().astype(np.uint8)
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        if self.cfg.SOLVER.TRAIN_LOG:
            f1, acc = calculate_score(self.cfg, preds, target)
            self.loss_avg.update(loss.cpu().item())
            self.acc_avg.update(acc)
            self.f1_avg.update(f1)

        return self.loss_avg.avg, self.acc_avg.avg, self.f1_avg.avg

    def evaluate(self, eval_model):

        eval_model.eval()

        conf_mat = np.zeros((self.cfg.MODEL.N_CLASS, self.cfg.MODEL.N_CLASS)).astype(np.int64)
        with torch.no_grad():

            for batch in tqdm(self.val_dl, total=len(self.val_dl),
                              leave=False):

                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = eval_model(data)
                    
                loss = self.loss_func(outputs, target, val=True)

                target = target.data.cpu()
                if self.cfg.SOLVER.MULTI_SCALE:
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0]

                if isinstance(outputs, tuple) or isinstance(outputs, list):
                    outputs = outputs[0]

                outputs = outputs.data.cpu()

                self.val_loss_avg.update(loss.cpu().item())

                _, preds = torch.max(outputs, 1)

                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                target = target.data.cpu().numpy().squeeze().astype(np.uint8)
                preds = preds[target!=255]
                target = target[target!=255]
                conf_mat = conf_mat + calculate_score(self.cfg, preds, target, 'val')

        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(conf_mat)
    
        table = PrettyTable(["order", "name", "acc", "IoU"])
        for i in range(self.cfg.MODEL.N_CLASS):
            table.add_row([i, self.cfg.DATASETS.CLASS_NAMES[i], val_acc_per_class[i], val_IoU[i]])

        self.logger.info('Validation Result:')
        self.logger.info(table)
        self.logger.info('VAL_LOSS: %s, VAL_ACC: %s VAL_MEAN_IOU: %s VAL_KAPPA: %s \n' %
                         (self.val_loss_avg.avg, val_acc, val_mean_IoU, val_kappa))

        self.logger.info('-' * 20)

        if self.summary_writer:

            self.summary_writer.add_scalar('Valid/loss', self.val_loss_avg.avg, self.train_epoch)
            self.summary_writer.add_scalar('Valid/acc', np.mean(val_acc), self.train_epoch)

    def swa_save(self, update_bn=False):
        if not update_bn:
            torch.save(self.swa_model.state_dict(), osp.join(self.output_dir, self.cfg.MODEL.NAME + '_swa_epoch' +
                                                             str(self.train_epoch) + '.pth'))
        else:
            torch.save(self.swa_model.state_dict(), osp.join(self.output_dir,
                                                     'swa_final.pth'))

    def save(self):

        torch.save(self.model.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch' + str(self.train_epoch) + '.pth'))
