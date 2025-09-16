import os
import argparse

import torch
import pytorch_lightning as pl
import torch.nn as nn

import jutil
from Model.loss import Anatomical_Loss, CompoundLoss, MS_SSIM_L1_LOSS
from measure import compute_MSE, compute_SSIM


def window_denormalize(img, window = (-20, 100), tgt_window= (0, 80)):
    win_range = abs(window[0]-window[1])
    img *= win_range
    img += window[0]
    return img.clip(tgt_window[0], tgt_window[1])

class ADE_solver(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Update Args
        if args.__class__ == dict:
            args = argparse.Namespace(**args)
        self.args = args
        self.save_hyperparameters(args)

        # Set Model
        if self.args.model_name == 'KPNFeat':
            from Model.KPN.kpn_feat_model import KPNfeat
            self.model = KPNfeat(in_nc=1, out_nc=1, nf=self.args.nf, nk=self.args.nk, kpn_sz=5, pytorch_init=args.pytorch_init)
        elif self.args.model_name == 'REDCNN':
            from Model.REDCNN.redcnn_model import RED_CNN
            self.model = RED_CNN()
        elif self.args.model_name == 'EDCNN':
            from Model.EDCNN.edcnn_model import EDCNN
            self.model = EDCNN()

        # Set Loss
        self.l2_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()
        self.ssim_l1_criterion = MS_SSIM_L1_LOSS()
        self.compound_criterion = CompoundLoss()
        self.anatomical_criterion = Anatomical_Loss(w_p = args.ana_model_path)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams['lr'])
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                         lr_lambda=lambda epoch: self.hparams['lrdecay'] ** epoch,
                                                         last_epoch=-1,
                                                         verbose=False)

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)
        return [optimizer], [lr_scheduler]

    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch['LDCT'], batch['NDCT']
        pred = self(x)
        if self.args.loss_type == 'MSE':
            l2_loss = self.l2_criterion(pred, y)
            Loss = l2_loss
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_MSE', l2_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.args.loss_type == 'l1':
            l1_loss = self.l1_criterion(pred, y)
            Loss = l1_loss
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1', l1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.args.loss_type == 'l1_weighted':
            l1_loss = self.l1_criterion(pred, y)
            ind_map = torch.logical_and(y>=1024/4096, y<1104/4096)
            l1_weighted_loss = self.l1_criterion(pred[ind_map], y[ind_map]) * 4096 / 80
            Loss = l1_loss + l1_weighted_loss
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1', l1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1(0~80)', l1_weighted_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.args.loss_type == 'l1_l1ana_1':
            l1_loss = self.l1_criterion(pred, y)
            l1ana_loss = self.anatomical_criterion(pred, y, 'l1_l1ana_1')
            Loss = l1_loss * (1-self.args.ana_weight) + l1ana_loss * self.args.ana_weight
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1', l1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1ana_1', l1ana_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.args.loss_type == 'l1_l1ana_2':
            l1_loss = self.l1_criterion(pred, y)
            l1ana_loss = self.anatomical_criterion(pred, y, 'l1_l1ana_2')
            Loss = l1_loss * (1-self.args.ana_weight) + l1ana_loss * self.args.ana_weight
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1', l1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1ana_2', l1ana_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.args.loss_type == 'Compound':
            img_loss, img_loss_weight, feat_loss, feat_loss_weight = self.compound_criterion(pred, y)
            Loss = img_loss_weight * img_loss + feat_loss_weight * feat_loss
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_MSE', img_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_Feat', feat_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        elif self.args.loss_type == 'SSIM_L1':
            Loss, l1, ssim = self.ssim_l1_criterion(pred, y)
            # Loss = img_loss_weight * img_loss + feat_loss_weight * feat_loss
            self.log('train/Loss', Loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_l1', l1, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('train/Loss_ssim', ssim, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return Loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['LDCT'], batch['NDCT']
        pred = self(x)
        confusion = self.anatomical_criterion(pred,y, 'Confusion')
        return {'x':x, 'y':y, 'pred':pred, 'confusion':confusion}

    def validation_epoch_end(self, outputs):
        Anatomical_Loss_List = []
        confusion_labels = {}
        for label in range(1,18):
            confusion_labels[label] = {}
            confusion_labels[label]['TP'] = 0
            confusion_labels[label]['TN'] = 0
            confusion_labels[label]['FP'] = 0
            confusion_labels[label]['FN'] = 0

        MSE1_List = []
        MSE2_List = []
        SSIM1_List = []
        SSIM2_List = []
        cnt=0
        for item in outputs:
            y_1 = window_denormalize(item['y'], window=self.args.window[0], tgt_window=(0, 80))
            pred_1 = window_denormalize(item['pred'], window=self.args.window[0], tgt_window=(0, 80))
            y_2 = window_denormalize(item['y'], window=self.args.window[0], tgt_window=(-1024, 3072))
            pred_2 = window_denormalize(item['pred'], window=self.args.window[0], tgt_window=(-1024, 3072))

            for label in range(1, 18):
                Anatomical_Loss_List.append(item['confusion']['Loss'])
                confusion_labels[label]['TP'] += item['confusion']['TP'][label]
                confusion_labels[label]['TN'] += item['confusion']['TN'][label]
                confusion_labels[label]['FP'] += item['confusion']['FP'][label]
                confusion_labels[label]['FN'] += item['confusion']['FN'][label]

            for b in range(len(item['pred'])):
                MSE1_List.append(compute_MSE(pred_1[b, None].float(), y_1[b, None].float()))
                MSE2_List.append(compute_MSE(pred_2[b, None].float(), y_2[b, None].float()))
                SSIM1_List.append(compute_SSIM(pred_1[b, None].float(), y_1[b, None].float(), data_range=80))
                SSIM2_List.append(compute_SSIM(pred_2[b, None].float(), y_2[b, None].float(), data_range=4096))
                cnt+=1

        aloss = torch.stack(Anatomical_Loss_List).mean()
        self.log('val/Loss_Anatomical' , aloss, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.logger.experiment.add_scalar('val/Loss_Anatomical', aloss, global_step=self.current_epoch)

        dice_labels = {}
        for label in range(1,18):
            TP = confusion_labels[label]['TP']
            TN = confusion_labels[label]['TN']
            FP = confusion_labels[label]['FP']
            FN = confusion_labels[label]['FN']
            dice_labels[label] = 2*TP / (2*TP + FP + FN + 1e-4)
            self.log('val_Anatomical/dice_%d'%(label), dice_labels[label], on_step=False, on_epoch=True, prog_bar=False, logger=False)
            self.logger.experiment.add_scalar('val_Anatomical/dice_%d'%(label), dice_labels[label], global_step=self.current_epoch)

        # [0~80]
        MSE1 = torch.stack(MSE1_List,dim=0)
        SSIM1 = torch.stack(SSIM1_List, dim=0)

        Avg_MSE1 = MSE1.mean()
        Avg_PSNR1 = (10 * torch.log10((80 ** 2) / MSE1[MSE1>0])).mean()
        Avg_SSIM1 = SSIM1.mean()
        Total_PSNR1 = 10 * torch.log10((80 ** 2) / Avg_MSE1)

        self.log('val/MSE', Avg_MSE1, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('val/MSE', Avg_MSE1, global_step=self.current_epoch)
        self.log('val/Total_PSNR', Total_PSNR1, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.logger.experiment.add_scalar('val/Total_PSNR', Total_PSNR1, global_step=self.current_epoch)
        self.log('val/Avg_PSNR', Avg_PSNR1, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.logger.experiment.add_scalar('val/Avg_PSNR', Avg_PSNR1, global_step=self.current_epoch)
        self.log('val/Avg_SSIM', Avg_SSIM1, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('val/Avg_SSIM', Avg_SSIM1, global_step=self.current_epoch)

        # Full Range
        MSE2 = torch.stack(MSE2_List, dim=0)
        SSIM2 = torch.stack(SSIM2_List, dim=0)

        Avg_MSE2 = MSE2.mean()
        Avg_PSNR2 = (10 * torch.log10((4096 ** 2) / MSE2[MSE2>0])).mean()
        Avg_SSIM2 = SSIM2.mean()
        Total_PSNR2 = 10 * torch.log10((4096 ** 2) / Avg_MSE2)

        self.log('val/(FR)MSE', Avg_MSE2, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('val/(FR)MSE', Avg_MSE2, global_step=self.current_epoch)
        self.log('val/(FR)Total_PSNR', Total_PSNR2, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.logger.experiment.add_scalar('val/(FR)Total_PSNR', Total_PSNR2, global_step=self.current_epoch)
        self.log('val/(FR)Avg_PSNR', Avg_PSNR2, on_step=False, on_epoch=True, prog_bar=False, logger=False)
        self.logger.experiment.add_scalar('val/(FR)Avg_PSNR', Avg_PSNR2, global_step=self.current_epoch)
        self.log('val/(FR)Avg_SSIM', Avg_SSIM2, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalar('val/(FR)Avg_SSIM', Avg_SSIM2, global_step=self.current_epoch)

