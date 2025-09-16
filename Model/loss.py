import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from Model.Anatomical_Segmentation.torch_model import unet

class CompoundLoss(_Loss):
    def __init__(self, blocks=[1, 2, 3, 4], mse_weight=1, resnet_weight=0.01):
        super(CompoundLoss, self).__init__()
        self.mse_weight = mse_weight
        self.resnet_weight = resnet_weight
        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.criterion = nn.MSELoss()
    def forward(self, input, target):
        loss_value = 0
        input_feats = self.model(torch.cat([input, input, input], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))
        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num

        img_loss = self.criterion(input, target)
        img_loss_weight = self.mse_weight

        feat_loss = loss_value
        feat_loss_weight = self.resnet_weight

        return img_loss, img_loss_weight, feat_loss, feat_loss_weight

class MS_SSIM_L1_LOSS(nn.Module):
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)
    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)
    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)
    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=1, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean(), loss_l1.mean(), loss_ms_ssim

class MS_SSIM_L1_custom(nn.Module):
    # Have to use cuda, otherwise the speed is too slow.
    def __init__(self, alpha=0.05, channel=1):
        super().__init__()
        self.alpha=alpha
        self.l1 = torch.nn.L1Loss()
    def forward(self, pred, y):
        L1_loss = self.l1(pred, y)
        MS_SSIM_loss = 1 - self.msssim(pred, y, normalize='relu')
        loss_mix = self.alpha * (MS_SSIM_loss) + (1 - self.alpha) * L1_loss
        return loss_mix, L1_loss, MS_SSIM_loss
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    def ssim(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            cs = cs.mean()
            ret = ssim_map.mean()
        else:
            cs = cs.mean(1).mean(1).mean(1)
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret
    def msssim(self, img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.ssim(img1, img2, window_size=window_size, size_average=size_average, full=True,
                                val_range=val_range)

            # Relu normalize (not compliant with original definition)
            if normalize == "relu":
                ssims.append(torch.relu(sim))
                mcs.append(torch.relu(cs))
            else:
                ssims.append(sim)
                mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Simple normalize (not compliant with original definition)
        if normalize == "simple" or normalize == True:
            ssims = (ssims + 1) / 2
            mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1]) * pow2[-1]
        return output

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, blocks=[1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = torchvision.models.resnet50(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks
    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats

class Anatomical_Loss(nn.Module):
    def __init__(self, w_p):
        super().__init__()
        self.model = unet(nb_classes=17, inter_feat=True)
        self.model.load_state_dict(torch.load(w_p))
        # self.model = self.model.half()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()
        self.l1 = nn.L1Loss()
        self.MSE = nn.MSELoss()
    def forward(self, pred, y, mode):
        pred = pred * 4096 - 1024
        y = y * 4096 - 1024
        if mode == 'l1_l1ana_1':
            pf6, pf7, pf8, pf9, pout = self.model(pred)
            with torch.no_grad():
                yf6, yf7, yf8, yf9, yout = self.model(y)
            l6 = self.l1(pf6, yf6)
            l7 = self.l1(pf7, yf7)
            l8 = self.l1(pf8, yf8)
            l9 = self.l1(pf9, yf9)
            l10 = self.l1(pout, yout)
            Loss = (l6+l7+l8+l9+l10)/5
            return Loss

        elif mode == 'l1_l1ana_2':
            ana_chroi = self.ana_chroi
            _, _, _, _, pout = self.model(pred)
            with torch.no_grad():
                _, _, _, _, yout = self.model(y)
            Loss = self.l1(pout[:,ana_chroi], yout[:,ana_chroi])
            return Loss

        elif mode == 'Confusion':
            with torch.no_grad():
                pf6, pf7, pf8, pf9, pout = self.model(pred)
                yf6, yf7, yf8, yf9, yout = self.model(y)

            l6 = self.MSE(pf6, yf6)
            l7 = self.MSE(pf7, yf7)
            l8 = self.MSE(pf8, yf8)
            l9 = self.MSE(pf9, yf9)
            l10 = self.MSE(pout, yout)
            Loss = (l6 + l7 + l8 + l9 + l10) / 5

            pout = torch.argmax(pout, dim=1)
            yout = torch.argmax(yout, dim=1)
            TP = {}
            TN = {}
            FP = {}
            FN = {}
            for label in range(1,18):
                p_mask = pout==label
                y_mask = yout==label

                TP[label] = torch.logical_and(p_mask==1, y_mask==1).sum()
                TN[label] = torch.logical_and(p_mask==0, y_mask==0).sum()
                FP[label] = torch.logical_and(p_mask==1, y_mask==0).sum()
                FN[label] = torch.logical_and(p_mask==0, y_mask==1).sum()

            return {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN, 'Loss':Loss}
        else:
            raise Exception
