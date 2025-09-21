import os
import numpy as np
import torch.nn as nn
import torch

class SGDNet(nn.Module):
    def __init__(self, segmentation_model, out_ch=32):
        super(SGDNet, self).__init__()
        # Segmentation Model
        self.segmentation_model = segmentation_model
        
        # Encoder
        self.encoder = nn.Conv2d(1, out_ch, kernel_size=3, padding=1)

        # Semantic Fusion Module1
        self.ldct_encoder1_1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.ldct_encoder1_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.mask_encoder1_1 = nn.Conv2d(17, out_ch, kernel_size=3, padding=1)
        self.mask_encoder1_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.fusion1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Semantic Fusion Module2
        self.ldct_encoder2_1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.ldct_encoder2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.mask_encoder2_1 = nn.Conv2d(17, out_ch, kernel_size=3, padding=1)
        self.mask_encoder2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.fusion2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Semantic Fusion Module3
        self.ldct_encoder3_1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.ldct_encoder3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.mask_encoder3_1 = nn.Conv2d(17, out_ch, kernel_size=3, padding=1)
        self.mask_encoder3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.fusion3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Decoder
        self.decoder = nn.Conv2d(out_ch, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Segmentation
        with torch.no_grad():
            _, _, _, _, x_segs = self.segmentation_model(4096.*x-1024)
        
        # Encoder
        out = self.relu(self.encoder(x))
        residual_1 = out

        # Semantic Fusion Module1
        out = self.relu(self.ldct_encoder1_1(out))
        out = self.relu(self.ldct_encoder1_2(out))

        out2 = self.relu(self.mask_encoder1_1(x_segs))
        out2 = self.sigmoid(self.mask_encoder1_2(out2))

        out2 = out*out2
        out2 = self.relu(self.fusion1(out2))

        out = out + out2

        # Semantic Fusion Module2
        out = self.relu(self.ldct_encoder2_1(out))
        out = self.relu(self.ldct_encoder2_2(out))

        out2 = self.relu(self.mask_encoder2_1(x_segs))
        out2 = self.sigmoid(self.mask_encoder2_2(out2))

        out2 = out*out2
        out2 = self.relu(self.fusion2(out2))

        out = out + out2

        # Semantic Fusion Module3
        out = self.relu(self.ldct_encoder3_1(out))
        out = self.relu(self.ldct_encoder3_2(out))

        out2 = self.relu(self.mask_encoder3_1(x_segs))
        out2 = self.sigmoid(self.mask_encoder3_2(out2))

        out2 = out*out2
        out2 = self.relu(self.fusion3(out2))

        out = out + out2 + residual_1

        out = self.relu(self.decoder(out))
        
        return out