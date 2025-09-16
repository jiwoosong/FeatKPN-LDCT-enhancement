import os, sys
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
class unet(nn.Module):
    def __init__(self, nb_classes=17, inter_feat=False):
        super().__init__()
        self.inter_feat = inter_feat
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling2d = nn.MaxPool2d(2, stride=2)
        self.UpSampling2D = nn. Upsample(scale_factor=2, mode='nearest')

        self.conv1A = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.layer_normalization = LayerNorm( features = [64,1,1] ,eps=1e-3)
        self.conv1B = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.layer_normalization_1 = LayerNorm( features = [64,1,1] ,eps=1e-3)

        self.conv2A = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.layer_normalization_2 = LayerNorm( features = [128,1,1] ,eps=1e-3)
        self.conv2B = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.layer_normalization_3 = LayerNorm( features = [128,1,1] ,eps=1e-3)

        self.conv3A = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.layer_normalization_4 = LayerNorm( features = [256,1,1] ,eps=1e-3)
        self.conv3B = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.layer_normalization_5 = LayerNorm( features = [256,1,1] ,eps=1e-3)

        self.conv4A = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)
        self.layer_normalization_6 = LayerNorm( features = [512,1,1] ,eps=1e-3)
        self.conv4B = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.layer_normalization_7 = LayerNorm( features = [512,1,1] ,eps=1e-3)

        self.conv5A = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=1)
        self.layer_normalization_8 = LayerNorm( features = [1024,1,1] ,eps=1e-3)
        self.conv5B = nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1)
        self.layer_normalization_9 = LayerNorm( features = [1024,1,1] ,eps=1e-3)

        self.even_pad = torch.nn.ZeroPad2d((0, 1, 0, 1))
        self.conv6A = nn.Conv2d(1024, 512, kernel_size=(2, 2), padding=0) # Even Kernel Conv
        self.conv6B = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=1) # Merge Conv
        self.layer_normalization_10 = LayerNorm( features = [512,1,1] ,eps=1e-3)
        self.conv6C = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)
        self.layer_normalization_11 = LayerNorm( features = [512,1,1] ,eps=1e-3)

        self.conv7A = nn.Conv2d(512, 256, kernel_size=(2, 2), padding=0) # Even Kernel Conv
        self.conv7B = nn.Conv2d(512, 256, kernel_size=(3, 3), padding=1) # Merge Conv
        self.layer_normalization_12 = LayerNorm( features = [256,1,1] ,eps=1e-3)
        self.conv7C = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.layer_normalization_13 = LayerNorm( features = [256,1,1] ,eps=1e-3)

        self.conv8A = nn.Conv2d(256, 128, kernel_size=(2, 2), padding=0) # Even Kernel Conv
        self.conv8B = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1) # Merge Conv
        self.layer_normalization_14 = LayerNorm( features = [128,1,1] ,eps=1e-3)
        self.conv8C = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        self.layer_normalization_15 = LayerNorm( features = [128,1,1] ,eps=1e-3)

        self.conv9A = nn.Conv2d(128, 64, kernel_size=(2, 2), padding=0) # Even Kernel Conv
        self.conv9B = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1) # Merge Conv
        self.layer_normalization_16 = LayerNorm( features = [64,1,1] ,eps=1e-3)
        self.conv9C = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.layer_normalization_17 = LayerNorm( features = [64,1,1] ,eps=1e-3)

        self.conv10 = nn.Conv2d(64, nb_classes, kernel_size=(1, 1), padding=0)
    def forward(self, x):
        conv1 = self.conv1A(x)
        conv1 = self.relu(self.layer_normalization(conv1))
        conv1 = self.conv1B(conv1)
        conv1 = self.relu(self.layer_normalization_1(conv1))
        pool1 = self.max_pooling2d(conv1)

        conv2 = self.conv2A(pool1)
        conv2 = self.relu(self.layer_normalization_2(conv2))
        conv2 = self.conv2B(conv2)
        conv2 = self.relu(self.layer_normalization_3(conv2))
        pool2 = self.max_pooling2d(conv2)

        conv3 = self.conv3A(pool2)
        conv3 = self.relu(self.layer_normalization_4(conv3))
        conv3 = self.conv3B(conv3)
        conv3 = self.relu(self.layer_normalization_5(conv3))
        pool3 = self.max_pooling2d(conv3)

        conv4 = self.conv4A(pool3)
        conv4 = self.relu(self.layer_normalization_6(conv4))
        conv4 = self.conv4B(conv4)
        conv4 = self.relu(self.layer_normalization_7(conv4))
        pool4 = self.max_pooling2d(conv4)

        conv5 = self.conv5A(pool4)
        conv5 = self.relu(self.layer_normalization_8(conv5))
        conv5 = self.conv5B(conv5)
        conv5 = self.relu(self.layer_normalization_9(conv5))

        up6 = self.conv6A(self.even_pad(self.UpSampling2D(conv5)))
        merge6 = torch.cat([conv4,up6],dim=1)
        conv6 = self.conv6B(merge6)
        conv6 = self.relu(self.layer_normalization_10(conv6))
        conv6 = self.conv6C(conv6)
        conv6 = self.relu(self.layer_normalization_11(conv6))


        up7 = self.conv7A(self.even_pad(self.UpSampling2D(conv6)))
        merge7 = torch.cat([conv3,up7],dim=1)
        conv7 = self.conv7B(merge7)
        conv7 = self.relu(self.layer_normalization_12(conv7))
        conv7 = self.conv7C(conv7)
        conv7 = self.relu(self.layer_normalization_13(conv7))

        up8 = self.conv8A(self.even_pad(self.UpSampling2D(conv7)))
        merge8 = torch.cat([conv2,up8],dim=1)
        conv8 = self.conv8B(merge8)
        conv8 = self.relu(self.layer_normalization_14(conv8))
        conv8 = self.conv8C(conv8)
        conv8 = self.relu(self.layer_normalization_15(conv8))

        up9 = self.conv9A(self.even_pad(self.UpSampling2D(conv8)))
        merge9 = torch.cat([conv1,up9],dim=1)
        conv9 = self.conv9B(merge9)
        conv9 = self.relu(self.layer_normalization_16(conv9))
        conv9 = self.conv9C(conv9)
        conv9 = self.relu(self.layer_normalization_17(conv9))

        conv10 = self.conv10(conv9)
        if self.inter_feat:
            return conv6,conv7,conv8,conv9,conv10
        else:
            return conv10

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    '''
    Usage:
    for i in range(100):
        printProgress(i+1, 99, 'Progress:', '', 1, 50)
    '''
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    if iteration == total:
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', 'Done'))
        sys.stdout.write('\n')
    else:
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    sys.stdout.flush()
def run_Anatomical_Segmentation(model, read_p = r'', save_p = r'', device='cuda:0', batch_size=1, half_precision=False):
    nifti = nib.load(read_p)
    input_data = nifti.get_fdata()
    input_data = np.moveaxis(input_data, -1, 0)
    input_data = torch.tensor(np.expand_dims(input_data, 1).astype(np.float32))

    input_list = []
    for idx in range(0, input_data.shape[0],batch_size):
        if idx + batch_size > input_data.shape[0]:
            input_list.append(input_data[idx:])
        else:
            input_list.append(input_data[idx:idx+batch_size])

    if half_precision:
        model = model.half()
    output_list = []
    with torch.no_grad():
        for idx, img in enumerate(input_list):
            img = img.to(device)
            if half_precision:
                img = img.half()
            pred = model(img)
            output_list.append(pred)
            # plt.imshow(torch.argmax(out[0], dim=1)[0].cpu())
    prediction = torch.cat(output_list,0)
    prediction = torch.argmax(prediction, dim=1)
    prediction = prediction.int()
    prediction = prediction.cpu().numpy()
    nib.save(nib.Nifti1Image(prediction.transpose(1, 2, 0), nifti.affine), save_p)
    pass


if __name__ == '__main__':
    read_path = r'A:\HeadCTSegmentation-master\_Anatomical_Segmentation\Data'
    save_path = r'A:\HeadCTSegmentation-master\_Anatomical_Segmentation\Results'
    device='cuda:0'
    weight_path = r'A:\HeadCTSegmentation-master\_Anatomical_Segmentation\model.pth'

    model = unet(nb_classes=17)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model = model.to(device)

    for idx,p in enumerate(os.listdir(read_path)):
        for  nii_p in os.listdir(os.path.join(read_path,p)):
            if nii_p.__contains__('.nii'):
                read_p = os.path.join(read_path, p, nii_p)
                if not os.path.isdir(os.path.join(save_path, p)):
                    os.makedirs(os.path.join(save_path, p))
                save_p = os.path.join(save_path, p, nii_p)
                run_Anatomical_Segmentation(model, read_p=read_p, save_p=save_p, device=device, batch_size=16, half_precision=True)
        printProgress(idx+1, len(os.listdir(read_path)), 'Anatomical Segmentation', p, 1, 50)


