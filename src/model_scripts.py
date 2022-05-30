import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ConvBNReLU(nn.Module):

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvBNReLU, self).__init__()
        if not isBN:
            self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_bn_relu = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv_bn_relu(x)


class ConvNoPool(nn.Module):

    def __init__(self, in_ch, out_ch, is_BN=True):
        super(ConvNoPool, self).__init__()
        self.conv_no_pool = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, is_BN),
            ConvBNReLU(out_ch, out_ch, is_BN)
        )

    def forward(self, x):
        return self.conv_no_pool(x)


class ConvPool(nn.Module):

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvPool, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x):
        return self.conv_pool(x)


class UpsampleConv(nn.Module):

    def __init__(self, in_ch, out_ch, is_deconv=True, is_BN=True):
        super(UpsampleConv, self).__init__()
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
        self.conv_twice = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, is_BN),
            ConvBNReLU(out_ch, out_ch, is_BN)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_twice(x)
        return x


class ConvOut(nn.Module):

    def __init__(self, in_ch):
        super(ConvOut, self).__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.Conv2d(in_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output_ = self.conv_out(x)
        return output_


class Unet(nn.Module):

    def __init__(self, img_ch, fch_base=16, is_BN=True, is_deconv=True):
        super(Unet, self).__init__()

        self.blocks = nn.ModuleList()

        self.down1 = ConvNoPool(img_ch, fch_base, is_BN)
        self.down2 = ConvPool(fch_base, fch_base * 2, is_BN)
        self.down3 = ConvPool(fch_base * 2, fch_base * 4, is_BN)
        self.down4 = ConvPool(fch_base * 4, fch_base * 8, is_BN)

        self.encoder = ConvPool(fch_base * 8, fch_base * 16, is_BN)

        self.up1 = UpsampleConv(fch_base * 16, fch_base * 8, is_deconv, is_BN)
        self.up2 = UpsampleConv(fch_base * 8, fch_base * 4, is_deconv, is_BN)
        self.up3 = UpsampleConv(fch_base * 4, fch_base * 2, is_deconv, is_BN)
        self.up4 = UpsampleConv(fch_base * 2, fch_base, is_deconv, is_BN)

        self.out = ConvOut(fch_base)

        self.blocks = nn.ModuleList([self.down1, self.down2, self.down3, \
                                     self.down4, self.encoder, self.up1, self.up2, \
                                     self.up3, self.up4, self.out])

    def forward(self, input_):
        d1 = self.down1(input_)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        enc = self.encoder(d4)
        u1 = self.up1(enc, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        output_ = self.out(u4)
        return output_


def print_metrics_evaluation(batch_label, batch_pred, batch_mask):

    masked_pred = batch_pred * batch_mask
    masked_label = batch_label * batch_mask
    masked_prediction_class = (masked_pred > 0.5).float()

    precision = float(torch.sum(masked_prediction_class * masked_label)) / (float(torch.sum(masked_prediction_class)) + 1)
    recall = float(torch.sum(masked_prediction_class * masked_label)) / (float(torch.sum(masked_label.float())) + 1)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-8)

    prediction_ls = np.array(batch_pred[batch_mask > 0].detach())
    label_ls = np.array(batch_label[batch_mask > 0].detach(), dtype=np.int)
    batch_auc = roc_auc_score(label_ls, prediction_ls)

    print(" >>> precision: {:.4f} recall: {:.4f} f1_score: {:.4f} auc: {:.4f}".format(precision, recall, f1_score,
                                                                                      batch_auc))

    return precision, recall, f1_score, batch_auc


def paste_and_save(batch_image, batch_label, batch_pred_class, batch_size, cur_batch_number, save_img='./storage/pred_imgs'):
    w, h = batch_image.size()[2:4]
    for batch in range(batch_image.size()[0]):
        image = Image.fromarray(np.moveaxis(np.array(batch_image * 255.0, dtype=np.uint8)[batch, :, :, :], 0, 2))
        label = Image.fromarray(np.array(batch_label * 255.0, dtype=np.uint8)[batch, 0, :, :])
        prediction_class = Image.fromarray(np.array(batch_pred_class * 255.0, dtype=np.uint8)[batch, 0, :, :])

        result = (cur_batch_number - 1) * batch_size + batch
        target = Image.new('RGB', (3 * w, h))
        target.paste(image, box=(0, 0))
        target.paste(label, box=(w, 0))
        target.paste(prediction_class, box=(2 * w, 0))

        target.save(os.path.join(save_img, "result_{}.png".format(result)))
    return
