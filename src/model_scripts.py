import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# some replicate units for u-net
class ConvBNReLU(nn.Module):
    """
    combination of [conv] + [BN] + [ReLU]
    """

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvBNReLU, self).__init__()
        if isBN:
            self.convbnrelu = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.convbnrelu = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.convbnrelu(x)


class ConvNoPool(nn.Module):
    """
    conv twice and no pooling
    """

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvNoPool, self).__init__()
        self.convnopool = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x):
        return self.convnopool(x)


class ConvPool(nn.Module):
    """
    conv twice with a pooling layer follows
    """

    def __init__(self, in_ch, out_ch, isBN=True):
        super(ConvPool, self).__init__()
        self.convpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x):
        return self.convpool(x)


class UpsampleConv(nn.Module):
    """
    upsample feature maps to given shape and conv twice (with skip connection)
    """

    def __init__(self, in_ch, out_ch, isDeconv=True, isBN=True):
        super(UpsampleConv, self).__init__()
        if isDeconv:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=1)
            )
        self.convtwice = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, isBN),
            ConvBNReLU(out_ch, out_ch, isBN)
        )

    def forward(self, x1, x2):
        # this forward func is from (to solve the size incompatibility issue) :
        # https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        x1 = self.up(x1)
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        # print(x1.size(), x2.size())
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        # print(x1.size(), x2.size())
        x = torch.cat([x2, x1], dim=1)
        x = self.convtwice(x)
        return x


class ConvOut(nn.Module):
    """
    last layer for generating probability map
    """

    def __init__(self, in_ch):
        super(ConvOut, self).__init__()
        self.convout = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
            nn.Conv2d(in_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.size())
        output_ = self.convout(x)
        # print(output_.size())
        return output_


class Unet(nn.Module):

    def __init__(self, img_ch, fch_base=16, isBN=True, isDeconv=True):
        super(Unet, self).__init__()

        self.blocks = nn.ModuleList()

        self.down1 = ConvNoPool(img_ch, fch_base, isBN)
        self.down2 = ConvPool(fch_base, fch_base * 2, isBN)
        self.down3 = ConvPool(fch_base * 2, fch_base * 4, isBN)
        self.down4 = ConvPool(fch_base * 4, fch_base * 8, isBN)

        self.encoder = ConvPool(fch_base * 8, fch_base * 16, isBN)

        self.up1 = UpsampleConv(fch_base * 16, fch_base * 8, isDeconv, isBN)
        self.up2 = UpsampleConv(fch_base * 8, fch_base * 4, isDeconv, isBN)
        self.up3 = UpsampleConv(fch_base * 4, fch_base * 2, isDeconv, isBN)
        self.up4 = UpsampleConv(fch_base * 2, fch_base, isDeconv, isBN)

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


def eval_print_metrics(bat_label, bat_pred, bat_mask):
    assert len(bat_label.size()) == 4 \
           and len(bat_pred.size()) == 4 and len(bat_mask.size()) == 4
    assert bat_label.size()[1] == 1 \
           and bat_pred.size()[1] == 1 and bat_mask.size()[1] == 1

    masked_pred = bat_pred * bat_mask
    masked_label = bat_label * bat_mask
    masked_pred_class = (masked_pred > 0.5).float()

    precision = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_pred_class)) + 1)
    recall = float(torch.sum(masked_pred_class * masked_label)) / (float(torch.sum(masked_label.float())) + 1)
    f1_score = 2.0 * precision * recall / (precision + recall + 1e-8)

    pred_ls = np.array(bat_pred[bat_mask > 0].detach())
    label_ls = np.array(bat_label[bat_mask > 0].detach(), dtype=np.int)
    bat_auc = roc_auc_score(label_ls, pred_ls)

    print("[*] ...... Evaluation ...... ")
    print(" >>> precision: {:.4f} recall: {:.4f} f1_score: {:.4f} auc: {:.4f}".format(precision, recall, f1_score,
                                                                                      bat_auc))

    return precision, recall, f1_score, bat_auc


def paste_and_save(bat_img, bat_label, bat_pred_class, batch_size, cur_bat_num, save_img='./storage/pred_imgs'):
    w, h = bat_img.size()[2:4]
    for bat_id in range(bat_img.size()[0]):
        # img = Image.fromarray(np.moveaxis(np.array((bat_img + 1) / 2 * 255.0, dtype=np.uint8)[bat_id, :, :, :], 0, 2))
        img = Image.fromarray(np.moveaxis(np.array(bat_img * 255.0, dtype=np.uint8)[bat_id, :, :, :], 0, 2))
        label = Image.fromarray(np.array(bat_label * 255.0, dtype=np.uint8)[bat_id, 0, :, :])
        pred_class = Image.fromarray(np.array(bat_pred_class * 255.0, dtype=np.uint8)[bat_id, 0, :, :])

        res_id = (cur_bat_num - 1) * batch_size + bat_id
        target = Image.new('RGB', (3 * w, h))
        target.paste(img, box=(0, 0))
        target.paste(label, box=(w, 0))
        target.paste(pred_class, box=(2 * w, 0))

        target.save(os.path.join(save_img, "result_{}.png".format(res_id)))
    return
