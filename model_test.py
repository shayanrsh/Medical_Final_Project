import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.insert(1, './src')

from model_scripts import Unet
from model_scripts import paste_and_save, eval_print_metrics
from dataset_loader import load_dataset


def model_test(net, batch_size=2):
    x_tensor, y_tensor, m_tensor = load_dataset(mode='test', resize=True, resize_shape=(256, 256))
    sample_numbers = x_tensor.shape[0]
    iteration_number = int(np.ceil(sample_numbers / batch_size))
    for iteration in range(iteration_number):
        print("{}th batch prediction".format(iteration + 1))
        if iteration == iteration_number - 1:
            start = iteration * batch_size
            batch_image = torch.Tensor(x_tensor[start:, :, :, :])
            batch_label = torch.Tensor(y_tensor[start:, 0: 1, :, :])
            batch_mask = torch.Tensor(m_tensor[start:, 0: 1, :, :])
        else:
            start, end = iteration * batch_size, (iteration + 1) * batch_size
            batch_image = torch.Tensor(x_tensor[start: end, :, :, :])
            batch_label = torch.Tensor(y_tensor[start: end, 0: 1, :, :])
            batch_mask = torch.Tensor(m_tensor[start: end, 0: 1, :, :])
        batch_prediction = net(batch_image)
        eval_print_metrics(batch_label, batch_prediction, batch_mask)

        batch_prediction_class = batch_prediction.detach() * batch_mask
        paste_and_save(batch_image, batch_label, batch_prediction_class, batch_size, iteration + 1)

    plt.imshow(batch_prediction[0, 0, :, :].detach().numpy(), cmap='jet')
    plt.colorbar()
    plt.show()

    return


if __name__ == "__main__":
    if not os.path.exists("storage/pred_imgs"):
        os.mkdir("storage/pred_imgs")
    if not os.path.exists("storage/datasets/test"):
        os.mkdir("storage/datasets/test")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected_model = glob("storage/checkpoint/Unet_epoch*.model")[-1]
    print("testing: {} ".format(selected_model))
    unet_ins = Unet(img_ch=3, isDeconv=True, isBN=True)
    unet_ins.load_state_dict(torch.load(selected_model))
    unet_ins.to(device)
    model_test(unet_ins, batch_size=2)
