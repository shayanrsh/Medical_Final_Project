import sys

sys.path.insert(1, './src')

import numpy as np
import torch
from torch import nn
import os
from dataset_loader import load_dataset
from model_scripts import Unet
from model_scripts import eval_print_metrics
import warnings

warnings.filterwarnings('ignore')


def model_train(network, epochs, batch_size, learning_rate, save_every, evaluation_every, is_eval):
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)

    x_tensor, y_tensor, m_tensor = load_dataset(mode="training", resize=True, resize_shape=(256, 256))
    sample_numbers = x_tensor.shape[0]

    for epoch in range(epochs):
        epoch_total_loss = 0
        iteration_number = int(np.ceil(sample_numbers / batch_size))
        shuffle = np.random.permutation(sample_numbers)
        x_tensor = x_tensor[shuffle, :, :, :]
        y_tensor = y_tensor[shuffle, :, :, :]
        m_tensor = m_tensor[shuffle, :, :, :]

        for iteration in range(iteration_number):
            if not iteration == iteration_number - 1:
                start, end = iteration * batch_size, (iteration + 1) * batch_size
                batch_image = torch.Tensor(x_tensor[start: end, :, :, :])
                batch_label = torch.Tensor(y_tensor[start: end, 0: 1, :, :])
                batch_mask = torch.Tensor(m_tensor[start: end, 0: 1, :, :])
            else:
                start = iteration * batch_size
                batch_image = torch.Tensor(x_tensor[start:, :, :, :])
                batch_label = torch.Tensor(y_tensor[start:, 0: 1, :, :])
                batch_mask = torch.Tensor(m_tensor[start:, 0: 1, :, :])
            criterion = nn.BCELoss()
            optimizer.zero_grad()
            batch_pred = network(batch_image)
            loss = criterion(batch_pred, batch_label.float())
            print("Epoch: {}, Iter: {} current loss: {:.4f}".format(epoch + 1, iteration + 1, loss.item()))
            if is_eval:
                if epoch % evaluation_every == 0:
                    print("Eval for Epoch: {}, Iter: {}".format(epoch + 1, iteration + 1))
                    eval_print_metrics(batch_label, batch_pred, batch_mask)

            if not iteration == iteration_number - 1:
                epoch_total_loss += loss.item()
            else:
                epoch_total_loss += loss.item()
                epoch_avg_loss = epoch_total_loss / (iteration + 1)
                print("Epoch {} finished, avg_loss : {:.4f}".format(epoch + 1, epoch_avg_loss))
            loss.backward()
            optimizer.step()
        if epoch % save_every == 0:
            torch.save(network.state_dict(),
                       "./storage/checkpoint/Unet_epoch{}_loss{:.4f}_retina.model".format(str(epoch + 1).zfill(5),
                                                                                          epoch_avg_loss))
    return network


if __name__ == "__main__":

    if not os.path.exists("storage/checkpoint"):
        os.mkdir("storage/checkpoint")
    if not os.path.exists("storage/datasets"):
        os.mkdir("storage/datasets")
    if not os.path.exists("storage/datasets/training"):
        os.mkdir("storage/datasets/training")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unet_ins = Unet(img_ch=3, isDeconv=True, isBN=True)
    unet_ins.to(device)
    trained_unet = model_train(unet_ins,
                               batch_size=8,
                               learning_rate=0.1,
                               epochs=5,
                               save_every=1,
                               evaluation_every=30,
                               is_eval=True)
