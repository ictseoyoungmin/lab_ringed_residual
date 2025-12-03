# -*- coding: utf-8 -*-
"""
defacto dataset으로 RRU-Net 네트워크 훈련 코드
python train.py로 네트워크 훈련한 다음 RRU_train_test.ipynb 파일에서 결과 확인
가까운 시일 내 predict.py로 성능평가 해야 함
"""

import os
import time

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim

from dataset.Defacto import load_dataset
from loss.dice_loss import dice_coeff
from model.unet_model import Ringed_Res_Unet


class Trainer:
    def __init__(self, net, train_dataloader, val_dataloader, optimizer, criterion, device, dir_logs, dataset, lr, image_size, checkpoint=True):
        self.net = net
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dir_logs = dir_logs
        self.dataset = dataset
        self.lr = lr
        self.image_size = image_size
        self.checkpoint = checkpoint

        self.train_losses = []
        self.val_dices = []
        self.epochs_run = []
        self.spend_total_time = []

        print(f'''
    Starting training:
        Batch size: {train_dataloader.batch_size}
        Image size: {image_size}
        Learning rate: {lr}
        Training size: {len(train_dataloader)}
        Validation size: {len(val_dataloader)}
        Checkpoints: {str(checkpoint)}
        CUDA: {str(device.type == "cuda")}
    ''')

    def train_one_epoch(self, epoch):
        self.net.train()
        epoch_loss = 0.0
        start_epoch = time.time()
        n_train = len(self.train_dataloader) * self.train_dataloader.batch_size

        print('Starting epoch {}/{}.'.format(epoch + 1, self.total_epochs))
        batch_idx = 0
        for batch_idx, data in enumerate(self.train_dataloader, 1):
            start_batch = time.time()
            imgs = data['image'].to(self.device)
            true_masks = data['landmarks'].to(self.device)

            self.optimizer.zero_grad()

            masks_pred = self.net(imgs)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = self.criterion(masks_probs_flat, true_masks_flat)

            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(batch_idx * self.train_dataloader.batch_size / n_train, loss, time.time() - start_batch))

        mean_loss = epoch_loss / max(batch_idx, 1)
        spend_per_time = time.time() - start_epoch
        print('Epoch finished ! Loss: {:.4f}'.format(mean_loss))
        print('Spend time: {:.3f}s'.format(spend_per_time))
        self.spend_total_time.append(spend_per_time)
        return mean_loss

    def validate(self):
        self.net.eval()
        tot = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_count, val in enumerate(self.val_dataloader, 1):  # 느려지면 imgs -> img
                imgs = val['image'].to(self.device)
                true_mask = val['landmarks'].to(self.device)

                mask_pred = self.net(imgs)[0]
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                tot += dice_coeff(mask_pred, true_mask).item()

        val_dice = tot / max(batch_count, 1)
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))
        return val_dice

    def save_checkpoint(self, epoch, val_dice, train_loss):
        if self.checkpoint and epoch < 140:
            checkpoint_path = os.path.join(
                self.dir_logs,
                '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}-ep{}.pkl'.format(self.dataset, val_dice, train_loss, epoch + 15)
            )
            torch.save(self.net.state_dict(), checkpoint_path)

    def log_metrics(self):
        fig = plt.figure()
        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(self.epochs_run, self.train_losses, c='red')
        l2, = plt.plot(self.epochs_run, self.val_dices, c='blue')

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_dice'], loc='best')
        plt.savefig(self.dir_logs + 'Training Process for lr-{}.png'.format(self.lr), dpi=600)
        plt.close()

    def run(self, epochs):
        self.total_epochs = epochs
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(epoch)
            val_dice = self.validate()

            self.train_losses.append(train_loss)
            self.val_dices.append(val_dice)
            self.epochs_run.append(epoch)

            self.log_metrics()
            self.save_checkpoint(epoch, val_dice, train_loss)
            print()

        total_time = int(sum(self.spend_total_time))
        print('Total time : {}m {}s'.format(total_time // 60, total_time % 60))


def train_net(net,
              epochs=5,
              batch_size=1,
              img_size=512,
              lr=1e-2,
              checkpoint=True,
              gpu=True,
              dataset=None,
              dir_logs=None,
              dir_image=r'E:\\splicing_1_img\\img_jpg',
              dir_mask=r"E:\\splicing_1_annotations\\probe_mask"
              ):
    train_dataloader, val_dataloader = load_dataset(
        10764,
        img_size,
        batch_size,
        dir_img=dir_image,
        dir_mask=dir_mask
    )

    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    trainer = Trainer(
        net=net,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        dir_logs=dir_logs,
        dataset=dataset,
        lr=lr,
        image_size=img_size,
        checkpoint=checkpoint
    )

    trainer.run(epochs)


def main():
    # config parameters
    epochs = 3
    batchsize = 2
    image_size = 512 #  x(512,512,3) y(512,512,1)
    gpu = True
    lr = 1e-3
    checkpoint = True
    dataset = "defactor" #'CASIA'
    model = 'Ringed_Res_Unet'
    dir_logs = './result/logs/{}/{}/'.format(dataset, model)
    dir_image=r'E:\\splicing_2_img\\img_jpg'
    dir_mask =  r"E:\\splicing_2_annotations\\probe_mask"

    # log directory 생성
    os.makedirs(dir_logs,exist_ok=True)

    net = Ringed_Res_Unet(n_channels=3, n_classes=1)
    # 훈련 epoch 나눠서 진행 할 때 True 사용
    if checkpoint: # epoch 3-img_1 3-img_2 *4
        net.load_state_dict(torch.load('./result/logs/{}/{}/\
defactor-[val_dice]-0.7546-[train_loss]-0.3125-ep15.pkl'.format(dataset, model)))
        print('Load checkpoint')

    if gpu:
        net.cuda()
        # cudnn.benchmark = True  # faster convolutions, but more memory

    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              img_size=image_size,
              lr=lr,
              gpu=gpu,
              dataset=dataset,
              dir_logs=dir_logs,
              checkpoint=checkpoint,
              dir_image=dir_image,
              dir_mask=dir_mask)

if __name__ == '__main__':
    main()
