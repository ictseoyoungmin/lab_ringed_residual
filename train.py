# -*- coding: utf-8 -*-
"""
defacto dataset으로 RRU-Net 네트워크 훈련 코드
python train.py로 네트워크 훈련한 다음 RRU_train_test.ipynb 파일에서 결과 확인
가까운 시일 내 predict.py로 성능평가 해야 함 
"""

import importlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch import nn, optim

from dataset.Defacto import load_dataset
from loss.dice_loss import dice_coeff
from model.unet_model import Ringed_Res_Unet

yaml_spec = importlib.util.find_spec("yaml")
yaml = importlib.import_module("yaml") if yaml_spec else None


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 1
    img_size: int = 512
    lr: float = 1e-2
    checkpoint: bool = True
    gpu: bool = True
    dataset: str = "defactor"
    dir_logs: str = './result/logs/defactor/Ringed_Res_Unet/'
    dir_image: str = r'E:\\splicing_1_img\\img_jpg'
    dir_mask: str = r"E:\\splicing_1_annotations\\probe_mask"
    model: str = 'Ringed_Res_Unet'
    checkpoint_path: Optional[str] = None

    @staticmethod
    def from_file(path: str) -> "TrainingConfig":
        with open(path, 'r') as f:
            if path.lower().endswith(('.yaml', '.yml')):
                if yaml is None:
                    raise ImportError("PyYAML is required to load YAML configuration files.")
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return TrainingConfig(**data)

def train_net(net, config: TrainingConfig):
    # training images are square
    # ids = split_ids(get_ids(dir_img))
    # iddataset = split_train_val(ids, val_percent)

    # splicing_1_img 개수는 10765
    train_dataloader, val_dataloader = load_dataset(
        10764,
        config.img_size,
        config.batch_size,
        dir_img=config.dir_image,
        dir_mask=config.dir_mask
    )

    print(f'''
    Starting training:
        Epochs: {config.epochs}
        Batch size: {config.batch_size}
        Image size: {config.img_size}
        Learning rate: {config.lr}
        Training size: {train_dataloader.__len__()}
        Validation size: {val_dataloader.__len__()}
        Checkpoints: {str(config.checkpoint)}
        CUDA: {str(config.gpu)}
    ''')
    N_train = train_dataloader.__len__() * config.batch_size

    optimizer = optim.Adam(net.parameters(),
                           lr=config.lr)
    criterion = nn.BCELoss()
    Train_loss  = []
    Valida_dice = []
    EPOCH = []
    spend_total_time = []

    for epoch in range(config.epochs):
        net.train()

        start_epoch = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, config.epochs))
        # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale, dataset)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale, dataset)

        epoch_loss = 0

        for i, data in enumerate(train_dataloader,1):
            start_batch = time.time()
            imgs = data['image']
            true_masks = data['landmarks']
            # imgs = np.array([i[0] for i in b]).astype(np.float32)
            # true_masks = np.array([i[1] for i in b]).astype(np.float32) / 255.

            # imgs = torch.from_numpy(imgs)
            # true_masks = torch.from_numpy(true_masks)

            if config.gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            optimizer.zero_grad()

            masks_pred = net(imgs)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * config.batch_size / N_train, loss, time.time()-start_batch))

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        with torch.no_grad():
            net.eval()
            tot = 0.0

            for i,val in enumerate(val_dataloader): #느려지면 imgs -> img
                imgs = val['image']
                true_mask = val['landmarks']

                if config.gpu:
                    imgs = imgs.cuda()
                    true_mask = true_mask.cuda()
                    # tot.cuda()

                mask_pred = net(imgs)[0]
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                tot += dice_coeff(mask_pred, true_mask).item()

            val_dice = tot / i
        print('Validation Dice Coeff: {:.4f}'.format(val_dice))

        Train_loss.append(epoch_loss / i)
        Valida_dice.append(val_dice)
        EPOCH.append(epoch)
            
        fig = plt.figure()
        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(EPOCH, Train_loss, c='red')
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_dice'], loc='best')
        plt.savefig(config.dir_logs + 'Training Process for lr-{}.png'.format(config.lr), dpi=600)
        plt.close()

        if epoch < 140:
            torch.save(net.state_dict(),
                   config.dir_logs + '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}-ep{}.pkl'.format(config.dataset, val_dice, epoch_loss / i,epoch+15))
        spend_per_time = time.time() - start_epoch
        print('Spend time: {:.3f}s'.format(spend_per_time))
        spend_total_time.append(spend_per_time)
        print()

    Tt = int(sum(spend_total_time))    
    print('Total time : {}m {}s'.format(Tt//60,Tt%60))


def train_from_config(config: TrainingConfig):
    os.makedirs(config.dir_logs, exist_ok=True)

    net = Ringed_Res_Unet(n_channels=3, n_classes=1)
    if config.checkpoint and config.checkpoint_path:
        net.load_state_dict(torch.load(config.checkpoint_path))
        print('Load checkpoint')

    if config.gpu:
        net.cuda()

    train_net(net=net, config=config)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train Ringed Res-UNet with configuration file")
    parser.add_argument('--config', required=True, help='Path to JSON or YAML configuration file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig.from_file(args.config)
    train_from_config(config)

if __name__ == '__main__':
    main()
