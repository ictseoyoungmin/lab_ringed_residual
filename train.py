# -*- coding: utf-8 -*-
"""
defacto dataset으로 RRU-Net 네트워크 훈련 코드
python train.py로 네트워크 훈련한 다음 RRU_train_test.ipynb 파일에서 결과 확인
가까운 시일 내 predict.py로 성능평가 해야 함 
"""

import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch import optim
from model.unet_model import Ringed_Res_Unet
from dataset.Defacto import load_dataset
import matplotlib.pyplot as plt
import time,os
# __________________________________________
from loss.dice_loss import dice_coeff

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=1e-2,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=1,
              dataset=None,
              dir_logs=None):
    # training images are square
    # ids = split_ids(get_ids(dir_img))
    # iddataset = split_train_val(ids, val_percent)

    train_dataloader, val_dataloader = load_dataset(1000,batch_size,
                                                    dir_img =  r'E:\splicing_1_img\img',
                                                    dir_mask =  r"E:\splicing_1_annotations\donor_mask"
    )


    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs,
               batch_size,
               lr,
               train_dataloader.__len__(),
               val_dataloader.__len__(),
               str(save_cp),
               str(gpu)))
    # return 0
    N_train = train_dataloader.__len__()
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           weight_decay=0)
    criterion = nn.BCELoss()

    Train_loss  = []
    Valida_dice = []
    EPOCH = []
    spend_total_time = []
    max_loss = 0.0
    for epoch in range(epochs):
        net.train()

        start_epoch = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale, dataset)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale, dataset)

        epoch_loss = 0

        for i, data in enumerate(train_dataloader):
            start_batch = time.time()
            imgs = data['image']
            true_masks = data['landmarks']
            # imgs = np.array([i[0] for i in b]).astype(np.float32)
            # true_masks = np.array([i[1] for i in b]).astype(np.float32) / 255.

            # imgs = torch.from_numpy(imgs)
            # true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            optimizer.zero_grad()

            masks_pred = net(imgs)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_masks.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / N_train, loss, time.time()-start_batch))

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        with torch.no_grad():
            net.eval()
            tot = 0

            for i,val in enumerate(val_dataloader):
                img = val['image']
                true_mask = val['landmarks']
               
                if gpu:
                    img = img.cuda()
                    true_mask = true_mask.cuda()
               
                mask_pred = net(img)[0]
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
        plt.savefig(dir_logs + 'Training Process for lr-{}.png'.format(lr), dpi=600)
        plt.close()
        if epoch > 40:
            torch.save(net.state_dict(),
                   dir_logs + '{}-[val_dice]-{:.4f}-[train_loss]-{:.4f}.pkl'.format(dataset, val_dice, epoch_loss / i))
        spend_per_time = time.time() - start_epoch
        print('Spend time: {:.3f}s'.format(spend_per_time))
        spend_total_time.append(spend_per_time)
        print()
    print('Total time : {}'.format(sum(spend_total_time)))

def main():
    epochs, batchsize, scale, gpu = 50, 6, 1, True
    lr = 1e-3
    checkpoint = True
    ft = False
    dataset = "defactor"#'CASIA'
    model = 'Ringed_Res_Unet'

    

    dir_logs = './result/logs/{}/{}/'.format(dataset, model)
    os.makedirs(dir_logs,exist_ok=True)

    net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if checkpoint:
        net.load_state_dict(torch.load('./result/logs/{}/{}/defactor-[val_dice]-0.2679-[train_loss]-2.4120.pkl'.format(dataset, model)))
        print('Load checkpoint')
    if ft:
        fine_tuning_model = './result/logs/{}/{}/test.pkl'.format(dataset, model)
        net.load_state_dict(torch.load(fine_tuning_model))
        print('Model loaded from {}'.format(fine_tuning_model))

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              lr=lr,
              gpu=gpu,
              img_scale=scale,
              dataset=dataset,
              dir_logs=dir_logs)

if __name__ == '__main__':
    main()
