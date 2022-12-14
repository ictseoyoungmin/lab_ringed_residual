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
from model.dct_model import FC_DCT_Only
from dataset.Defacto import load_dataset
import matplotlib.pyplot as plt
import time,os
# __________________________________________
from loss.dice_loss import dice_coeff

def train_net(net,
              epochs=5,
              batch_size=1,
              img_size=512,
              lr=1e-2,
              checkpoint=True,
              gpu=True,
              dataset=None,
              dir_logs=None,
              dir_image=r'E:\splicing_1_img\img_jpg',
              dir_mask = r"E:\splicing_1_annotations\probe_mask"
              ):
    # training images are square
    # ids = split_ids(get_ids(dir_img))
    # iddataset = split_train_val(ids, val_percent)

    # splicing_1_img 개수는 10765
    train_dataloader, val_dataloader = load_dataset(10764,
                                                    img_size,
                                                    batch_size,
                                                    dir_img =  dir_image,
                                                    dir_mask = dir_mask
    )

    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Image size: {img_size}
        Learning rate: {lr}
        Training size: {train_dataloader.__len__()}
        Validation size: {val_dataloader.__len__()}
        Checkpoints: {str(checkpoint)}
        CUDA: {str(gpu)}
    ''')
    N_train = train_dataloader.__len__() * batch_size

    optimizer = optim.Adam(net.parameters(),
                           lr=lr)
    criterion = nn.BCELoss()
    Train_loss  = []
    Valida_dice = []
    EPOCH = []
    spend_total_time = []

    for epoch in range(epochs):
        net.train()
        start_epoch = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        
        # {'image': image,'artifact':x, 'landmarks': y,'qtable': z}
        for i, data in enumerate(train_dataloader,1):
            start_batch = time.time()
            jpg_artifact = data['artifact']
            qtable = data['qtable']
            true_label = torch.ones((batch_size,)).squeeze(-1)
            
            if gpu:
                jpg_artifact = jpg_artifact.cuda()
                true_label = true_label.cuda()
                qtable = qtable.cuda()

            optimizer.zero_grad()

            pred = net(jpg_artifact,qtable)
            probs = torch.sigmoid(pred)
            probs_flat = probs.view(-1)
            true_label_flat = true_label.view(-1)
            loss = criterion(probs_flat, true_label_flat)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            print('{:.4f} --- loss: {:.4f}, {:.3f}s'.format(i * batch_size / N_train, loss, time.time()-start_batch))

        print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / i))

        # validate the performance of the model
        with torch.no_grad():
            net.eval()
            tot = 0.0

            for i,val in enumerate(val_dataloader):
                jpg_artifact = data['artifact']
                qtable = data['qtable']
                true_label = torch.ones((batch_size,)).squeeze(-1)
            
                if gpu:
                    jpg_artifact = jpg_artifact.cuda()
                    true_label = true_label.cuda()
                    qtable = qtable.cuda()

                pred = net(jpg_artifact,qtable)
                probs = torch.sigmoid(pred)
                probs_flat = probs.view(-1)
                true_label_flat = true_label.view(-1)
                loss = criterion(probs_flat, true_label_flat)
                tot += loss.item()

            val_dice = tot / i
        print('Validation loss: {:.4f}'.format(val_dice))

        Train_loss.append(epoch_loss / i)
        Valida_dice.append(val_dice)
        EPOCH.append(epoch)
            
        fig = plt.figure()
        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(EPOCH, Train_loss, c='red')
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_loss'], loc='best')
        plt.savefig(dir_logs + 'Training Process for dct-{}.png'.format(lr), dpi=600)
        plt.close()

        if epoch < 140:
            torch.save(net.state_dict(),
                   dir_logs + 'DCT-[val_dice]-{:.4f}-[train_loss]-{:.4f}-ep{}.pkl'.format(val_dice, epoch_loss / i,epoch))
        spend_per_time = time.time() - start_epoch
        print('Spend time: {:.3f}s'.format(spend_per_time))
        spend_total_time.append(spend_per_time)
        print()

    Tt = int(sum(spend_total_time))    
    print('Total time : {}m {}s'.format(Tt//60,Tt%60))

def main():
    # config parameters
    epochs = 3
    batchsize = 16
    image_size = 512 #  x(512,512,3) y(512,512,1)
    gpu = True
    lr = 1e-3
    checkpoint = False
    dataset = "defactor" #'CASIA'
    model = 'FC_DCT_Only'
    dir_logs = './result/logs/{}/{}/'.format(dataset, model)
    dir_image=r'E:\splicing_2_img\img_jpg'
    dir_mask =  r"E:\splicing_2_annotations\probe_mask"

    # log directory 생성
    os.makedirs(dir_logs,exist_ok=True)

    net = FC_DCT_Only(n_channels=512, n_classes=1)
    # 훈련 epoch 나눠서 진행 할 때 True 사용
    if checkpoint: # epoch 3-img_1 3-img_2 *4
        net.load_state_dict(torch.load('./result/logs/{}/{}/\
defactor-[val_dice]-0.7084-[train_loss]-0.3988-ep2.pkl'.format(dataset, model)))
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
