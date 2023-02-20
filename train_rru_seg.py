# -*- coding: utf-8 -*-
"""
defacto dataset으로 RRU-Net 네트워크 훈련 코드
python train.py로 네트워크 훈련한 다음 RRU_train_test.ipynb 파일에서 결과 확인
가까운 시일 내 predict.py로 성능평가 해야 함 
"""
import matplotlib.pyplot as plt
import time,os
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader,random_split
from model.unet_model import Ringed_Res_Unet,DCT_RRUnet
from dataset.Defacto import DefactoDataset
from loss.dice_loss import dice_coeff

def load_dataset(total_nums,img_size,batch_size,dir_img,dir_mask):
    # ImageNet 표준으로 정규화 <- 일반적인 관행

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                img_size,
                'train',None,
                blocks=['RGB'])

    # dataset_size = len(dataset)
    train_size =  10000 #int(dataset_size * 0.8)
    validation_size = 764 #int(dataset_size * 0.2)

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    
    return train_dataloader,val_dataloader

def train_net(net,
              epochs=5,
              batch_size=1,
              img_size=512,
              lr=1e-2,
              checkpoint=True,
              gpu=True,
              dataset=None,
              dir_logs=None,
              dir_image=r'E:\splicing_2_img\img_jpg',
              dir_mask = r"E:\splicing_2_annotations\probe_mask",
              chk_ep=-1
              ):

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
        
        # {'image': image,'image':x, 'landmarks': y,'qtable': z}
        for i, data in enumerate(train_dataloader,1):
            start_batch = time.time()
            img = data['image']
            true_mask = data['landmarks']
            
            if gpu:
                img = img.cuda()
                true_mask = true_mask.cuda()

            optimizer.zero_grad()

            masks_pred = net(img)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)
            true_masks_flat = true_mask.view(-1)
            loss = criterion(masks_probs_flat, true_masks_flat)

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
                img = val['image']
                true_mask = val['landmarks']
                
                if gpu:
                    img = img.cuda()
                    true_mask = true_mask.cuda()

                mask_pred = net(img)
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                tot += dice_coeff(mask_pred, true_mask).item()

            val_dice = tot / i

        print('Validation Dice: {:.4f}'.format(val_dice))
        spend_per_time = time.time() - start_epoch
        print('Spend time: {:.3f}s'.format(spend_per_time))
        spend_total_time.append(spend_per_time)
        print()
        Train_loss.append(epoch_loss / i)
        Valida_dice.append(val_dice)
        EPOCH.append(epoch)
            
        fig = plt.figure()
        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(EPOCH, Train_loss, c='red')
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_Dice'], loc='best')
        plt.savefig(dir_logs + 'Training Process for dct-{}.png'.format(lr), dpi=600)
        plt.close()

        if epoch < 140:
            torch.save(net.state_dict(),
                   dir_logs + 'RRU-[val_dice]-{:.4f}-[train_loss]-{:.4f}-ep{}.pkl'.format(val_dice, epoch_loss / i,chk_ep+epoch+1))
        

    Tt = int(sum(spend_total_time))    
    print('Spend times : ',spend_total_time,'   mean : ',Tt/len(spend_total_time))
    print('Total time : {}m {}s'.format(Tt//60,Tt%60))


def main():
    # config parameters
    epochs = 6
    batchsize = 1
    image_size = 512 #  x(512,512,3) y(512,512,1)
    gpu = True
    lr = 1e-3
    dataset = "defactor" #'CASIA'
    model = 'RRUNet'
    dir_logs = './result/logs/{}/{}/'.format(dataset, model)
    dir_image=r'F:\datasets\Defacto_splicing\splicing_2_img\img_jpg'
    dir_mask =  r"F:\datasets\Defacto_splicing\splicing_2_annotations\probe_mask"
    checkpoint = True
    transfer_learing = False # RRU의 다운 샘플 블록 전이학습 한번 진행 
    chk_model = "RRU-[val_dice]-0.6735-[train_loss]-0.3139-ep18.pkl"
    chk_ep = 18#int(tl_model[-5]) if checkpoint else 0
    print("chk-ep: ",chk_ep)
    # log directory 생성
    os.makedirs(dir_logs,exist_ok=True)

    net = Ringed_Res_Unet(n_channels=3, n_classes=1) # 1ep 7633.230s
    # 훈련 epoch 나눠서 진행 할 때 True 사용

    if checkpoint: # epoch 3-img_1 3-img_2 *4
        net.load_state_dict(torch.load(os.path.join('./result/logs/',dataset, model,chk_model)))
        print('Load checkpoint')

    if gpu:
        net.cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory

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
              dir_mask=dir_mask,
              chk_ep=chk_ep)

if __name__ == '__main__':
    main()
