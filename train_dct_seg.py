# -*- coding: utf-8 -*-
"""
defacto dataset으로 RRU-Net 네트워크 훈련 코드
python train.py로 네트워크 훈련한 다음 RRU_train_test.ipynb 파일에서 결과 확인
가까운 시일 내 predict.py로 성능평가 해야 함 
"""
import gc
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch import optim
from model.unet_model import Ringed_Res_Unet,DCT_RRUnet
from dataset.Defacto import load_dataset
import matplotlib.pyplot as plt
import time
# __________________________________________
from loss.dice_loss import dice_coeff
from utils.io import build_checkpoint_path, build_plot_path, ensure_log_dir

def train_net(net,
              epochs=5,
              batch_size=1,
              img_size=512,
              lr=1e-2,
              checkpoint=True,
              gpu=True,
              dataset=None,
              model=None,
              dir_image=r'E:\splicing_2_img\img_jpg',
              dir_mask = r"E:\splicing_2_annotations\probe_mask",
              initial_epoch=0
              ):
    if dataset is None or model is None:
        raise ValueError('dataset and model must be provided for logging.')

    ensure_log_dir(dataset, model)

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
        epoch_start_time = time.time()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        epoch_loss = 0
        
        # {'image': image,'artifact':x, 'landmarks': y,'qtable': z}
        for i, data in enumerate(train_dataloader,1):
            start_batch = time.time()
            jpg_artifact = data['artifact']
            qtable = data['qtable']
            true_mask = data['landmarks']
            
            if gpu:
                jpg_artifact = jpg_artifact.cuda()
                true_mask = true_mask.cuda()
                qtable = qtable.cuda()

            optimizer.zero_grad()

            masks_pred = net(jpg_artifact,qtable)
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
                jpg_artifact = val['artifact']
                qtable = val['qtable']
                true_mask = val['landmarks']
                
                if gpu:
                    jpg_artifact = jpg_artifact.cuda()
                    true_mask = true_mask.cuda()
                    qtable = qtable.cuda()

                mask_pred = net(jpg_artifact,qtable)
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                tot += dice_coeff(mask_pred, true_mask).item()

            val_dice = tot / i
        print('Validation Dice: {:.4f}'.format(val_dice))

        current_epoch = initial_epoch + epoch + 1
        Train_loss.append(epoch_loss / i)
        Valida_dice.append(val_dice)
        EPOCH.append(current_epoch)
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3.0)

        fig = plt.figure()
        plt.title('Training Process')
        plt.xlabel('epoch')
        plt.ylabel('value')
        l1, = plt.plot(EPOCH, Train_loss, c='red')
        l2, = plt.plot(EPOCH, Valida_dice, c='blue')

        plt.legend(handles=[l1, l2], labels=['Tra_loss', 'Val_Dice'], loc='best')
        plot_path = build_plot_path(dataset, model, lr, suffix='Training Process')
        plt.savefig(plot_path, dpi=600)
        plt.close()

        if True : #epoch < 140:
            checkpoint_path = build_checkpoint_path(dataset, model, current_epoch, val_dice, epoch_loss / i)
            torch.save(net.state_dict(), checkpoint_path)
        spend_per_time = time.time() - epoch_start_time
        print('Spend time: {:.3f}s'.format(spend_per_time))
        spend_total_time.append(spend_per_time)
        print()

    # for n,p in net.named_parameters():
    #     print(n,p)
    #     break

    Tt = int(sum(spend_total_time))    
    print('Total time : {}m {}s'.format(Tt//60,Tt%60))


def main():
    # config parameters
    epochs = 6
    batchsize = 1
    image_size = 512 #  x(512,512,3) y(512,512,1)
    gpu = True
    lr = 1e-3
    dataset = "defactor" #'CASIA'
    model = 'DRRUNet_transfer'
    dir_image=r'F:\datasets\Defacto_splicing\splicing_2_img\img_jpg'
    dir_mask =  r"F:\datasets\Defacto_splicing\splicing_2_annotations\probe_mask"
    log_dir = ensure_log_dir(dataset, model)
    checkpoint = False
    transfer_learing = True # RRU의 다운 샘플 블록 전이학습 한번 진행
    chk_model = ""
    tl_model = "RRU-[val_dice]-0.6810-[train_loss]-0.3083-ep19.pkl" # 19시작
    chk_ep = 0 #int(str(sorted(os.listdir(os.path.join('./result/logs/',dataset,model)))[-2])[-5])
    initial_epoch = 0
    print("chk-ep: ",chk_ep)

    net = DCT_RRUnet(n_channals=3, n_classes=1,mode = "ori") # ori - npnfreeze
    # 훈련 epoch 나눠서 진행 할 때 True 사용

    if checkpoint: # epoch 3-img_1 3-img_2 *4
        checkpoint_path = log_dir / chk_model
        net.load_state_dict(torch.load(checkpoint_path))
        initial_epoch = chk_ep
        print('Load checkpoint')

    if transfer_learing:
        # tr = Ringed_Res_Unet()
        tr = Ringed_Res_Unet()
        rru_log_dir = ensure_log_dir(dataset, "RRUNet")
        transfer_path = rru_log_dir / tl_model
        tr.load_state_dict(torch.load(
            transfer_path,
            map_location=torch.device('cpu')))
        for i in range(5): # 0~4
            if not i:
                net.down.load_state_dict(tr.down.state_dict())
                net.down.requires_grad_(False)
            else:
                eval(f"net.down{str(i)}.load_state_dict(tr.down{str(i)}.state_dict())")
                eval(f"net.down{str(i)}.requires_grad_(False)")
        print("Finish load tr weights")

    if gpu:
        net = torch.nn.DataParallel(net, device_ids=(0,)).cuda()
        cudnn.benchmark = True  # faster convolutions, but more memory
        cudnn.deterministic = False
        cudnn.enabled = True

    train_net(net=net,
              epochs=epochs,
              batch_size=batchsize,
              img_size=image_size,
              lr=lr,
              gpu=gpu,
              dataset=dataset,
              model=model,
              checkpoint=checkpoint,
              dir_image=dir_image,
              dir_mask=dir_mask,
              initial_epoch=initial_epoch)

if __name__ == '__main__':
    main()
