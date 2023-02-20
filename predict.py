# -*- coding: utf-8 -*-
"""
모델 테스트 및 성능평가 작성 예정
아직 사용x
"""

from dataset.Defacto import *
from loss.dice_loss import SoftDiceLoss,dice_coeff_loss
from model.unet_model import Ringed_Res_Unet,DCT_RRUnet
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader


def test_dataset_dice(model,img_size,dir_img,dir_mask,name = 'RRU-Net',test_loader=None):
    model.eval()
    if test_loader == None:
        if dir_img == r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Tp_jpg":
            num = 3000
        else:
            num = 10000
        print("eval nums : ",num)
        testdata = DefactoDataset(dir_img,
                dir_mask,
                num,
                img_size,
                "0",None)
        test_dataloader = DataLoader(testdata,1)
    else:
        test_dataloader = test_loader

    test_dice = 0.0
    acc_count = 0 # only tamp
    if 'DCT' in name or 'DRRU' in name:
        for i,(data) in enumerate(test_dataloader,1):
            jpg_artifact = data['artifact']
            mask = torch.relu(data['landmarks']) # ignore index -1
            qtable =data['qtable']

            if torch.cuda.is_available() :
                jpg_artifact = jpg_artifact.cuda()
                mask = mask.cuda()
                qtable = qtable.cuda()
                model.cuda()

            with torch.no_grad():
                pred = model(jpg_artifact,qtable) # normalize none
          
            pred = torch.sigmoid(pred)
            d = dice_coeff_loss(pred, mask).item()
            test_dice += d
            pred = pred.squeeze(0)
            pred = pred.permute(1,2,0).detach().cpu().numpy()

            acc_count+= int(True in (pred>0.5))
    else:
        # transformi = transforms.Compose([
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        #         ])
        for i,(data) in enumerate(test_dataloader,1):
            img = data['image']
            mask = data['landmarks']

            if torch.cuda.is_available() :
                img = img.cuda()
                mask = mask.cuda()
                model.cuda()

            with torch.no_grad():
                pred = model(img) # normalize none
            pred = torch.sigmoid(pred)
            d = dice_coeff_loss(pred, mask).item()
            test_dice += d
            pred = pred.squeeze(0)
            pred = pred.permute(1,2,0).detach().cpu().numpy()
            acc_count+= int(True in (pred>0.5))
        
    test_dice = test_dice / i
    print("dice : ",test_dice)
    print('accuracy : ',acc_count/i)
    return test_dice


























def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def plot_img_and_mask(img, mask):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    plt.imshow(img)

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    plt.imshow(mask)
    plt.show()

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_gpu=True):
    net.eval()

    img = resize_and_crop(full_img, scale=scale_factor).astype(np.float32)
    img = np.transpose(normalize(img), (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(dim=0)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        mask = net(img)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()

    return mask > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def normalize(x):
    return x / 255



if __name__ == "__main__":
    scale, mask_threshold, cpu,  viz, no_save = 1, 0.5, False, False, False
    network = 'Ringed_Res_Unet'

    img = Image.open('your_test_img.png')
    model = 'result/logs/test.pkl'

    net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if not cpu:
        net.cuda()
        net.load_state_dict(torch.load(model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=scale,
                       out_threshold=mask_threshold,
                       use_gpu=not cpu)

    if viz:
        print("Visualizing results for image {}, close to continue ...".format(j))
        plot_img_and_mask(img, mask)

    if not no_save:
        result = mask_to_image(mask)

        if network == 'Unet':
            result.save('predict_u.png')
        elif network == 'Res_Unet':
            result.save('predict_ru.png')
        else:
            result.save('predict_rru.png')