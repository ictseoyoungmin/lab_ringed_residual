# -*- coding: utf-8 -*-
"""
defacto-splicing Dataset utills 
데이터세트 정의 클래스, 데이터세트 불러오기 함수 구현
_crop_size = (256,256)
_grid_crop = True
_blocks = ('RGB', 'DCTvol', 'qtable')
tamp_list = None
DCT_channels = 1
"""
import os,sys
from .abstractDataset import AbstractDataset
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,random_split
import PIL.Image as Image
import cv2 as cv
import shutil as sh
import random

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# https://www.kaggle.com/code/alerium/defacto-test
class DefactoDataset(AbstractDataset):
    def __init__(self, im_root_dir,label_root_dir, num, img_size, mode='test' , transform=None, 
                # crop_size=(512,512), grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1 \
                    crop_size=(512,512), grid_crop=True, blocks=('RGB','DCTvol', 'qtable'), DCT_channels=1 \
                    ):
        
        super().__init__(crop_size, grid_crop, blocks, DCT_channels)
        fix_seed(42)
        self.im_root_dir = im_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.num = num
        self.mode = mode
        self.img_size = img_size
        self.df = pd.DataFrame({
            'image_path':[os.path.join(self.im_root_dir,_) for _ in sorted(os.listdir(self.im_root_dir))[:self.num]],
            'mask_path':[os.path.join(self.label_root_dir,_) for _ in sorted(os.listdir(self.label_root_dir))[:self.num]]
                         })
        # self.df = pd.DataFrame({
        #     'image_path':[_ for _ in sorted(os.listdir(self.im_root_dir))],
        #     'mask_path':[_ for _ in sorted(os.listdir(self.label_root_dir))]
        #                  })

        self.name = self.df['image_path'].iloc[:self.num]
        self.label = self.df['mask_path'].iloc[:self.num] 

    def __len__(self):
        return len(self.name)

    def _resize(self , sample):
        image, mask = sample[0], sample[1]
        n = self.img_size
        image = TF.resize(image , size=(n,n),interpolation=transforms.InterpolationMode.BICUBIC)
        mask = TF.resize(mask , size=(n,n) ,interpolation=transforms.InterpolationMode.BICUBIC)

        return image , mask
    
    def __getitem__(self,idx):
        mask = np.array(Image.open(self.df['mask_path'].iloc[idx]).convert("L"))
        mask[mask > 0] = 1
        # x,y,z = self._create_tensor(self.df['image_path'].iloc[idx], mask)
        # image = x[:3]

        # return {'image': image,'artifact':x, 'landmarks': y,'qtable': z}
        return self._create_tensor(self.df['image_path'].iloc[idx], mask)
    

def load_dataset(total_nums,img_size,batch_size,dir_img,dir_mask):
    # ImageNet 표준으로 정규화 <- 일반적인 관행

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                img_size,
                'train',None)

    # dataset_size = len(dataset)
    train_size =  10000 #int(dataset_size * 0.8)
    validation_size = 764 #int(dataset_size * 0.2)

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    return train_dataloader,val_dataloader

def test(model,device,index,mode,img_size,dir_img,dir_mask):
    model.eval()
    # transformi = transforms.Compose([
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
                # ])
    testdata = DefactoDataset(dir_img,
               dir_mask,
               12000,
               img_size,
               mode,None,blocks=('RGB',))

    img,mask,_ =testdata.__getitem__(index) 
    inp = torch.Tensor([img.numpy()]).to(device)
    with torch.no_grad():
        pred = model(inp) # normalize none

    img = img.permute(1,2,0)
    h,w= 512,512#np.minimum(512,img.shape[0]),np.minimum(512,img.shape[1])
    try:
        for i in range(len(img)-1,0,-1):
            if img[0,i,0]!=0.:
                w = i  
                break
        for i in range(len(img)-1,0,-1):
            if img[i,0,0]!=0.:
                h = i  
                break
        img = img[:h,:w,:]
    except Exception as e:
        raise (f"""
            h,w : {h,w}
            img_shape : {img.shape}
            value : {img[0,i,0],img[i,0,0]}

        """)
    return pred, img, mask

def test_dct(model,device,index,mode,img_size,dir_img,dir_mask):
    model.eval()
    testdata = DefactoDataset(dir_img,
               dir_mask,
               12000,
               img_size,
               mode,None)

    jpg_artifact,mask,qtable =  testdata.__getitem__(index)
    img = jpg_artifact[:3]

    inp = jpg_artifact.unsqueeze(0).to(device)
    qtable = qtable.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp,qtable) # normalize none

    img = img.permute(1,2,0)
    img = (img*125.5+125.5)/255.0
    h,w=512,512
    for i in range(len(img)-1,0,-1):
        if img[0,i,0]!=125.5/255.0:
            w = i  
            break
    for i in range(len(img)-1,0,-1):
        if img[i,0,0]!=125.5/255.0:
            h = i  
            break
    # print(w,h)
    img = img[:h,:w,:]

    return pred, img, mask

def load_dataset_for_casia(total_nums,img_size,batch_size,dir_img,dir_mask):
    # ImageNet 표준으로 정규화 <- 일반적인 관행

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                img_size,
                'train',None)

    # dataset_size = len(dataset)
    train_size =  2300 #int(dataset_size * 0.8)
    validation_size = 200 #int(dataset_size * 0.2)
    test_size = 559
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size,test_size])
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())
    print("test images len : ",test_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
    
    return train_dataloader,val_dataloader,test_dataloader


# tif to jpg in tif dir
if __name__ == '__main__':
    tif_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Tp"
    out_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Tp_jpg"
    mask_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Groundtruth"
    mask_out_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Groundtruth_jpg"
    os.makedirs(out_dir,exist_ok=True)
    os.makedirs(mask_out_dir,exist_ok=True)

    source = os.listdir(tif_dir)
    l = len(source)
    # print('total iamges : ',l)

    # for i,infile in enumerate(source,1):
    #     try:
    #         if infile[-3:] == "tif":
    #             outfile = infile[:-3] + "jpg"
    #             im = Image.open(os.path.join(tif_dir,infile))
    #             out = im.convert("RGB")
    #             out.save(os.path.join(out_dir,outfile), "JPEG", quality=100)
    #         if  i %100 ==0: 
    #             print(f"{i} images processed. {i/l:.3}  done. ")
    #     except:
    #         print(i)
       

    print(len(os.listdir(tif_dir)))
    print(len(os.listdir(mask_dir)))
    """
    10765
    10765
    """
    images = sorted(os.listdir(tif_dir))
    
    maskes = sorted(os.listdir(mask_dir))

    for i,data in enumerate(zip(images,maskes),1):
        infile,mask = data
        try:
            if infile[-3:] == "tif":
                # image
                outfile = infile[:-3] + "jpg"
                im = Image.open(os.path.join(tif_dir,infile))
                out = im.convert("RGB")
                out.save(os.path.join(out_dir,outfile), "JPEG", quality=100)

                # mask
                outfile = mask[:-3] + "jpg"
                im = Image.open(os.path.join(mask_dir,mask))
                out = im.convert("L")
                out.save(os.path.join(mask_out_dir,outfile), "JPEG", quality=100)
            if  i %100 ==0: 
                print(f"{i} images processed. {i/l:.3}  done. ")
        except:
            print(i)

    mask_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\Groundtruth_jpg"
    out_dir = r"C:\Users\zxcas\PythonWork\DATASETS\CASIA2.0\TP_jpg"

    testdata = DefactoDataset(out_dir,
            mask_dir,
            -1,
            (512,512),
            "tr0in",None)

    def match_test():
        name = testdata.name
        label = testdata.label
        for data in zip(name,label):
            n,l = data   
            if not (n[:-4] in l):
                print("doesn't match pare")
                print(n)
                print(l)
                return 0
        return 1
    print(match_test())
    # def mask_move(isMatch:bool,mask_dir,new_mask_path):
    #     os.makedirs( new_mask_path,exist_ok=True)
    #     for img_name in os.listdir(out_dir):
    #         sh.move(os.path.join(mask_dir,img_name),
    #         os.path.join(new_mask_path,img_name))


    # # mask_move(-1,mask_dir,new_mask_path)
    # with open("./CASIA_list.txt",'r') as f:
    #     text = f.read()
    # text = text.replace(',png','')
    # with open("./CASIA_list2.txt",'w') as f:
    #     f.write(text)
