# -*- coding: utf-8 -*-
"""
defacto-splicing Dataset utills 
데이터세트 정의 클래스, 데이터세트 불러오기 함수 구현
"""
import os
import torch
import pandas as pd
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,random_split
import torchvision
import cv2 as cv
import matplotlib.pyplot as plt

# https://www.kaggle.com/code/alerium/defacto-test
class DefactoDataset(torch.utils.data.Dataset):
    def __init__(self, im_root_dir,label_root_dir, num, mode='test' , transform=None,tra = None):
        self.im_root_dir = im_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.tra = tra
        self.num = num
        self.mode = mode
        name = []
        lab = []
        name , label = self.prepare(self.im_root_dir , self.label_root_dir, self.num,self.mode)

        self.name = name
        self.label = label

    def __len__(self):
        return len(self.name)

    ###3 for defacto
    def prepare(self,im_root_dir , lab_root_dir , num=2000,mode='test'):
        name = []
        lab = []
        
        df = pd.DataFrame({'name':[_ for _ in os.listdir(im_root_dir)]})
        zipta = df['name'].tolist()[0:num]
        labels = sorted(os.listdir(lab_root_dir))
        
        for fname in zipta:
            lab_ = [a for a in labels if fname in a]
            lab.append(lab_)
        name = sorted(zipta)
        label = sorted(lab)
        
        return name , label 

    def transform0(self , sample, mode = 'train'):
        if mode in ['train','test']:
            image, mask = sample[0], sample[1]
            n = 256

            image = TF.resize(image , size=(n,n),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            mask = TF.resize(mask , size=(n,n) ,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else :
            image, mask = sample[0], sample[1]
            n = 256
            image = image.permute(2,0,1)
            image = TF.resize(image , size=(n,n),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            mask = TF.resize(mask , size=(n,n) ,interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            image = image.permute(1,2,0)
        return image , mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torch.FloatTensor(cv.imread(f"{self.im_root_dir}/{self.name[idx][:-3]}tif"))
        label = torch.FloatTensor(cv.imread(f"{self.label_root_dir}/{self.name[idx][:-3]}tif",cv.IMREAD_GRAYSCALE))
        if len(label.shape)>=3:
            label = torch.max(label , 2) 
            
        if self.mode =='train':
            # z = self.tra(image = np.array(image),mask = (np.array(label)))
            # x = torch.FloatTensor(z['image'])
            # y = torch.FloatTensor(z['mask'])
            x,y  = image, label
            x = x.permute(2,0,1)
            x = self.transform(x)
            y = y.unsqueeze(0)
            x,y = self.transform0((x/255.0,y))
            y = y.ge(0.5).float()
        elif self.mode == 'test':
            x = image.permute(2,0,1)
            x = self.transform(x)
            y = label.unsqueeze(0)/255.0
            x,y = self.transform0((x/255.0,y))
            y = y.ge(0.5).float()
        else:
            x = image
            y = label.unsqueeze(0)/255.0
            x,y = self.transform0((x/255.0,y),'none')
            y = y.ge(0.5).float()
            y = y.permute(1,2,0)
        
        return {'image': x, 'landmarks': y}

def load_dataset(total_nums,batch_size,dir_img,dir_mask):
    transformi = transforms.Compose([
                # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #              std=[0.229, 0.224, 0.225])
    ])
    transform = None

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                'train',transformi,transform)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
    print(train_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    
    return train_dataloader,val_dataloader

def test(model,device,index,mode,dir_img,dir_mask):
    model.eval()
    transformi = transforms.Compose([
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                ])
    transform = None
    testdata = DefactoDataset(dir_img,
               dir_mask,
               2000,
               mode,transformi,transform)
    
    absolute_path = False
    if absolute_path :
        image = torch.cuda.FloatTensor([cv.imread('./test_example.jpg')]) #0_000000103897.tif
        # image = torch.cuda.FloatTensor([cv.imread('./0_000000103897.tif')]) 

        x = image.permute(0,3,1,2)
        x = testdata.transform(x)
        x = TF.resize(x/255. , size=(256,256),interpolation=torchvision.transforms.InterpolationMode.BICUBIC)

    img = testdata.__getitem__(index)['image']
    mask = testdata.__getitem__(index)['landmarks']

    inp = torch.Tensor([img.numpy()]).to(device)
    print(inp.shape)
    with torch.no_grad():
        pred = model(inp.permute(0,3,1,2))#.permute(0,3,1,2))
    return pred, img, mask
# pred,ad = test(model_man,device)