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
import cv2 as cv

# https://www.kaggle.com/code/alerium/defacto-test
class DefactoDataset(torch.utils.data.Dataset):
    def __init__(self, im_root_dir,label_root_dir, num, img_size, mode='test' , transform=None):
        self.im_root_dir = im_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.num = num
        self.mode = mode
        self.img_size = img_size

        name , label = self.prepare(self.im_root_dir , self.label_root_dir, self.num)

        self.name = name
        self.label = label

    def __len__(self):
        return len(self.name)

    ### for defacto : num 개수 만큼 데이터세트 잘라서 사용
    def prepare(self,im_root_dir , lab_root_dir , num=2000):
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

    def _resize(self , sample):
        image, mask = sample[0], sample[1]
        n = self.img_size
        image = TF.resize(image , size=(n,n),interpolation=transforms.InterpolationMode.BICUBIC)
        mask = TF.resize(mask , size=(n,n) ,interpolation=transforms.InterpolationMode.BICUBIC)

        return image , mask
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(cv.imread(f"{self.im_root_dir}/{self.name[idx][:-3]}tif"))
        label = torch.FloatTensor(cv.imread(f"{self.label_root_dir}/{self.name[idx][:-3]}jpg",cv.IMREAD_GRAYSCALE))

        if self.transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size,self.img_size),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            ])

        if self.mode in ['train','eval']:
            x,y  = self._resize((image.permute(2,0,1), label.unsqueeze(0))) # (C,W,H)로 마춰줘야 함
            x = self.transform(x)
            y = y.ge(0.5).float()
        else: # test : 네트워크에 통과되지 않고 바로 plot 가능한 image 반환
            x = image.permute(2,0,1)
            y = label.unsqueeze(0)/255.0
            x,y = self._resize((x/255.0,y))
            y = y.ge(0.5).float() # element-wise로 값을 비교해 크거나 같으면 True를, 작으면 False를 반환한다.
            y = y.permute(1,2,0)
        return {'image': x, 'landmarks': y}

def load_dataset(total_nums,img_size,batch_size,dir_img,dir_mask):
    # ImageNet 표준으로 정규화 <- 일반적인 관행
    transformi = transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])

    dataset = DefactoDataset(dir_img,
                dir_mask,
                total_nums,
                img_size,
                'train',transformi)
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.2)

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print("train images len : ",train_dataset.__len__())
    print("validation images len : ",validation_dataset.__len__())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    
    return train_dataloader,val_dataloader

def test(model,device,index,mode,img_size,dir_img,dir_mask):
    model.eval()
    transformi = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
                ])
    testdata = DefactoDataset(dir_img,
               dir_mask,
               2000,
               img_size,
               mode,transformi)
    print(testdata.__len__())

    img = testdata.__getitem__(index)['image']
    mask = testdata.__getitem__(index)['landmarks']


    inp = torch.Tensor([img.numpy()]).to(device)
    print(inp.shape)
    with torch.no_grad():
        pred = model(transformi(inp)) # normalize none

    return pred, img, mask
