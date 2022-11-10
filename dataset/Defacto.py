# -*- coding: utf-8 -*-
"""
defacto-splicing Dataset utills 
데이터세트 정의 클래스, 데이터세트 불러오기 함수 구현
"""
import os,sys
sys.path.append(os.getcwd())
import torch
import pandas as pd
from abstractDataset import AbstractDataset
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms
from torch.utils.data import DataLoader,random_split
import PIL.Image as Image
import cv2 as cv

# https://www.kaggle.com/code/alerium/defacto-test
class DefactoDataset(AbstractDataset):
    def __init__(self, im_root_dir,label_root_dir, num, img_size, mode='test' , transform=None):
        self.im_root_dir = im_root_dir
        self.label_root_dir = label_root_dir
        self.transform = transform
        self.num = num
        self.mode = mode
        self.img_size = img_size
        self.df = pd.DataFrame({
            'image_path':[os.path.join(self.im_root_dir,_) for _ in sorted(os.listdir(self.im_root_dir))],
            'mask_path':[os.path.join(self.label_root_dir,_) for _ in sorted(os.listdir(self.label_root_dir))]
                         })
        name , label = self.prepare()

        self.name = name
        self.label = label

    def __len__(self):
        return len(self.name)

    ### for defacto : num 개수 만큼 데이터세트 잘라서 사용
    def prepare(self):
        return self.df['image_path'].iloc[:self.num], self.df['mask_path'].iloc[:self.num] 

    def _resize(self , sample):
        image, mask = sample[0], sample[1]
        n = self.img_size
        image = TF.resize(image , size=(n,n),interpolation=transforms.InterpolationMode.BICUBIC)
        mask = TF.resize(mask , size=(n,n) ,interpolation=transforms.InterpolationMode.BICUBIC)

        return image , mask
    
    def __getitem__(self,idx):
        return 0
        return self._create_tensor(tamp_path, mask)

    
    # def __getitem__(self, idx):
    #     image = torch.FloatTensor(cv.imread(f"{self.im_root_dir}/{self.name[idx][:-3]}tif"))
    #     label = torch.FloatTensor(cv.imread(f"{self.label_root_dir}/{self.name[idx][:-3]}jpg",cv.IMREAD_GRAYSCALE))

    #     if self.transform == None:
    #         self.transform = transforms.Compose([
    #             transforms.Resize((self.img_size,self.img_size),interpolation=transforms.InterpolationMode.BICUBIC),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    #         ])

    #     if self.mode in ['train','eval']:
    #         x,y  = self._resize((image.permute(2,0,1), label.unsqueeze(0))) # (C,W,H)로 마춰줘야 함
    #         x = self.transform(x)
    #         y = y.ge(0.5).float()
    #     else: # test : 네트워크에 통과되지 않고 바로 plot 가능한 image 반환
    #         x = image.permute(2,0,1)
    #         y = label.unsqueeze(0)/255.0
    #         x,y = self._resize((x/255.0,y))
    #         y = y.ge(0.5).float() # element-wise로 값을 비교해 크거나 같으면 True를, 작으면 False를 반환한다.
    #         y = y.permute(1,2,0)

    #     # 위조면 [0,1] 아니면 [1,0]
    #     # todo abstract dataset 상속으로 위조,정상 데이터 세트 정의 구별
    #     # label = torch.zeros((2,)).float()
    #     if True :# label == 'forgery'
    #         label = torch.tensor(1,dtype=torch.long)
    #     else:
    #         label[0] = 1.0

    #     return {'image': x, 'landmarks': y,'label':label}

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

# tif to jpg in tif dir
if __name__ == '__main__':
    tif_dir = r"E:\splicing_1_img\img"
    out_dir = r"E:\splicing_1_img\img_jpg"

    os.makedirs(out_dir,exist_ok=True)

    # for infile in os.listdir(tif_dir):
    #     if infile[-3:] == "tif":
    #         outfile = infile[:-3] + "jpg"
    #         im = Image.open(os.path.join(tif_dir,infile))
    #         out = im.convert("RGB")
    #         out.save(os.path.join(out_dir,outfile), "JPEG", quality=100)
    print(len(os.listdir(tif_dir)))
    print(len(os.listdir(out_dir)))
    """
    10765
    10765
    """

    mask_dir = r"E:\splicing_1_annotations\probe_mask"

    testdata = DefactoDataset(out_dir,
            mask_dir,
            2000,
            (512,512),
            "train",None)

    def match_test():
        name = testdata.name
        label = testdata.label
        for data in zip(name,label):
            n,l = data
            print(n in l)    
            if not (n in l):
                print("doesn't match pare")
                break
    #match_test()
    print(testdata.df.head())
