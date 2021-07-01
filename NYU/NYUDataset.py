import torch
from torchvision import transforms
from torch.utils.data import Dataset
import json
from PIL import Image
import random




class NYUDataset(Dataset):

    def __init__(self,file_path,target_height,target_width):

        self.target_height=target_height
        self.target_width=target_width

        f=open(file_path)
        self.data_path_list=json.load(f)

    def __len__(self):

        return len(self.data_path_list)


    def crop(self,rgb,depth):

        height_list=[480,450,420,390,360,330,300,270,240,self.target_height]
        width_list=[640,600,560,520,480,440,400,360,320,self.target_width]

        crop_height=random.choice(height_list)
        crop_width=random.choice(width_list)

        top_left_corner_x=random.uniform(0,640-crop_width)
        top_left_corner_y=random.uniform(0,480-crop_height)

        bottom_right_corner_x=top_left_corner_x+crop_width
        bottom_right_corner_y=top_left_corner_y+crop_height

        rgb=rgb.crop((top_left_corner_x,top_left_corner_y,bottom_right_corner_x,bottom_right_corner_y))
        depth=depth.crop((top_left_corner_x,top_left_corner_y,bottom_right_corner_x,bottom_right_corner_y))

        ratio=crop_height/self.target_height
        depth=depth/ratio

        rgb=rgb.resize((self.target_width,self.target_height),Image.BILINEAR)
        depth=depth.resize((self.target_width,self.target_height),Image.BILINEAR)

        return rgb,depth


    def rgb_norm(self,rgb):
        rgb_mean=[0.485, 0.456, 0.406]
        rgb_std=[0.229, 0.224, 0.225]

        normalizer=transforms.Normalize(mean=rgb_mean,std=rgb_std)

        return normalizer(rgb)


    def __getitem__(self, item):

        rgb_path=self.data_path_list[item]['rgb_path']
        depth_path=self.data_path_list[item]['depth_path']

        rgb=Image.open(rgb_path)
        depth=Image.open(depth_path)

        # flip augmentation
        flip_flag=random.uniform(0,1)
        if flip_flag>0.5:
            rgb=rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth=depth.transpose(Image.FLIP_LEFT_RIGHT)

        # color augmentation,rgb image only
        color_transformer=transforms.ColorJitter(0.2,0.2,0.2,0.2)
        rgb=color_transformer(rgb)


        # random crop image and depth
        rgb,depth=self.crop(rgb,depth)

        # rotate augmentation
        degree=random.uniform(-10,10)
        rgb=rgb.rotate(degree)
        depth=depth.rotate(degree)






















        return 0





saved_path="/Users/a1/workspace/DepthDataset/NYU/train_annotations.json"
NYUDataset(saved_path,100,100)
