import torch
from torchvision import transforms
from torch.utils.data import Dataset
import json
from PIL import Image
import random




class NYUDataset_Train(Dataset):

    def __init__(self,file_path,target_height,target_width):
        """
        Initialization of NYUDataset Class.
        :param file_path: the path of the file denoting all the paths of training data
            The annotation json file is of following format:
            [{'rgb_path':'the path to the rgb image','depth_path':'the path to the depth map img'},{},{},{},.......]
        :param target_height: the height of tensor as input to model
        :param target_width: the width of tensor as input to model
        """

        self.target_height=target_height
        self.target_width=target_width

        self.raw_height=480
        self.raw_width=640

        f=open(file_path)
        self.data_path_list=json.load(f)

    def __len__(self):

        return len(self.data_path_list)


    def crop(self,rgb,depth):

        """
        randomly crop the rgb image and depth map then resize them to the target height and width
        :param rgb: the rgb image to crop,PIL image type
        :param depth: the depth map to crop,PIL image type
        :return: the cropped and resized rgb image and depth map
        """

        # choose the height and width to crop to
        height_list=[self.raw_height,450,420,390,360,330,300,270,240,self.target_height]
        width_list=[self.raw_width,600,560,520,480,440,400,360,320,self.target_width]

        crop_height=random.choice(height_list)
        crop_width=random.choice(width_list)

        # generate the top left corner coordinate
        top_left_corner_x=random.uniform(0,self.raw_width-crop_width)
        top_left_corner_y=random.uniform(0,self.raw_height-crop_height)

        # generate the right bottom corner coordinate
        bottom_right_corner_x=top_left_corner_x+crop_width
        bottom_right_corner_y=top_left_corner_y+crop_height

        rgb=rgb.crop((top_left_corner_x,top_left_corner_y,bottom_right_corner_x,bottom_right_corner_y))
        depth=depth.crop((top_left_corner_x,top_left_corner_y,bottom_right_corner_x,bottom_right_corner_y))

        rgb=rgb.resize((self.target_width,self.target_height),Image.BILINEAR)
        depth=depth.resize((self.target_width,self.target_height),Image.BILINEAR)

        # scale ratio
        ratio=crop_height/self.target_height

        return rgb,depth,ratio


    def rgb_norm(self,rgb):
        """
        transform PIL rgb image to pytorch tensor type,scale the rgb value from 0-255 to 0.0-1.0 and normalize with
        predefined mean and std
        :param rgb:rgb image,PIL image
        :return:corresponding tensor
        """

        rgb_mean=[0.485, 0.456, 0.406]
        rgb_std=[0.229, 0.224, 0.225]

        normalizer=transforms.Normalize(mean=rgb_mean,std=rgb_std)

        transformer=transforms.Compose([transforms.ToTensor(),
                                        normalizer])

        # transformer = transforms.Compose([transforms.ToTensor()])

        return transformer(rgb)

    def depth_norm(self,depth,ratio):
        """
        transform the PIL depth map to pytorch tensor type and scale the depth value to range of [0.0,1.0]
        :param depth: depth map,PIL image
        :param ratio: larger object correspond to close position.Use param ratio to scale the depth value to make it accord with the resized image.
        :return: corresponding tensor
        """

        # used to scale the depth value to the range[0.0,1.0].Modify it according to own processing method.
        depth_scale_factor=65535

        depth=transforms.ToTensor()(depth)

        depth=depth/depth_scale_factor*ratio

        return depth

    def __getitem__(self, item):

        rgb_path=self.data_path_list[item]['rgb_path']
        depth_path=self.data_path_list[item]['depth_path']

        # crop the blank areas
        rgb=Image.open(rgb_path).crop((8, 8, 632, 472))
        depth=Image.open(depth_path).crop((8, 8, 632, 472))  # 0-65535 value
        self.raw_height,self.raw_width=rgb.size

        # flip augmentation
        flip_flag=random.uniform(0,1)
        if flip_flag>0.5:
            rgb=rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth=depth.transpose(Image.FLIP_LEFT_RIGHT)

        # color augmentation,rgb image only
        color_transformer=transforms.ColorJitter(0.2,0.2,0.2,0.2)
        rgb=color_transformer(rgb)


        # random crop image and depth
        rgb,depth,ratio=self.crop(rgb,depth)

        # rotate augmentation
        # degree=random.uniform(-10,10)
        # rgb=rgb.rotate(degree,expand=False)
        # depth=depth.rotate(degree)

        # transform rgb image to tensor while performing scaling and normalization
        rgb_tensor=self.rgb_norm(rgb)

        # transform depth map to tensor while performing scaling
        depth_tensor=self.depth_norm(depth,ratio)


        return rgb_tensor,depth_tensor


class NYUDataset_TnV(Dataset):

    def __init__(self,file_path,target_height,target_width):
        """
        Initialization of NYUDataset Class.
        :param file_path: the path of the file denoting all the paths of training data
            The annotation json file is of following format:
            [{'rgb_path':'the path to the rgb image','depth_path':'the path to the depth map img'},{},{},{},.......]
        :param target_height: the height of tensor as input to model
        :param target_width: the width of tensor as input to model
        """

        self.target_height=target_height
        self.target_width=target_width

        self.raw_height = 480
        self.raw_width = 640

        f=open(file_path)
        self.data_path_list=json.load(f)

    def __len__(self):

        return len(self.data_path_list)


    def rgb_norm(self,rgb):
        """
        transform PIL rgb image to pytorch tensor type,scale the rgb value from 0-255 to 0.0-1.0 and normalize with
        predefined mean and std
        :param rgb:rgb image,PIL image
        :return:corresponding tensor
        """

        rgb_mean=[0.485, 0.456, 0.406]
        rgb_std=[0.229, 0.224, 0.225]

        normalizer=transforms.Normalize(mean=rgb_mean,std=rgb_std)

        transformer=transforms.Compose([transforms.ToTensor(),
                                        normalizer])

        return transformer(rgb)

    def depth_norm(self,depth,ratio):
        """
        transform the PIL depth map to pytorch tensor type and scale the depth value to range of [0.0,1.0]
        :param depth: depth map,PIL image
        :param ratio: larger object correspond to close position.Use param ratio to scale the depth value to make it accord with the resized image.
        :return: corresponding tensor
        """

        # used to scale the depth value to the range[0.0,1.0].Modify it according to own processing method.
        depth_scale_factor=65535

        depth=transforms.ToTensor()(depth)

        depth=depth/depth_scale_factor*ratio

        return depth

    def __getitem__(self, item):
        rgb_path = self.data_path_list[item]['rgb_path']
        depth_path = self.data_path_list[item]['depth_path']

        rgb = Image.open(rgb_path)
        depth = Image.open(depth_path)  # 0-65535 value

        rgb = rgb.resize((self.target_width, self.target_height), Image.BILINEAR)

        ratio = self.raw_height/self.target_height

        rgb_tensor=self.rgb_norm(rgb)
        depth_tensor=self.depth_norm(depth,ratio)

        return rgb_tensor,depth_tensor






saved_path="/Users/a1/workspace/DepthDataset/NYU/train_annotations.json"
dt=NYUDataset_TnV(saved_path,300,300)

rgb,depth=dt[20]

# rgb_img=transforms.ToPILImage()(rgb)
# depth_img=transforms.ToPILImage()(depth)
#
# rgb_img.show()

