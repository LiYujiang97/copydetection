from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image
import torch
import PIL.ImageOps


class SiameseNetworkDataset(Dataset):

    def __init__(self, image_folder_dataset, transform=None, should_invert=False):
        self.imageFolderDataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        # 把图片根据文件夹（类）分到字典中，以类作为key，方便下面随机获取img0时指定获取每一类的第一张图
        image_dictionary = {}
        for img_tuple in image_folder_dataset.imgs:
            key = img_tuple[1]
            if key in image_dictionary.keys():
                image_dictionary[key].append(img_tuple)
            else:
                image_dictionary[key] = [img_tuple]
        self.image_dictionary = image_dictionary

    def __getitem__(self, index):
        # img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # 获取每一类中的第一张图（数据集中的第一张图为未作变换的完整原图）作为img0训练
        image_dictionary_num = len(self.image_dictionary)
        # python 自带的 randint 最大最小值都可取到，所以在此-1
        randint = random.randint(0, image_dictionary_num - 1)
        img0_tuple = self.image_dictionary[randint][0]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class TwoChannelNetworkDataset(Dataset):

    def __init__(self, image_folder_dataset, transform=None, should_invert=False):
        self.imageFolderDataset = image_folder_dataset
        self.transform = transform
        self.should_invert = should_invert
        # 把图片根据文件夹（类）分到字典中，以类作为key，方便下面随机获取img0时指定获取每一类的第一张图
        image_dictionary = {}
        for img_tuple in image_folder_dataset.imgs:
            key = img_tuple[1]
            if key in image_dictionary.keys():
                image_dictionary[key].append(img_tuple)
            else:
                image_dictionary[key] = [img_tuple]
        self.image_dictionary = image_dictionary

    def __getitem__(self, index):
        # img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # 获取每一类中的第一张图（数据集中的第一张图为未作变换的完整原图）作为img0训练
        image_dictionary_num = len(self.image_dictionary)
        # python 自带的 randint 最大最小值都可取到，所以在此-1
        randint = random.randint(0, image_dictionary_num - 1)
        img0_tuple = self.image_dictionary[randint][0]

        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        two_channel_image = torch.cat([img0, img1], dim=0)
        return two_channel_image, int(img1_tuple[1] != img0_tuple[1])

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
