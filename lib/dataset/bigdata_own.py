from __future__ import print_function, absolute_import
import torch.utils.data as data
from torchvision import transforms
import os
import numpy as np
import cv2
from PIL import Image


class _OWN(data.Dataset):
    def __init__(self, config, is_train=True):

        self.root = config.DATASET.ROOT
        self.is_train = is_train
        mode = 'train' if self.is_train else 'test'
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W
        self.size = (self.inp_w, self.inp_h)

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        self.toTensor = transforms.ToTensor()

        # 这里是label的地址
        txt_file = config.DATASET.JSON_FILE['train'] if is_train else config.DATASET.JSON_FILE['val']
        # 这里是图片的地址
        self.img_path = os.path.join(self.root, mode)

        # convert name:indices to name:string
        with open(txt_file, 'r', encoding='utf-8') as file:
            self.labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

        # 这里输出载入多少张图片
        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = list(self.labels[idx].keys())[0]
        img = cv2.imread(os.path.join(self.img_path, img_name))
        # 这个不能注释，否则val_data就不对了
        h, w = img.shape[0:2]
        matrix = self.process(img, w, h)
        image = self.toTensor(matrix)

        # 返回图片和序号，可以改序号，下面注释的语句是可以通过测试的
        return image, idx
        # return image, 1

    # 用在process里面
    def frame_construct(self, image, width):
        h, w = image.shape[0:2]
        if width % h != 0:
            extend_width = (width // h + 1) * h
            block_num = (width // h + 1) * 2 - 1
        else:
            extend_width = (width // h) * h
            block_num = (width // h) * 2 - 1
        image_temp = cv2.copyMakeBorder(image, 0, 0, 0, extend_width - width, cv2.BORDER_CONSTANT, value=0)
        waiting_paint = np.zeros((h, h * block_num, 3))
        for i in range(block_num):
            waiting_paint[:, h * i:h * (i + 1), :] = image_temp[:, h // 2 * i:h // 2 * i + h, :]
        return waiting_paint.astype(np.uint8)

    # 重塑图片的尺寸到target_size
    def process(self, img, w, h, target_size=(160, 32)):
        image = Image.fromarray(self.frame_construct(img, w))

        if w / h < 280 / 32:
            image_temp = image.resize((int(32 / h * w), 32), Image.BILINEAR)
            w_temp, h_temp = image_temp.size
            image_temp = cv2.cvtColor(np.asarray(image_temp), cv2.COLOR_RGB2BGR)
            image_temp = cv2.copyMakeBorder(
                image_temp, 0, 0, 0, 280 - w_temp, cv2.BORDER_CONSTANT, value=0)
            res_image = Image.fromarray(cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB))
            res_image = res_image.resize(target_size, Image.BILINEAR)
            res_image = res_image.convert('L')
            numpy_image = np.expand_dims(np.asarray(res_image), axis=-1)
            return numpy_image
        if w / h > 280 / 32:
            image_temp = image.resize((280, int(280 / w * h)), Image.BILINEAR)
            w_temp, h_temp = image_temp.size
            image_temp = cv2.cvtColor(np.asarray(image_temp), cv2.COLOR_RGB2BGR)
            image_temp = cv2.copyMakeBorder(
                image_temp, 0, 32 - h_temp, 0, 0, cv2.BORDER_CONSTANT, value=0)
            res_image = Image.fromarray(cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB))
            res_image = res_image.resize(target_size, Image.BILINEAR)
            res_image = res_image.convert('L')
            numpy_image = np.expand_dims(np.asarray(res_image), axis=-1)
            return numpy_image
        if w / h == 280 / 32:
            image = image.resize(target_size, Image.BILINEAR)
            image = image.convert('L')
            numpy_image = np.expand_dims(np.asarray(image), axis=-1)
            return numpy_image








