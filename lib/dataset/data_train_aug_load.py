from __future__ import print_function, absolute_import
import torch.utils.data as data
from torchvision import transforms
import os
import numpy as np
import cv2
from PIL import Image
import sys
from .aug.new_Local import new_local
from .aug.Information_extraction import information_extraction
import multiprocessing


def get_train_data_list(config):

    # 初始化一个list
    list_train_orig_data = []

    # 需要用到的参数
    mode = 'train'
    root = config.DATASET.ROOT
    txt_file = config.DATASET.JSON_FILE['train']
    img_path = os.path.join(root, mode)

    # 从txt文件中拿图片名
    with open(txt_file, 'r', encoding='utf-8') as file:
        labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]

    len_train_data = len(labels)
    print("load {} images original training set size!".format(len_train_data))

    # 把list先初始化好
    for i in range(len_train_data):
        list_train_orig_data.append([])
        img_name = list(labels[i].keys())[0]
        img = cv2.imread(os.path.join(img_path, img_name))
        list_train_orig_data[i].append(img)
        list_train_orig_data[i].append(i)

    return list_train_orig_data, labels


# 写载入训练集的类
class _DATA(data.Dataset):
    def __init__(self, list_train_orig_data, labels):
        self.toTensor = transforms.ToTensor()
        self.list_train_orig_data = list_train_orig_data
        self.labels = labels

    def __len__(self):
        return len(self.list_train_orig_data)

    def __getitem__(self, idex):
        img = self.list_train_orig_data[idex][0]
        idx = self.list_train_orig_data[idex][1]

        # 先注释掉这个 这个不能注释，否则val_data就不对了
        h, w = img.shape[0:2]
        matrix = self.process(img, w, h)
        image = self.toTensor(matrix)

        # 返回图片和序号，可以改序号，下面注释的语句是可以通过测试的
        return image, idx

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


def get_train_data(config):

    return _DATA


# 增广数据并且改好数据结构
def bezier_aug(original_train_dataset_list, orig_labels, list_policy):

    list_final_aug_dataset = []   # 格式也是要  [ [array1, idx1], [array2, idx2], ...... ]

    # 先把原始数据打进去
    for g in range(len(orig_labels)):
        list_final_aug_dataset.append(original_train_dataset_list[g])

    print("begin augment\n")

    # 第一个多进程，先算信息
    print("begin get information 1/2\n")
    list_information = []
    para_list = []  # 送进多进程的参数
    for i in range(len(orig_labels)):
        src = original_train_dataset_list[i][0]
        idx = original_train_dataset_list[i][1]
        para_list.append([src, idx])
    # 多进程跑new_local，增广样本
    with multiprocessing.Pool(processes=40) as pool:
        list_information = pool.map(information_extraction, para_list)

    # 第二个多进程，增广数据
    print("begin draw the image 2/2\n")
    num_aug = len(list_policy)
    for k in range(num_aug):
        # 要加一个增广的进度,可视化
        sys.stdout.write('\r>> aug time {:d}/{:d}'.format(k+1, num_aug))
        sys.stdout.flush()
        para_list = []  # 送进多进程的参数
        list_aug = []  # 缓存
        for i in range(len(orig_labels)):
            inf = list_information[i][0]
            idx = list_information[i][1]
            para_list.append([inf, 1, 3, list_policy[k], idx])  # 图片；增广次数；笔画半径；控制域比率;idx
        # 多进程跑new_local，增广样本
        with multiprocessing.Pool(processes=40) as pool:
            list_aug = pool.map(new_local, para_list)
        # 增广的样本打进list_final_aug_dataset
        for p in range(len(list_aug)):
            list_final_aug_dataset.append([list_aug[p][0][0], list_aug[p][1]])

    return list_final_aug_dataset, orig_labels


# # 增广数据并且改好数据结构
# def bezier_aug(original_train_dataset_list, orig_labels, num_aug, list_policy):
#
#     list_final_aug_dataset = []   # 格式也是要  [ [array1, idx1], [array2, idx2], ...... ]
#
#     # 先把原始数据打进去
#     for g in range(len(orig_labels)):
#         list_final_aug_dataset.append(original_train_dataset_list[g])
#
#     # 主循环
#     for k in range(num_aug):
#         # 要加一个增广的进度,可视化
#         sys.stdout.write('\r>> aug time {:d}/{:d}'.format(k+1, num_aug))
#         sys.stdout.flush()
#         para_list = []  # 送进多进程的参数
#         list_aug = []  # 缓存
#
#         for i in range(len(orig_labels)):
#             # 参数
#             src = original_train_dataset_list[i][0]
#             idx = original_train_dataset_list[i][1]
#             para_list.append([src, 1, 3, list_policy[k], idx])  # 图片；增广次数；笔画半径；控制域比率;idx
#
#         # 多进程跑new_local，增广样本
#         with multiprocessing.Pool(processes=40) as pool:
#             list_aug = pool.map(new_local, para_list)
#
#         # 增广的样本打进list_final_aug_dataset
#         for p in range(len(list_aug)):
#             list_final_aug_dataset.append([list_aug[p][0][0], list_aug[p][1]])
#
#     return list_final_aug_dataset, orig_labels




