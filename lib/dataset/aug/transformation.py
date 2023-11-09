import numpy as np
import random
import cv2 as cv
import os
import math
from PIL import Image
import time


# 骨架索引点
def skeleton_index(skeleton_image):
    index = []
    row, col = skeleton_image.shape
    for a in range(0, row):
        for b in range(0, col):
            if skeleton_image[a][b] == 1:
                index.append((b, a))

    return index


# 求骨架的端点
def endpoint_index(list_ske):
    list_corner = []
    for point in list_ske:
        num_branches = 0
        for a in range(-1, 2):
            for b in range(-1, 2):
                if(point[0] + a, point[1] + b) in list_ske:
                    num_branches = num_branches + 1
        if num_branches == 1 or num_branches == 2:
            list_corner.append(point)

    return list_corner


# 计算两点距离
def two_points_distance(point1, point2):
    p1 = point1[0] - point2[0]
    p2 = point1[1] - point2[1]
    distance = math.hypot(p1, p2)

    return distance


# 角点归位
def pt_cor_back(list_endpoint, pt_cor):
    if not list_endpoint:
        return pt_cor
    else:
        list_distance = []
        for pt in list_endpoint:
            list_distance.append(two_points_distance(pt_cor, pt))
        return list_endpoint[list_distance.index(min(list_distance))]


# flag判定
def flag_judge(list_child_already):

    if not list_child_already[0][0]:
        if not list_child_already[0][1]:
            flag_situation_judge = 1  # 角点1空，角点2也空
        else:
            flag_situation_judge = 2  # 角点1空，角点2不空
    elif not list_child_already[0][1]:
        flag_situation_judge = 3  # 角点1不空，角点2空
    else:
        flag_situation_judge = 4  # 角点1不空，角点2也不空

    return flag_situation_judge


# 指定一个角点
def identify_reference_corner(pt_wait, list_reference):

    list_distance = []
    for list_child in list_reference:
        pt_compare = list_child[0]
        list_distance.append(two_points_distance(pt_wait, pt_compare))
    index_min = list_distance.index(min(list_distance))
    list_need = list_reference[index_min]

    return list_need


# 方法一：贝塞尔曲线变形
def bezier_transformation(list_already, list_information, flag_special):
    """
        list_information:
        [[list_cor], [list_ske], [list_bezier_use_information]]
            /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                    |
        [[pt_cor1, pt_cor2],[pt_third_bezier]]     # 这是list_total 需要的格式
                    |
        [[cor1, cor2],[ske]]     # 这是list_already 需要的格式
    """

    # 拿数据
    list_bezier_inf = list_information[2]  # 取整个信息中贝塞尔变形所用部分
    list_total = []
    for k in range(len(list_bezier_inf)):
        list_total.append([[[], []], []])
    # 根据list_already的不同情形先把list_total的角点打上去
    if flag_special == 2:  # 角点1空，角点2不空
        list_total[-1][0][1] = list_already[0][1]
    elif flag_special == 3:  # 角点1不空，角点2空
        list_total[0][0][0] = list_already[0][0]
    elif flag_special == 4:  # 角点1不空，角点2也不空
        list_total[-1][0][1] = list_already[0][1]
        list_total[0][0][0] = list_already[0][0]

    # 以list_total为主循环，算出数据并打入,算list_total
    for index_list_total in range(len(list_total)):
        list_total_cor = list_total[index_list_total][0]
        list_bezier = list_bezier_inf[index_list_total]
        # 先处理第一个角点
        # 定义方向
        theta = random.uniform(0, 2 * np.pi)
        # 定义R
        r_radius = list_bezier[2][0][0]
        if not list_total[index_list_total][0][0]:  # 表示角点1是空的,只有空的才处理
            # 表示角点2也是空的，两个都空,说明是整个图像的第一对儿,则直接以自己为基准确定落点
            if not list_total[index_list_total][0][1]:
                x_get, y_get = list_bezier[0][0][0], list_bezier[0][0][1]
                # 定义最终点
                x_already = x_get + r_radius * np.cos(theta)
                y_already = y_get + r_radius * np.sin(theta)
                pt_already = [x_already, y_already]
                list_total[index_list_total][0][0].extend(pt_already)
            # 表示角点2存在，所以要以角点2为基准点
            else:
                x_change = list_total_cor[1][0] - list_bezier[0][1][0]
                y_change = list_total_cor[1][1] - list_bezier[0][1][1]
                x_get, y_get = list_bezier[0][0][0] + x_change, list_bezier[0][0][1] + y_change
                # 定义最终点
                x_already = x_get + r_radius * np.cos(theta)
                y_already = y_get + r_radius * np.sin(theta)
                pt_already = [x_already, y_already]
                list_total[index_list_total][0][0].extend(pt_already)
        # 再处理第二个角点
        # 定义方向
        theta = random.uniform(0, 2 * np.pi)
        # 定义R
        r_radius = list_bezier[2][1][0]
        if not list_total[index_list_total][0][1]:  # 表示角点2是空的，只有空的才处理，此时，角点1一定存在，所以以角点1为基准点确定落点
            x_change = list_total_cor[0][0] - list_bezier[0][0][0]
            y_change = list_total_cor[0][1] - list_bezier[0][0][1]
            x_get, y_get = list_bezier[0][1][0] + x_change, list_bezier[0][1][1] + y_change
            x_already = x_get + r_radius * np.cos(theta)
            y_already = y_get + r_radius * np.sin(theta)
            pt_already = [x_already, y_already]
            list_total[index_list_total][0][1].extend(pt_already)
            # 如果还有下一对，就需要把这个点补到第一个点
            if index_list_total + 1 < len(list_total):
                list_total[index_list_total + 1][0][0].extend(pt_already)
        # 有了两个角点后，计算第三贝塞尔控制点
        # 定义方向
        theta = random.uniform(0, 2 * np.pi)
        # 定义R
        r_radius = list_bezier[3][0][0]

        pt_cor1_new, pt_cor2_new = list_total_cor[0], list_total_cor[1]
        pt_cor1_old, pt_cor2_old = list_bezier[0][0], list_bezier[0][1]
        x_corner1_change, y_corner1_change = pt_cor1_new[0] - float(pt_cor1_old[0]), pt_cor1_new[1] - float(pt_cor1_old[1])
        x_corner2_change, y_corner2_change = pt_cor2_new[0] - float(pt_cor2_old[0]), pt_cor2_new[1] - float(pt_cor2_old[1])
        x_change = float(0.5 * x_corner1_change) + float(0.5 * x_corner2_change)
        y_change = float(0.5 * y_corner1_change) + float(0.5 * y_corner2_change)
        x_get, y_get = list_bezier[1][0][0] + x_change, list_bezier[1][0][1] + y_change
        x_already = x_get + r_radius * np.cos(theta)
        y_already = y_get + r_radius * np.sin(theta)
        pt_already = [x_already, y_already]
        list_total[index_list_total][1].extend(pt_already)

    # list_total所有处理完之后，就把list_total[0][0][0] 和 list_total[-1][0][1] 替换 list_already[0]就行
    list_already[0] = [list_total[0][0][0],  list_total[-1][0][1]]
    # 然后再求list_already里的ske
    list_ske = []

    # 做list_already
    for list_total_child in list_total:
        pt_cor1, pt_cor2 = np.array(list_total_child[0][0]), np.array(list_total_child[0][1]),
        pt_bezier = np.array(list_total_child[1])
        p = lambda t: (1 - t) ** 2 * pt_cor1 + 2 * t * (1 - t) * pt_bezier + t ** 2 * pt_cor2
        points = np.array([p(t) for t in np.linspace(0, 1, 500)])
        x, y = points[:, 0].astype(np.int).tolist(), points[:, 1].astype(np.int).tolist()
        # 一个个点打进list_ske
        for t in range(len(x)):
            list_ske.append((x[t], y[t]))
    # 去一次重复的点
    list_ske = list(set(list_ske))
    # 数据打入list_already
    list_already[1].extend(list_ske)

    return list_already



