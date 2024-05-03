import cv2 as cv
import numpy as np
import math
import random
import copy
import time
from .Information_extraction import information_extraction
from .transformation import flag_judge, identify_reference_corner
from .transformation import bezier_transformation


def new_local(para_list):

    src = para_list[0]
    times = para_list[1]
    stroke_radius = para_list[2]
    rate = para_list[3]
    idx = para_list[4]

    # 计算两点距离
    def two_points_distance(point1, point2):
        p1 = point1[0] - point2[0]
        p2 = point1[1] - point2[1]
        distance = math.hypot(p1, p2)

        return distance

    # 计算点到两点所组成直线距离
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        # 对于两点坐标为同一点时,返回点与点的距离
        if line_point1 == line_point2:
            point_array = np.array(point)
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array - point1_array)
        # 计算直线的三个参数
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        # 根据点到直线的距离公式计算距离
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
        return distance

    # 控制域计算
    def control_field(list_total, rate):  # （rate 表示半径的比率)
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
        """

        # 首先需要所有角点的集合
        list_cor_all = []
        for p in list_total:
            for q in p[2]:
                for pt in q[0]:
                    list_cor_all.append(pt)
        list_cor_all = list(set(list_cor_all))

        # 主循环开始
        for index_list_total in range(len(list_total)):
            list_bezier_information = list_total[index_list_total][2]
            for index_child_bezier in range(len(list_bezier_information)):
                list_child_bezier = list_bezier_information[index_child_bezier]
                list_cor, pt_third_bezier = list_child_bezier[0], list_child_bezier[1][0]
                pt_cor1, pt_cor2 = list_cor[0], list_cor[1]
                list_control_field_corner, list_control_field_third_bezier = [], []

                # 先计算角点控制域
                for pt_cor in list_cor:
                    list_temp = copy.copy(list_cor_all)  # copy所有角点的list
                    list_temp.remove(pt_cor)  # 在list中删除掉此刻遍历的角点
                    list_distance = []  # 创建一个用于缓存角点间距离的list
                    for pt_temp in list_temp:
                        list_distance.append(two_points_distance(pt_temp, pt_cor))
                    min_distance = min(list_distance)  # 取最小距离
                    # 控制域大小
                    rate_choice_cor = random.choice(rate)

                    control_field = float(min_distance * rate_choice_cor)
                    # 数据打进list_control_field_corner
                    list_control_field_corner.append([control_field])

                # 计算第三控制点控制域
                len_rectangular = float(get_distance_from_point_to_line(pt_third_bezier, pt_cor1, pt_cor2))
                rate_choice_third = random.choice(rate)
                list_control_field_third_bezier.append([float(len_rectangular * rate_choice_third)])

                # 打数据
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_corner)
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_third_bezier)

        return list_total

    def deformation(list_all):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
          /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
                    |
                    |
        [[cor1, cor2],[ske]]
        """

        # 首先生成需要的数据格式
        list_already, len_list_all, list_pt_reference = [], len(list_all), []
        for a in range(len_list_all):
            list_already.append([[[], []], []])

        # 变形主程序
        for index_list_already in range(len_list_all):
            list_child_already = list_already[index_list_already]
            list_child_all = list_all[index_list_already]
            # 情形判断：几种情况： 1: 1、2空； 2: 1空； 3: 2空； 4: 只有3空
            flag_situation_judge = flag_judge(list_child_already)

            # 每个笔画的变形选择
            if flag_situation_judge == 4:  # 特殊情况4,只能使用贝塞尔曲线变形
                list_already[index_list_already] = bezier_transformation(list_child_already,
                                                                         list_child_all, flag_situation_judge)
                continue  # 特殊情况就不需要后面的扫描过程了
            # 其他都属于正常情况，三种变形方式都能用

            else:
                # 如果是情况1且不是第一对儿角点，需要先指定一个,以最近点作为参考点
                if flag_situation_judge == 1 and index_list_already != 0:
                    flag_situation_judge = 3  # flag标志首先得变，变为flag = 3
                    list_reference = identify_reference_corner(list_all[index_list_already][0][0], list_pt_reference)
                    # index_temp = index_list_already - 1
                    index_temp1, index_temp2 = list_reference[2], list_reference[1]
                    # 定义方向
                    theta = random.uniform(0, 2 * np.pi)
                    # 定义R
                    r_radius = list_all[index_list_already][2][0][2][0][0]
                    x_change = list_already[index_temp1][0][index_temp2][0] - list_all[index_temp1][0][index_temp2][0]
                    y_change = list_already[index_temp1][0][index_temp2][1] - list_all[index_temp1][0][index_temp2][1]
                    x_get = list_all[index_list_already][0][0][0] + x_change
                    y_get = list_all[index_list_already][0][0][1] + y_change
                    x_already = x_get + r_radius * np.cos(theta)
                    y_already = y_get + r_radius * np.sin(theta)
                    list_child_already[0][0] = [x_already, y_already]
                    # list_child_already[0][0] = [x_get, y_get]

                # 贝塞尔曲线变形
                list_already[index_list_already] = bezier_transformation(
                    list_child_already, list_child_all, flag_situation_judge)

            # 扫描相同角点过程,一视同仁
            for index_already_pt in range(2):
                pt_standard = list_all[index_list_already][0][index_already_pt]
                pt_change = list_already[index_list_already][0][index_already_pt]
                list_pt_reference.append((pt_standard, index_already_pt, index_list_already))
                for index_scan in range(index_list_already + 1, len_list_all):
                    for index_scan_child in range(2):
                        if pt_standard == list_all[index_scan][0][index_scan_child]:
                            list_already[index_scan][0][index_scan_child] = pt_change

        return list_already

    def draw_src(list_all):

        # 数据准备
        list_pt, list_judge_x, list_judge_y = [], [], []
        # 先确定字体长宽
        for a in list_all:
            b = a[1]
            for pt in b:
                list_pt.append(pt)
        for a in list_pt:
            list_judge_x.append(a[0]), list_judge_y.append(a[1])
        x_min, x_max = min(list_judge_x), max(list_judge_x)
        y_min, y_max = min(list_judge_y), max(list_judge_y)
        x_len, y_len = x_max - x_min, y_max - y_min

        # 确定底片的长宽
        k1, k2 = 1.1, 1.1
        width, height = int(x_len * k1), int(y_len * k2)

        if width <= 10:
            width = 30
        if height <= 10:
            height = 30

        # 调底片格式
        image_film = np.zeros(shape=(height, width))
        image_film = np.expand_dims(image_film, axis=-1)
        image_film = np.concatenate((image_film, image_film, image_film), axis=-1)

        # 调整点的位置并画图
        size_blank = [int((width - x_len) / 2), int((height - y_len) / 2)]
        for a in list_pt:
            x_new, y_new = a[0] - x_min + size_blank[0], a[1] - y_min + size_blank[1]
            cv.circle(image_film, (x_new, y_new), stroke_radius, (255, 255, 255), -1)

        image_film = image_film.astype("uint8")

        return image_film

    # 控制域
    list_information = control_field(src, rate)

    # 变形和画图
    list_draw = []

    while len(list_draw) < times:    # times:1
        list_final = deformation(list_information)
        picture = draw_src(list_final)
        if picture.shape[2] != 3:
            continue
        list_draw.append(picture)

    return list_draw, idx


if __name__ == '__main__':
    # print(time.clock())
    img = cv.imread("./pending/1.jpg", cv.IMREAD_COLOR)
    list_augment = new_local(img, times=10, rate=0.4)
    cv.imwrite("./After processing/0.jpg", img)
    for i in range(len(list_augment)):
        cv.imwrite("./After processing/" + str(i + 1) + ".jpg", list_augment[i])
        print("done:" + str(i + 1) + "/10")

    # img = cv.imread("./pending/2.jpg", cv.IMREAD_COLOR)  # cv.imead默认就是cv.IMREAD_COLOR
    # img = cv.blur(img, (3, 3))
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # gray = 255 - gray
    # ret, img_bin = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    #
    # img_thinning = cv.ximgproc.thinning(img_bin, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)
    # print(img_thinning)
    # exit(0)
    # cv.imshow('1.jpg', img_thinning)
    # cv.waitKey(0)
    # exit(0)



