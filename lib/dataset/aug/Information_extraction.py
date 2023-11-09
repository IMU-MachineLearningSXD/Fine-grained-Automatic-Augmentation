import cv2 as cv
import numpy as np
import math
import random
import copy


def information_extraction(para_list):

    src = para_list[0]
    idx = para_list[1]

    """
    [[list_cor], [list_ske], [list_bezier_use_information]]
     /////[list_bezier_use_information] = [[pt_cor1, pt_cor2],[pt_third_bezier],[俩角点控制域],[第三控制点控制域]]
    """

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

    # 骨架索引
    def skeleton_index(skeleton_src):
        index = []
        row, col = skeleton_src.shape
        for a in range(0, row):
            for b in range(0, col):
                if skeleton_src[a][b] == 255:
                    index.append((b, a))

        return index

    # 求多分支点和端点
    def corner_index(list_ske):
        list_corner = []
        for point in list_ske:
            num_branches = 0
            for a in range(-1, 2):
                for b in range(-1, 2):
                    if(point[0] + a, point[1] + b) in list_ske:
                        num_branches = num_branches + 1
            if num_branches != 3:
                list_corner.append(point)

        return list_corner

    # 去掉骨架上所有角点,只留下骨架
    def skeleton_clean(list_ske, list_corner):
        for pt in list_corner:
            list_ske.remove(pt)

        return list_ske

    # 角点去冗余
    def list_corner_clean(image_original, list_corner):
        # 先做角点的连通域检测
        image_original = image_original.astype(np.uint8) * 255
        image_original_copy = np.zeros_like(image_original)
        for pt in list_corner:
            cv.circle(image_original_copy, pt, 0, (255, 255, 255), -1)

        # 连通域检测
        nums_label, labels = cv.connectedComponents(image_original_copy)

        # 连通域的点分类打成list
        list_total = []
        # 先确定要打多少个子list
        for a in range(0, nums_label - 1):
            list_total.append([])
        # 遍历整个labels
        row, col = labels.shape
        for a in range(0, row):
            for b in range(0, col):
                if labels[a][b] > 0:
                    list_total[(labels[a][b] - 1)].append((b, a))

        # 接下来直接选出各个子连通域的中心角点
        list_corner_new = []
        for list in list_total:
            if len(list) == 1:
                list_corner_new.append(list[0])
            else:
                list_sort_a = []
                list_sort_b = []
                list_distance = []
                for pt in list:
                    list_sort_a.append(pt[0])
                    list_sort_b.append(pt[1])
                list_sort_a.sort()
                list_sort_b.sort()
                mid_a = (list_sort_a[0] + list_sort_a[-1]) / 2
                mid_b = (list_sort_b[0] + list_sort_b[-1]) / 2
                for pt in list:
                    list_distance.append(two_points_distance((mid_a, mid_b), pt))
                list_corner_new.append(list[list_distance.index(min(list_distance))])

        return list_corner_new

    # 骨架连通域检测
    def skeleton_connected(image_original, ske_index):

        # 画去冗余后的骨架图
        image_original = image_original.astype(np.uint8) * 255
        image_original_copy = np.zeros_like(image_original)
        for pt in ske_index:
            cv.circle(image_original_copy, pt, 0, (255, 255, 255), 0)

        # 连通域检测
        nums_label, labels = cv.connectedComponents(image_original_copy)

        # 连通域的点分类打成list
        list_total = []
        # 先确定要打多少个子list
        for a in range(0, nums_label-1):
            list_total.append([])
        # 遍历整个labels
        row, col = labels.shape
        for a in range(0, row):
            for b in range(0, col):
                if labels[a][b] > 0:
                    list_total[(labels[a][b]-1)].append((b, a))

        # for a in list_total:
        #     print(a)
        # exit(0)

        return list_total

    # 骨架角点匹配
    def ske_cor_match(list_ske, list_cor):
        """
        [[end_pt1, end_pt2], [cor_pt1, cor_pt2], [list_ske_pt]]
        """
        list_total = []
        # 是有可能还剩余孤立角点没有骨架匹配的，因为孤立角点没有骨架能匹配
        for list in list_ske:
            list_integration = []
            list_endpoint = []
            list_corner = []
            # 骨架数量大于等于2的情况
            if len(list) >= 2:
                # 先数分支数
                for pt in list:
                    count_branches = 0
                    # 8邻域
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            if (pt[0] + a, pt[1] + b) in list:
                                count_branches = count_branches + 1
                    # 此时端点一定只有两个
                    if count_branches == 2:
                        list_endpoint.append(pt)
                list_integration.append(list_endpoint)
                # 找到端点后，匹配距离最近的角点
                for pt_end in list_endpoint:
                    # 计算该端点到所有角点的距离
                    list_distance = []
                    for pt_corner in list_cor:
                        list_distance.append(two_points_distance(pt_end, pt_corner))
                    # 取距离最小的角点
                    point_corner = list_cor[list_distance.index(min(list_distance))]
                    list_corner.append(point_corner)
                list_integration.append(list_corner)
                list_integration.append(list)
                list_total.append(list_integration)
            # 骨架数量等于1的情况
            if len(list) == 1:
                # 此时只有一个骨架点，找两个最近的角点匹配
                pt = list[0]
                list_endpoint.append(pt)
                list_endpoint.append(pt)
                list_integration.append(list_endpoint)
                list_distance = []
                for pt_corner in list_cor:
                    list_distance.append(two_points_distance(pt, pt_corner))
                list_copy_distance = copy.copy(list_distance)
                list_copy_distance.sort()
                d_min, d_second_min = list_copy_distance[0], list_copy_distance[1]
                pt_cor1, pt_cor2 = list_cor[list_distance.index(d_min)], list_cor[list_distance.index(d_second_min)]
                list_corner.append(pt_cor1)
                list_corner.append(pt_cor2)
                list_integration.append(list_corner)
                list_integration.append(list)
                list_total.append(list_integration)
        # 考虑有孤立角点存在的情况
        list_cor_wait_clean = []
        for list in list_total:
            for pt in list[1]:
                list_cor_wait_clean.append(pt)
        list_cor_clean = set(list_cor_wait_clean)
        for pt in list_cor_clean:
            list_cor.remove(pt)
        if len(list_cor) != 0:
            for pt in list_cor:
                list_integration = []
                list_endpoint = []
                list_corner = []
                list_endpoint.append(pt)
                list_endpoint.append(pt)
                list_integration.append(list_endpoint)
                # 骨架两个端点和两个角点是一样的
                list_integration.append(list_endpoint)
                list_corner.append(pt)
                list_integration.append(list_corner)
                list_total.append(list_integration)
        # 圆形笔画没有角点的情况
        for a in list_total:
            if len(a[0]) == 0:
                # 太少骨架的直接不要了
                if len(a[2]) <= 2:
                    list_total.remove(a)
                # 正常大小的圆圈：
                else:
                    # 指定角点
                    pt_cor_circle = a[2][0]
                    list_cor = [pt_cor_circle, pt_cor_circle]
                    # 骨架中删除角点
                    a[2].remove(pt_cor_circle)
                    # 循环骨架，找到相邻的两个end_pt
                    list_endpoint = []
                    x, y = pt_cor_circle[0], pt_cor_circle[1]
                    # 8邻域
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if (x + i, y + j) in a[2]:
                                list_endpoint.append((x + i, y + j))
                    if len(list_endpoint) != 2:
                        list_total.remove(a)
                        continue
                    a[0].extend(list_endpoint)
                    a[1].extend(list_cor)

        return list_total

    # 骨架顺序重排
    def ske_rearrangement(list_total):
        """
         [[cor1_1,cor1_2],[ske_list1]]
        """
        for a in list_total:
            point_temp = a[0][0]
            list_ske_new = []
            while point_temp != a[0][1]:
                list_ske_new.append(point_temp)
                a[2].remove(point_temp)
                list_temp = []
                # 查这个点的8领域
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if (point_temp[0]+x, point_temp[1]+y) in a[2]:
                            list_temp.append((point_temp[0]+x, point_temp[1]+y))
                point_temp = list_temp[0]
            list_ske_new.append(a[0][1])
            # 重构原来的list_from_ske_cor_match
            del(a[2])
            del(a[0])
            a.append(list_ske_new)

        return list_total

    # 角点增补
    def cor_addition(list_total):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[list_cor], [list_ske]]
        """
        # 按照骨架list来循环
        for index_list_total in range(len(list_total)):

            # 数据准备
            list_bezier_information, list_bezier_cor, list_bezier_ske, list_integration = [], [], [], []
            child_list_total = list_total[index_list_total]
            len_ske = len(child_list_total[1])
            # 少数量骨架直接不处理
            if len_ske <= 4:
                list_no_addition = copy.copy(child_list_total)
                list_bezier_information.append(list_no_addition)
                list_total[index_list_total].append(list_bezier_information)
                continue  # 下面的就不执行了，直接走下一次主循环

            # 正常骨架的情况
            list_cor = child_list_total[0]
            list_ske = child_list_total[1]
            # 先给list_bezier_cor打进第一个角点
            list_bezier_cor.append(list_cor[0])
            flag_first_pt = 1  # 设置初始flag，判断从第一个点开始
            flag_need_new_direction = 1
            pt_cor = list_cor[0]  # 定义角点 暂定是第一个角点
            flag_judge_s_change = 0  # 定义判定s型骨架是否改变的flag
            target_arc_x, target_arc_y = 0, 0  # 初始化圆弧形骨架方向
            flag_judge_integration = 0  # 初始化是否整合的判定标签
            list_judge_three_direction = []  # 初始化三个合理方向的列表
            x, y = 0, 0  # 初始化方向
            index_ske = 0

            # 从骨架开始循环
            for index_ske in range(len_ske-3):
                # 先判断需不需要定义新的方向
                if flag_need_new_direction == 1:  # 表示需要一个新的方向
                    flag_need_new_direction = 0  # 重置direction 不需要新方向了

                    # 判断是不是第一个角点
                    if flag_first_pt == 1:  # 表示是第一个角点
                        flag_first_pt = 0  # 重置 不再是第一个点了 只会出现一次
                        pt_cor = list_cor[0]
                        # 判断角点和骨架点是否相邻
                        flag_judge_near = 0  # 初始化near的flag
                        for a in range(-1, 2):
                            for b in range(-1, 2):
                                if list_ske[index_ske][0] + a == pt_cor[0] and list_ske[index_ske][1] + b == pt_cor[1]:
                                    flag_judge_near = 1  # 表示角点和骨架相邻
                                    x, y = list_ske[0][0] - pt_cor[0], list_ske[0][1] - pt_cor[1]
                        if flag_judge_near == 0:  # 表示角点和骨架不相邻，直接取第一和第二骨架的方向
                            x, y = list_ske[1][0] - list_ske[0][0], list_ske[1][1] - list_ske[0][1]
                    # 表示不是第一个角点，直接取角点和骨架的方向
                    else:
                        x, y = list_ske[index_ske][0] - pt_cor[0], list_ske[index_ske][1] - pt_cor[1]

                    # 定义了新的方向，那么判定目标方向也要做更改
                    target_arc_x, target_arc_y = -x, -y
                    # 设定好哪三个方向是符合flag_judge_s_change的,一共有8种情况
                    if (x, y) == (-1, -1):
                        list_judge_three_direction = [(-1, -1), (-1, 0), (0, -1)]
                    elif (x, y) == (-1, 0):
                        list_judge_three_direction = [(-1, 0), (-1, 1), (-1, -1)]
                    elif (x, y) == (-1, 1):
                        list_judge_three_direction = [(-1, 1), (-1, 0), (0, 1)]
                    elif (x, y) == (0, 1):
                        list_judge_three_direction = [(0, 1), (-1, 1), (1, 1)]
                    elif (x, y) == (1, 1):
                        list_judge_three_direction = [(1, 1), (0, 1), (1, 0)]
                    elif (x, y) == (1, 0):
                        list_judge_three_direction = [(1, 0), (1, 1), (1, -1)]
                    elif (x, y) == (1, -1):
                        list_judge_three_direction = [(1, -1), (1, 0), (0, -1)]
                    elif (x, y) == (0, -1):
                        list_judge_three_direction = [(0, -1), (-1, -1), (1, -1)]

                # 判定完是否需要新方向后，就来到这里（不管判定是否成功都要走这一段）
                # 拿此次循环的前后骨架方向用于判定
                direction_ske_x = list_ske[index_ske+1][0] - list_ske[index_ske][0]
                direction_ske_y = list_ske[index_ske+1][1] - list_ske[index_ske][1]

                # 判定是否有符合S型、圆弧形笔画，以此改变整合标签
                # 判定是否有符合圆弧形骨架的情况
                if target_arc_x == direction_ske_x and target_arc_y == direction_ske_y:
                    flag_judge_integration = 1
                # 判定是否有符合S型骨架的情况
                elif flag_judge_s_change == 1:
                    if direction_ske_x == x and direction_ske_y == y:
                        flag_judge_integration = 1
                elif flag_judge_s_change == 0:
                    if (direction_ske_x, direction_ske_y) not in list_judge_three_direction:
                        flag_judge_s_change = 1

                # 以整合标签判定是否整合
                if flag_judge_integration == 0:  # 表示不整合 但是需要把这个点打进新的骨架里
                    list_bezier_ske.append(list_ske[index_ske])
                elif flag_judge_integration == 1:  # 表示要整合,不要忘记初始化
                    # 整合
                    list_bezier_cor.append(list_ske[index_ske])
                    list_integration.append(list_bezier_cor), list_integration.append(list_bezier_ske)
                    list_bezier_information.append(list_integration)
                    # 初始化
                    list_bezier_cor, list_bezier_ske, list_integration = [], [], []
                    pt_cor = list_ske[index_ske]  # 角点需要重新定义
                    list_bezier_cor.append(pt_cor)
                    flag_judge_s_change, flag_judge_integration, flag_need_new_direction = 0, 0, 1

            # 循环走完了，但是还缺最后一次整合，因为还剩下几个骨架点
            list_bezier_cor.append(list_cor[1])
            list_bezier_ske.extend(list_ske[index_ske + 1:])
            list_integration.append(list_bezier_cor), list_integration.append(list_bezier_ske)
            list_bezier_information.append(list_integration)
            list_total[index_list_total].append(list_bezier_information)

        return list_total

    # 找骨架最突出点和第三贝塞尔曲线点
    def third_bezier_curve_point(list_total):
        """
        [[list_cor], [list_ske], [list_bezier_use_information]]
            ///////[list_bezier_use_information] = [[pt_cor1, pt_cor2], [pt_third_bezier]]
        """

        for index_list_total in range(len(list_total)):
            list_bezier_information = list_total[index_list_total][2]
            for index_child_bezier in range(len(list_bezier_information)):
                list_child_bezier = list_bezier_information[index_child_bezier]
                list_ske, list_distance_ske = list_child_bezier[1], []
                pt_cor1, pt_cor2 = list_child_bezier[0][0], list_child_bezier[0][1]
                # 找骨架的最突出点
                for pt_ske in list_ske:
                    list_distance_ske.append(get_distance_from_point_to_line(pt_ske, pt_cor1, pt_cor2))
                pt_bulge = list_ske[list_distance_ske.index(max(list_distance_ske))]  # 最突出点

                # 骨架现在就没用了，删掉
                del(list_total[index_list_total][2][index_child_bezier][1])
                # 第三贝塞尔控制点
                pt_center_x = 0.5 * float(pt_cor1[0]) + 0.5 * float(pt_cor2[0])
                pt_center_y = 0.5 * float(pt_cor1[1]) + 0.5 * float(pt_cor2[1])
                pt_third_bezier_x = int((float(pt_bulge[0]) - pt_center_x) * 2 + pt_center_x)
                pt_third_bezier_y = int((float(pt_bulge[1]) - pt_center_y) * 2 + pt_center_y)
                pt_third_bezier = [(pt_third_bezier_x, pt_third_bezier_y)]  # 第三控制点

                # 打进数据
                list_total[index_list_total][2][index_child_bezier].append(pt_third_bezier)

        return list_total

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
                    control_field = float(min_distance * rate)
                    # 数据打进list_control_field_corner
                    list_control_field_corner.append([control_field])

                # 计算第三控制点控制域
                len_rectangular = float(get_distance_from_point_to_line(pt_third_bezier, pt_cor1, pt_cor2))
                list_control_field_third_bezier.append([float(len_rectangular * rate)])

                # 打数据
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_corner)
                list_total[index_list_total][2][index_child_bezier].append(list_control_field_third_bezier)

        return list_total

    # 画控制域用的
    def draw_temp(list_total):

        # 数据准备
        list_pt, list_judge_x, list_judge_y = [], [], []
        # 先确定字体长宽
        for a in list_total:
            b = a[1]
            for pt in b:
                list_pt.append(pt)
        for a in list_pt:
            list_judge_x.append(a[0]), list_judge_y.append(a[1])
        x_min, x_max = min(list_judge_x), max(list_judge_x)
        y_min, y_max = min(list_judge_y), max(list_judge_y)
        x_len, y_len = x_max - x_min, y_max - y_min

        # 确定底片的长宽
        k1, k2 = 2.5, 2.5
        width, height = int(x_len * k1), int(y_len * k2)

        # 先调整格式
        skeleton_image = np.zeros(shape=(height, width))
        skeleton_image = 255 - skeleton_image
        skeleton_image = np.expand_dims(skeleton_image, axis=-1)
        skeleton_image = np.concatenate((skeleton_image, skeleton_image, skeleton_image), axis=-1)

        list_color = [(60, 230, 150), (230, 150, 150), (255, 80, 10), (60, 60, 60), (10, 70, 250),
                      (50, 255, 190)]

        # 点的位置需要调整到中心
        tiaozheng = int((width - x_len) / 2 - x_min), int((height - y_len) / 2 - y_min)

        # 画
        for k in list_total:
            # 画骨架
            # 随机选择一个颜色
            col = random.randint(0, 5)
            for pt in k[1]:
                x_new, y_new = pt[0] + tiaozheng[0], pt[1] + tiaozheng[1]
                pt = (x_new, y_new)
                cv.circle(skeleton_image, pt, 3, list_color[col], cv.FILLED)
            for g in k[2]:
                # 画角点
                for pt in g[0]:
                    x_new, y_new = pt[0] + tiaozheng[0], pt[1] + tiaozheng[1]
                    pt = (x_new, y_new)
                    cv.circle(skeleton_image, pt, 0, (0, 255, 0), cv.FILLED)
                # 画第三贝塞尔控制点
                pt_third_bezier = g[1][0]
                x_new, y_new = pt_third_bezier[0] + tiaozheng[0], pt_third_bezier[1] + tiaozheng[1]
                pt_third_bezier = (x_new, y_new)
                cv.circle(skeleton_image, pt_third_bezier, 0, (0, 255, 0), cv.FILLED)
                # 画控制域
                pt_cor1, pt_cor2 = g[0][0], g[0][1]
                x_new, y_new = pt_cor1[0] + tiaozheng[0], pt_cor1[1] + tiaozheng[1]
                pt_cor1 = (x_new, y_new)
                x_new, y_new = pt_cor2[0] + tiaozheng[0], pt_cor2[1] + tiaozheng[1]
                pt_cor2 = (x_new, y_new)
                list_control_field_corner,  list_control_field_third_bezier = g[2], g[3]

                # 画角点1控制域
                cv.circle(skeleton_image, pt_cor1, int(list_control_field_corner[0][0]), (255, 255, 0), 2)

                # 画角点2控制域
                cv.circle(skeleton_image, pt_cor2, int(list_control_field_corner[1][0]), (255, 255, 0), 2)

                # 画第三控制点控制域
                cv.circle(skeleton_image, pt_third_bezier, int(list_control_field_third_bezier[0][0]), (255, 255, 0), 2)

        return skeleton_image

    # 手写体转骨架
    img = src
    img = cv.blur(img, (3, 3))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = 255 - gray
    # gray = (gray < 127)
    ret, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)
    # skeleton_image = morphology.skeletonize(gray)
    skeleton_image = cv.ximgproc.thinning(gray, thinningType=cv.ximgproc.THINNING_ZHANGSUEN)

    # 计算骨架list
    list_skeleton = skeleton_index(skeleton_image)
    # 计算所有多分支点和端点
    list_corner_original = corner_index(list_skeleton)
    # 骨架去冗余
    list_skeleton = skeleton_clean(list_skeleton, list_corner_original)
    # 角点去冗余
    list_corner_new = list_corner_clean(skeleton_image, list_corner_original)
    # 骨架连通域检测
    list_skeleton = skeleton_connected(skeleton_image, list_skeleton)
    # 骨架角点匹配
    list_match = ske_cor_match(list_skeleton, list_corner_new)

    # 到这里，信息已经足够了，接下来是为了贝塞尔曲线变形而加入的补充信息
    ################################
    ################################

    # 骨架顺序重排
    list_match = ske_rearrangement(list_match)
    # 角点增补
    list_match = cor_addition(list_match)
    # 第三贝塞尔控制点
    list_match = third_bezier_curve_point(list_match)

    # 信息提取完成

    # 从policy中算控制域信息
    # list_final = []
    # for i in range(len(rate)):
    #     policy = rate[i]
    #     image_information = control_field(list_match, policy)
    #     list_final.append(image_information)

    # 控制域
    # image_information = control_field(list_match, rate)

    # 画出来看看
    # image_information = draw_temp(image_information)

    # return image_information
    return list_match, idx


if __name__ == '__main__':
    # for i in range(1, 101):
    #     img = cv.imread("./pending/"+str(i)+".jpg", cv.IMREAD_COLOR)
    #     image = information_extraction(img, 0.5, 0.4)
    #     cv.imwrite("./After processing/"+str(i)+".png", image)
    #     print("done:"+str(i)+"/100")

    img = cv.imread("./pending/1.jpg", cv.IMREAD_COLOR)
    image = information_extraction(img, 0.6)
    cv.imwrite("./After processing/" + str(1) + ".png", image)