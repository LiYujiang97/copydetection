import util
import cv2
import numpy
import random
import os
import shutil
import constant
from PIL import Image
import colorsys

def batch_generate_copy_images(dir_path, output_dir_path, sub_dir_num_limit=-1, per_copy_num=6,
                               remove_exist_output_dir=True, add_gaussian_noise=True, add_salt_noise=True,
                               add_crop=True, add_scale=True, add_rotate=False, random_use_attack=True):
    """
    自动生成一系列根据原图旋转缩放等生成的copy图像
    :param add_gaussian_noise:
    :param random_use_attack:
    :param add_crop:
    :param add_salt_noise:
    :param add_scale:
    :param add_rotate:
    :param remove_exist_output_dir: 是否删除原先存在的输出文件夹
    :param dir_path: 源图像父文件夹路径
    :param output_dir_path: copy图像输出文件夹路E径
    :param sub_dir_num_limit: 父文件夹下使用的系列0图片数量限制
    :param per_copy_num: 每一张原图生成的copy图数量
    """
    if remove_exist_output_dir and os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path, True)
    all_grouped_files_in_dir = util.get_all_grouped_file_path_in_dir(dir_path)
    # 限制遍历文件夹的数量
    grouped_file_len = len(all_grouped_files_in_dir)
    iterator_num = [grouped_file_len, min(sub_dir_num_limit, grouped_file_len)][sub_dir_num_limit > 0]
    # iterator_num = len(all_grouped_files_in_dir)
    for i in range(iterator_num):
        if i % 10 == 0:
            print("batch_generate_copy_images {} in {}".format(i, iterator_num))
        file_group_path = all_grouped_files_in_dir[i]
        if len(file_group_path) > 0:
            origin_file_path = file_group_path[0]
            pure_file_name = util.get_pure_file_name_from_path(origin_file_path)
            dir = output_dir_path + "/" + pure_file_name
            generate_copy_images(origin_file_path, per_copy_num, dir, add_salt_noise=add_salt_noise,add_gaussian_noise=add_gaussian_noise, add_crop=add_crop, add_scale=add_scale,
                                 add_rotate=add_rotate, random_use_attack=random_use_attack)


def generate_copy_images(origin_image_path, per_copy_num, output_dir_path=None, add_salt_noise=True,
                         add_gaussian_noise=True, add_crop=True, add_scale=True, add_rotate=True,color=True,
                         Histogtam= True,random_use_attack=True):
    if output_dir_path is None:
        output_dir_path = util.get_file_dir_path(origin_image_path)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    for j in range(per_copy_num):
        img = cv2.imread(origin_image_path, cv2.COLOR_BGR2GRAY)
        # img = cv2.imread(origin_image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(output_dir_path + "/" + util.get_file_name_from_path(origin_image_path), img)
        rand = 0
        if color:
            if random_use_attack:
                rand = random.randint(1, 10)
            if rand <= 3:
                img = modify_color(origin_image_path)
        # 椒盐噪声
        if add_salt_noise:
            if random_use_attack:
                rand = random.randint(1, 10)
            if rand <= 5:
                proportion = random.uniform(0, 0.05)
                img = salt_and_pepper_noise(img, proportion)
        # 高斯噪声
        if add_gaussian_noise:
            if random_use_attack:
                rand = random.randint(1, 10)
            if rand <= 5:
                sigma = random.uniform(0.05, 0.2)
                img = gaussian_noise(img, 0, sigma)
        if add_scale:
            if random_use_attack:
                rand = random.randint(1, 10)
            if rand <= 5:
                scale_ratio = random.uniform(0.5, 2)
                img, M = scale_image(img, scale_ratio, scale_ratio)
        if add_rotate:
            if random_use_attack:
                rand = random.randint(1, 10)
            if rand <= 5:
                rotate_angle = random.uniform(0, 360)
                img, M = rotate_bound(img, rotate_angle)
        if add_crop:
            if random_use_attack:
                rand = random.randint(1, 10)
            if rand <= 5:
                (h, w) = img.shape[:2]
                # 随即裁剪
                crop_ratio_range = [0.4, 1]
                new_width = int(random.uniform(crop_ratio_range[0], crop_ratio_range[1]) * w)
                new_height = int(random.uniform(crop_ratio_range[0], crop_ratio_range[1]) * h)
                x = random.randint(0, w - new_width)
                y = random.randint(0, h - new_height)
                img = img[y:y + new_height, x:x + new_width]

        # if Histogtam:
        #     if random_use_attack:
        #         rand = random.randint(1, 10)
        #     if rand <= 3:
        #         # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        #         img = cv2.imread(origin_image_path, cv2.IMREAD_GRAYSCALE)
        #         img = add_Histogtam(img)
        #         Histogtam = False

        # 平移变换
        # offset_ratio_limit = 0.5
        # offset_x = random.uniform(-offset_ratio_limit, offset_ratio_limit) * w
        # offset_y = random.uniform(-offset_ratio_limit, offset_ratio_limit) * h
        # img, M = translate_image(img, offset_x, offset_y)
        cv2.imwrite(
            output_dir_path + "/" + util.insert_str_in_file_name_end(util.get_file_name_from_path(origin_image_path),
                                                                     "_" + str(j)), img)
    # 生成完数据集后,制作直方图均衡化特效
    add_Histogtam(r'E:/img_data/by0222/copy/')


# 批量把一个文件夹下的图片以最小边为边长裁剪中间部分的正方形
def batch_center_crop_images(dir_path, output_dir_path, remove_exist_output_dir=True):
    util.create_dir(output_dir_path, remove_exist_output_dir)
    all_grouped_files_in_dir = util.get_all_grouped_file_path_in_dir(dir_path)
    grouped_file_len = len(all_grouped_files_in_dir)
    for i in range(grouped_file_len):
        if i % 10 == 0:
            print("batch_resize_images {} in {}".format(i, grouped_file_len))
        file_group_path = all_grouped_files_in_dir[i]
        for file_path in file_group_path:
            file_dir_name = util.get_file_dir_name(file_path)
            sub_dir = os.path.join(output_dir_path, file_dir_name)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            image = cv2.imread(file_path)
            height, width = image.shape[:2]
            min_side = min(height, width)
            start_width = int((width - min_side) / 2)
            start_height = int((height - min_side) / 2)
            center_crop_image = image[start_height:start_height + min_side, start_width:start_width + min_side]
            cv2.imwrite(os.path.join(sub_dir, util.get_file_name_from_path(file_path)), center_crop_image)


# 批量把一个文件夹下的图片等比缩放到最大边等于输入的目标像素大小
def batch_resize_images(dir_path, output_dir_path, target_max_size, remove_exist_output_dir=True):
    util.create_dir(output_dir_path, remove_exist_output_dir)
    all_grouped_files_in_dir = util.get_all_grouped_file_path_in_dir(dir_path)
    grouped_file_len = len(all_grouped_files_in_dir)
    for i in range(grouped_file_len):
        if i % 10 == 0:
            print("batch_resize_images {} in {}".format(i, grouped_file_len))
        file_group_path = all_grouped_files_in_dir[i]
        for file_path in file_group_path:
            file_dir_name = util.get_file_dir_name(file_path)
            sub_dir = os.path.join(output_dir_path, file_dir_name)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            image = cv2.imread(file_path)
            height, width = image.shape[:2]
            max_side = max(height, width)
            ratio = target_max_size / max_side
            resized_image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
            cv2.imwrite(os.path.join(sub_dir, util.get_file_name_from_path(file_path)), resized_image)


def scale_image(image, ratio_x, ratio_y):
    imgHeight, imgWidth, imgMode = image.shape
    dstHeight, dstWidth = int(imgHeight * ratio_y), int(imgWidth * ratio_x)

    M = numpy.float32([[ratio_x, 0, 0],
                       [0, ratio_y, 0]])

    dstImg = cv2.warpAffine(image, M, (dstWidth, dstHeight))
    return dstImg, M
    # return cv2.resize(image, (0, 0), None, ratio_x, ratio_y)


# 顺时针旋转
def rotate_bound(image, angle, avoid_crop=True):
    # if angle == 0:
    #     return image
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)
    # print("cx = ", cx, "cy = ", cy)
    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    target_w = w
    target_h = h
    if avoid_crop:
        # 改变生成图像的大小，保证图像旋转后不被裁减
        cos = numpy.abs(M[0, 0])
        sin = numpy.abs(M[0, 1])
        # 计算图像旋转后的新边界
        target_w = int((h * sin) + (w * cos))
        target_h = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (target_w / 2) - cx
        M[1, 2] += (target_h / 2) - cy

    rotated_image = cv2.warpAffine(image, M, (target_w, target_h))
    # if True:
    #     w, h = rotated_image.shape[:2]
    #     # 如果需要去除黑边
    #     # 裁剪角度的等效周期是180°
    #     angle_crop = angle % 180
    #     if angle > 90:
    #         angle_crop = 180 - angle_crop
    #     # 转化角度为弧度
    #     theta = angle_crop * numpy.pi / 180
    #     # 计算高宽比
    #     hw_ratio = float(h) / float(w)
    #     # 计算裁剪边长系数的分子项
    #     tan_theta = numpy.tan(theta)
    #     numerator = numpy.cos(theta) + numpy.sin(theta) * numpy.tan(theta)
    #
    #     # 计算分母中和高宽比相关的项
    #     r = hw_ratio if h > w else 1 / hw_ratio
    #     # 计算分母项
    #     denominator = r * tan_theta + 1
    #     # 最终的边长系数
    #     crop_mult = numerator / denominator
    #
    #     # 得到裁剪区域
    #     w_crop = int(crop_mult * w)
    #     h_crop = int(crop_mult * h)
    #     x0 = int((w - w_crop) / 2)
    #     y0 = int((h - h_crop) / 2)
    #
    #     rotated_image = rotated_image[x0:x0 + w_crop, y0 + h_crop]

    # return cv2.warpAffine(image, M, (target_w, target_h), None, None, cv2.BORDER_CONSTANT, (0, 0, 0)), M
    return rotated_image, M
    # return cv2.warpAffine(image, M, (w, h)), M


def translate_image(image, offset_x, offset_y):
    # print("offset_x = ", offset_x, "offset_y = ", offset_y)
    height, width = image.shape[:2]
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M = numpy.float32([[1, 0, offset_x], [0, 1, offset_y]])
    dst = cv2.warpAffine(image, M, (width, height), None, None, cv2.BORDER_CONSTANT, (0, 0, 0))
    return dst, M


# 获取图片二维数组中有像素值的横纵坐标的最大最小值，作为图像裁剪范围
def image_have_content_range(image):
    nonzero = numpy.nonzero(image)
    raw = nonzero[0]
    col = nonzero[1]
    min_raw = min(raw)
    min_col = min(col)
    max_raw = max(raw)
    max_col = max(col)
    return min_raw, min_col, max_raw, max_col


def connect_images(images):
    image = images[0]
    (h, w) = image.shape[:2]
    h_merge = image
    for i in range(1, len(images)):
        image = images[i]
        image = cv2.resize(image, (w, h))
        h_merge = numpy.hstack((h_merge, image))  # 水平拼接
    return h_merge


# flag： 是否拷贝关系（拷贝1，非拷贝0）
def image_to_flatten_array(image, flag):
    dst = numpy.zeros(image.shape, dtype=numpy.float32)
    cv2.normalize(image, dst=dst, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # 转化成numpy数组
    image_arr = numpy.array(dst)
    # image_arr = image_arr
    # print("image_arr.shape = ", image_arr.shape)
    flatten = image_arr.flatten()
    # flatten = numpy.append(flatten, 1)
    # flatten = flatten[:, numpy.newaxis]
    # 在第一位插入标志
    flatten = numpy.insert(flatten, 0, flag)
    flatten = flatten.T
    # print("flatten.shape = ", flatten.shape)
    return flatten


def images_to_matrix(dir_path, width, height, flag):
    image_list = []
    flatten_array = []
    image_file_paths = util.get_all_files_in_dir(dir_path)
    if len(image_file_paths) < 1:
        print("没有找到图片")
        return
    for image_file_path in image_file_paths:
        image = cv2.imread(image_file_path, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(image, (width, height))
        image_list.append(resized_image)
        array = image_to_flatten_array(resized_image, flag)
        # matrix = numpy.c_(matrix, array)
        flatten_array.append(array)
    matrix = numpy.array(flatten_array)
    return matrix


def save_to_used_match_dir(output_cache_image_file, img_origin, kp_origin, img_target, kp_target, used_match, flag,
                           save_file_name):
    if output_cache_image_file:
        out = img_origin
        img_out = cv2.drawMatches(img_origin, kp_origin, img_target, kp_target,
                                  used_match, out)
        cv2.imwrite([constant.used_SIFT_match_similar_dir, constant.used_SIFT_match_copy_dir][
                        flag] + save_file_name, img_out)


def save_to_input_feature_image_dir(output_cache_image_file, flag, image_name, image):
    if output_cache_image_file:
        cv2.imwrite([constant.input_feature_similar_image_dir, constant.input_feature_copy_image_dir][flag]
                    + image_name + ".jpg", image)


# 添加椒盐噪声
def salt_and_pepper_noise(img, proportion):
    noise_img = img
    height, width = noise_img.shape[0], noise_img.shape[1]
    num = int(height * width * proportion)  # 多少个像素点添加椒盐噪声
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if any(noise_img[h, w]) == 0:
            continue
        else:
            if random.randint(0, 1) == 0:
                noise_img[h, w] = 0
            else:
                noise_img[h, w] = 255

    return noise_img


# 添加高斯噪声
def gaussian_noise(img, mean, sigma):
    """
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    """
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = numpy.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = numpy.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = numpy.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out

#添加水印
# def add_watermark():
    # img1 = cv2.imread('linuxidc.com.jpg',cv2.IMREAD_COLOR)
    #文字，位置，字体，字号，颜色，厚度
    # text = 'www.linuxidc.com'
    # pos = (10,150)
    # font_type = 4
    # font_size = 2
    # color = (255,0,0)
    # bold = 1
    #
    # cv2.putText(img1,text,pos, font_type, font_size, color,bold)
    # cv2.imshow('www.linuxidc.com',img1)
    # cv2.waitKey(0)

#调整图像色调
def modify_color(filename):
    # 输入文件
    # filename = r'E:\img_data\copy-detect\Copy\000301\000301.jpg'
    # 目标色值
    target_hue = random.uniform(0,10)

    # 读入图片，转化为 RGB 色值
    image = Image.open(filename).convert('RGBA')

    # 将 RGB 色值分离
    image.load()
    r, g, b, a = image.split()
    result_r, result_g, result_b, result_a = [], [], [], []
    # 依次对每个像素点进行处理
    for pixel_r, pixel_g, pixel_b, pixel_a in zip(r.getdata(), g.getdata(),
                                                  b.getdata(), a.getdata()):
        # 转为 HSV 色值
        h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_b / 255., pixel_g / 255.)
        # 转回 RGB 色系
        rgb = colorsys.hsv_to_rgb(target_hue, s, v)
        pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
        # 每个像素点结果保存
        result_r.append(pixel_r)
        result_g.append(pixel_g)
        result_b.append(pixel_b)
        result_a.append(pixel_a)

    r.putdata(result_r)
    g.putdata(result_g)
    b.putdata(result_b)
    a.putdata(result_a)

    # 合并图片
    image = Image.merge('RGBA', (r, g, b, a))

    # PIL.Image转换成OpenCV格式
    image = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)

    return image
    # 输出图片
    # image.save('output.png')
    # Image._show(image)


#直方图均衡化##########################################
'''
说明：利用python/numpy/opencv实现直方图均衡化，其主要思想是将一副图像的直方图分布变成近似均匀分布，从而增强图像的对比度
算法思路:
        1)以灰度图的方式加载图片;
        2)求出原图的灰度直方图，计算每个灰度的像素个数在整个图像中所占的百分比;
		3)计算图像各灰度级的累积概率密度分布；
		4)求出新图像的灰度值。
'''
def Origin_histogram(img):
    # 建立原始图像各灰度级的灰度值与像素个数对应表
    histogram = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k in histogram:
                histogram[k] += 1
            else:
                histogram[k] = 1

    sorted_histogram = {}  # 建立排好序的映射表
    sorted_list = sorted(histogram)  # 根据灰度值进行从低至高的排序

    for j in range(len(sorted_list)):
        sorted_histogram[sorted_list[j]] = histogram[sorted_list[j]]

    return sorted_histogram


def equalization_histogram(histogram, img):
    pr = {}  # 建立概率分布映射表

    for i in histogram.keys():
        pr[i] = histogram[i] / (img.shape[0] * img.shape[1])

    tmp = 0
    for m in pr.keys():
        tmp += pr[m]
        pr[m] = max(histogram) * tmp

    new_img = numpy.zeros(shape=(img.shape[0], img.shape[1]), dtype=numpy.uint8)

    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = pr[img[k][l]]

    return new_img


def add_Histogtam(rootdir):
    filelist = os.listdir(rootdir)
    for i in range(len(filelist)):
        image_dir = os.listdir(rootdir + filelist[i])
        for x in range(3):
        #在每个文件夹中随机抽三个 做直方图均衡化
            num = random.randint(1,19)
            img = cv2.imread(rootdir + filelist[i] + '/' + image_dir[num],cv2.IMREAD_GRAYSCALE)

            origin_histogram = Origin_histogram(img)
            new_img = equalization_histogram(origin_histogram, img)

            cv2.imwrite(rootdir + filelist[i] + '/' + image_dir[num],new_img)

##########################################
