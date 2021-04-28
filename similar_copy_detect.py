import cv2
import numpy

import CNNProcess
import autoEncoder
import os
import random
import util
import shutil
import warnings
import imageUtils
import constant

from AugmentMatch import AugmentMatch
import pandas

# 是否输出处理后的图片到缓存文件夹以便查看
output_cache_image_file = True
# 清除之前的相减图片数据
clean_subtract_image_data = True

if output_cache_image_file:
    if os.path.exists(constant.image_cache_dir):
        shutil.rmtree(constant.image_cache_dir, True)


def generate_subtract_images(copy_dir_path, similar_dir_path, copy_dir_start_index=0, copy_dir_sub_dir_num_limit=-1,
                             copy_dir_is_successive_sub=False, similar_dir_start_index=0,
                             similar_dir_sub_dir_num_limit=-1):
    if clean_subtract_image_data:
        if os.path.exists(constant.align_image_dir):
            shutil.rmtree(constant.align_image_dir, True)
    init_cache_file()

    generate_subtract_images_in_self_folder(copy_dir_path, constant.copy_flag, copy_dir_start_index,
                                            copy_dir_sub_dir_num_limit, copy_dir_is_successive_sub)
    generate_subtract_images_with_other_folder(similar_dir_path, similar_dir_start_index, similar_dir_sub_dir_num_limit)


def init_cache_file():
    if output_cache_image_file:
        # if not os.path.exists(constant.image_cache_dir):
        try:
            os.makedirs(constant.input_feature_similar_image_dir)
            os.makedirs(constant.input_feature_copy_image_dir)
            os.makedirs(constant.used_SIFT_match_similar_dir)
            os.makedirs(constant.used_SIFT_match_copy_dir)
            os.makedirs(constant.dealt_similar_image_dir)
            os.makedirs(constant.dealt_copy_image_dir)
        except Exception as e:
            warnings.warn(str(e))
    if not os.path.exists(constant.align_image_dir):
        os.makedirs(constant.align_image_dir)
        os.makedirs(constant.align_similar_image_dir)
        os.makedirs(constant.align_copy_image_dir)


# 输入的父文件夹中的每一个文件夹中的第一张图与下一个文件夹中的每一种做差得到残差图像
def generate_subtract_images_with_other_folder(dir_path, start_index=0, sub_dir_num_limit=-1):
    all_grouped_files_in_dir = util.get_all_grouped_file_path_in_dir(dir_path)
    # 限制遍历文件夹的数量
    grouped_file_len = len(all_grouped_files_in_dir)
    if start_index + sub_dir_num_limit > grouped_file_len:
        warnings.warn("目标文件数量 {} 超过文件总数量 {}".format(start_index + sub_dir_num_limit, grouped_file_len))
        return
    iterator_num = [grouped_file_len - start_index, min(sub_dir_num_limit, grouped_file_len)][sub_dir_num_limit > 0]
    # iterator_num = len(all_grouped_files_in_dir)
    for i in range(start_index, start_index + iterator_num):
        file_path_group = all_grouped_files_in_dir[i]
        if len(file_path_group) > 1:
            origin_path = file_path_group[0]
            next_dir_file_path_group = all_grouped_files_in_dir[(i + 1) % grouped_file_len]
            # 每一张图依次跟后面一张图相减
            for j in range(len(next_dir_file_path_group)):
                target_file_path = next_dir_file_path_group[j]
                get_subtract_image(origin_path, target_file_path, constant.similar_flag)
        else:
            warnings.warn("第{}个文件夹中没有图片".format(i))
    pass


# 便利文件夹中的每张图片，两两对其相减（第一张与后面每一张相减或者每一张都与后一张相减）生成一张残差图
def generate_subtract_images_in_self_folder(dir_path, flag, start_index=0, sub_dir_num_limit=-1,
                                            is_successive_sub=False):
    all_grouped_files_in_dir = util.get_all_grouped_file_path_in_dir(dir_path)
    # 限制遍历文件夹的数量
    grouped_file_len = len(all_grouped_files_in_dir)
    if start_index + sub_dir_num_limit > grouped_file_len:
        warnings.warn("目标文件数量 {} 超过文件总数量 {}".format(start_index + sub_dir_num_limit, grouped_file_len))
        return
    iterator_num = [grouped_file_len - start_index, min(sub_dir_num_limit, grouped_file_len)][sub_dir_num_limit > 0]
    # iterator_num = len(all_grouped_files_in_dir)
    for i in range(start_index, start_index + iterator_num):
        file_path_group = all_grouped_files_in_dir[i]
        if len(file_path_group) > 1:
            if is_successive_sub:
                # 每一张图依次跟后面一张图相减
                for j in range(len(file_path_group) - 1):
                    origin_path = file_path_group[j]
                    target_file_path = file_path_group[j + 1]
                    subtract_image = get_subtract_image(origin_path, target_file_path, flag)
            else:
                origin_path = file_path_group[0]
                target_file_paths = file_path_group[1:]
                subtract_image_list = get_subtract_image_list(flag, origin_path, *target_file_paths)
        else:
            warnings.warn("第{}个文件夹中没有图片".format(i))
    return None


# 一张图与一组图的相减结果列表
def get_subtract_image_list(flag, origin_path, *target_files_path):
    subtract_image_list = []
    for target_file in target_files_path:
        # 每一组图片由于可能有多个匹配的SIFT特征点，相减完会返回多张图片
        subtract_image = get_subtract_image(origin_path, target_file, flag)
        if subtract_image is not None:
            subtract_image_list.append(subtract_image)
    return subtract_image_list


# 一对图片相减返回的图片
def get_subtract_image(origin_img_path, target_img_path, flag):
    print("target_img_path = ", target_img_path)
    img_read_origin = cv2.imread(origin_img_path, cv2.IMREAD_COLOR)
    img_read_target = cv2.imread(target_img_path, cv2.IMREAD_COLOR)
    if img_read_origin is None or img_read_target is None:
        warnings.warn("图片不存在")
        return None
    kp_origin, des = sift_kp(img_read_origin)
    kp_target, des_target = sift_kp(img_read_target)
    # print("image_origin_with_kp.shape = ", image_origin_with_kp.shape, "kp_origin.length = ", len(kp_origin),
    #       "des.shape = ", des.shape)
    # cv2.imshow('origin', image_origin_with_kp)
    # cv2.imshow('target', image_target_with_kp)
    good_match_array = get_good_match(des, des_target)

    return use_affine_get_subtract_image(flag, good_match_array, img_read_origin, img_read_target, kp_origin,
                                         kp_target, origin_img_path, target_img_path)
    # return use_perspective(flag, good_match_array, img_read_origin, img_read_target, kp_origin, image_origin_with_kp, image_target_with_kp,
    #                        kp_target, origin_img_path, target_img_path)


# 使用仿射变换
def use_affine_get_subtract_image(flag, good_match_array, img_read_origin, img_read_target, kp_origin, kp_target,
                                  origin_img_path, target_img_path):
    # out = img_read_origin
    # used_match = good_match_array
    # img_out = cv2.drawMatches(image_origin_with_kp, kp_origin, image_target_with_kp, kp_target, used_match, out)
    # cv2.imshow('matchImage %s' % target_img_path, img_out)
    # cv2.waitKey()
    target_image_name = util.get_file_name_from_path(target_img_path)
    origin_image_name = util.get_file_name_from_path(origin_img_path)
    if len(good_match_array) < 1:
        warnings.warn("{}与{}不存在匹配点".format(origin_img_path, target_img_path))
        subtract_image = direct_subtract_images(img_read_origin, img_read_target)
    else:
        print("过滤前的匹配数: {}".format(len(good_match_array)))
        good_match_array = RANSAC_filter(good_match_array, kp_origin, kp_target)
        print("{} RANSAC 过滤后的匹配数: {}".format(target_img_path, len(good_match_array)))
        # filtered_augment_match_array = generate_augment_match(kp_origin, kp_target, good_match_array)

        imageUtils.save_to_used_match_dir(output_cache_image_file, img_read_origin, kp_origin, img_read_target,
                                          kp_target,
                                          good_match_array, flag,
                                          util.insert_str_in_file_name_end(target_image_name,
                                                                           "_RANSAC_filtered_matched"))
        # show_good_matched_kp_image(good_match_array, img_read_origin, img_read_target, kp, kp_target)

        '''
        直方图过滤
        filtered_augment_match_array = histogram_filter_match_array(kp_origin, kp_target, good_match_array)
        print("{} 直方图统计过滤后的匹配数: {}".format(target_img_path, len(filtered_augment_match_array)))
    
        # out = img_read_origin
        # img_out = cv2.drawMatches(image_origin_with_kp, kp_origin, image_target_with_kp, kp_target,
        #                               good_match_array, out)
        # cv2.imshow("nofilter", img_out)
    
        filtered_match_length = len(filtered_augment_match_array)
        if filtered_match_length < 1:
            warnings.warn("过滤后不存在匹配点")
            return subtract_image_list
    
        used_match = AugmentMatch.get_match_array_by_augment_array(filtered_augment_match_array)
        imageUtils.save_to_used_match_dir(output_cache_image_file, img_read_origin, kp_origin, img_read_target, kp_target,
                                          used_match, flag, util.insert_str_in_file_name_end(target_image_name,
                                                                                             "_histogram_filtered_matched"))
        '''
        filtered_augment_match_array = generate_augment_match(kp_origin, kp_target, good_match_array)
        align_image = deal_with_align(flag, img_read_origin, img_read_target, kp_origin, kp_target,
                                      filtered_augment_match_array, target_image_name)
        if align_image is None:
            warnings.warn("根据sift对齐相减失败，直接相减")
            subtract_image = direct_subtract_images(img_read_origin, img_read_target)
        else:
            img_read_target_gray = cv2.cvtColor(img_read_target, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('align_image', align_image)
            align_image_gray = cv2.cvtColor(align_image, cv2.COLOR_BGR2GRAY)

            # 如果处理后的图片全黑，直接跳过。其实图片平移变换后如果是png图片应该是在边上留下无内容的透明区域，但是jpg无内容部分填充了黑色，如果匹配的特征点周围也是黑色，就可能出现全黑的情况。暂时这样简单处理
            if not numpy.any(align_image_gray):
                # print("%s_%d图片全黑" % target_img_path % str(i))
                warnings.warn("{}对齐{}的图片全黑".format(origin_img_path, target_img_path))
                return None
            # crop_image_gray, crop_img_target_gray = deal_image_have_common_content(align_image_gray, img_read_target_gray)

            # 把目标图片有内容的部分（像素值>5，如果设阈值为0会出现锯齿，原因不明）黑白二值化，作为后面的掩膜mask
            # ret, thresh = cv2.threshold(img_read_target_gray, 5, 255, cv2.THRESH_BINARY)
            # cv2.imshow("img_read_target_gray", img_read_target_gray)
            # cv2.imshow("thresh", thresh)
            # crop_image_gray = cv2.add(align_image_gray, numpy.zeros(numpy.shape(align_image_gray), dtype=numpy.uint8),
            #                           mask=thresh)
            # cv2.imshow("crop_image_gray", crop_image_gray)
            # cv2.waitKey()
            crop_image_gray = align_image_gray
            # print("align_image_gray.shape = ", align_image_gray.shape, "img_read_target_gray.shape = ", img_read_target_gray.shape)
            # mix_image = cv2.addWeighted(crop_image_gray, 0.5, crop_img_target_gray, 0.5, 0)
            subtract_image = cv2.absdiff(crop_image_gray, img_read_target_gray)
            if output_cache_image_file:
                connect_image = imageUtils.connect_images([img_read_origin, align_image, img_read_target])
                cv2.imwrite([constant.dealt_similar_image_dir, constant.dealt_copy_image_dir][
                                flag] + util.insert_str_in_file_name_end(
                    target_image_name, "_deal_compare"), connect_image)
                connect_image = imageUtils.connect_images([crop_image_gray, img_read_target_gray])
                cv2.imwrite([constant.dealt_similar_image_dir, constant.dealt_copy_image_dir][
                                flag] + util.insert_str_in_file_name_end(
                    target_image_name, "_deal_compare_crop"), connect_image)
                # cv2.imwrite([constant.align_similar_image_dir, constant.align_copy_image_dir][
                #                 flag] + util.insert_str_in_file_name_end(
                #     target_image_name, "_mix_image_" + str(i)), mix_image)

            # cv2.imshow('align_image_gray %d' % i, align_image_gray)
            # cv2.imwrite(cache_dir + util.insert_str_in_file_name_end(origin_image_name, "_origin_" + str(i)),
            #             align_image_gray)
            # cv2.imwrite(cache_dir + util.insert_str_in_file_name_end(target_image_name, "_target_" + str(i)),
            #             img_read_target_gray)
            # cv2.imshow('img_read_target_gray %d' % i, img_read_target_gray)
            # cv2.imshow('mix_image %d' % i, mix_image)
            # cv2.imshow('subtract_image %d' % i, subtract_image)

            # cv2.waitKey()
    if subtract_image is None:
        # cv2.imshow("align_image", align_image)
        # cv2.imshow("img_read_target", img_read_target)
        # cv2.waitKey()
        warnings.warn("{} subtract_image 的宽或高为0,{}".format(target_image_name, subtract_image))
        return None
    cv2.imwrite([constant.align_similar_image_dir, constant.align_copy_image_dir][
                    flag] + util.get_file_name_from_path(target_image_name), subtract_image)

    # mixed_subtract_image = mix_subtract_image(subtract_image_list)
    return subtract_image


# 两张图片调整对齐大小后直接相减
def direct_subtract_images(img_read_origin, img_read_target):
    height, width = img_read_target.shape[:2]
    resized_origin = cv2.resize(img_read_origin, (width, height))
    origin_image_gray = cv2.cvtColor(resized_origin, cv2.COLOR_BGR2GRAY)
    target_image_gray = cv2.cvtColor(img_read_target, cv2.COLOR_BGR2GRAY)
    subtract_image = cv2.absdiff(origin_image_gray, target_image_gray)
    return subtract_image


def RANSAC_filter(good_match_array, kp_origin, kp_target):
    if len(good_match_array) > 4:
        filter_good_match_array = []
        # 为了通过 RANSAC 获得进一步过滤误匹配之后的sift
        ptsA = numpy.float32([kp_origin[m.queryIdx].pt for m in good_match_array]).reshape(-1, 1, 2)
        ptsB = numpy.float32([kp_target[m.trainIdx].pt for m in good_match_array]).reshape(-1, 1, 2)
        ransacReprojThreshold = 3
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        status = status.flatten()
        for i in range(len(status)):
            if status[i]:
                filter_good_match_array.append(good_match_array[i])
        return filter_good_match_array
    else:
        return good_match_array


# 由于图像平移后可能出现一部分无像素的情况，裁剪图片获取两张图都开始有像素的公共部分
def deal_image_have_common_content(image1, image2):
    content_range_dealt = imageUtils.image_have_content_range(image1)
    content_range_target = imageUtils.image_have_content_range(image2)
    start_row = max(content_range_dealt[0], content_range_target[0])
    start_col = max(content_range_dealt[1], content_range_target[1])
    end_row = min(content_range_dealt[2], content_range_target[2])
    end_col = min(content_range_dealt[3], content_range_target[3])
    image1 = image1[start_row:end_row + 1, start_col:end_col + 1]
    image2 = image2[start_row:end_row + 1, start_col:end_col + 1]
    return image1, image2


# 多张对齐完相减的图再全部叠加生成一张图。由于两张图片不同SIFT点的尺度偏差，对齐相减的图大小可能会相差很多，所显示的区域也不同，不能这么直接叠加生成
def mix_subtract_image(subtract_image_list):
    if len(subtract_image_list) <= 1:
        print("匹配的特征点小于2")
        return
    result = subtract_image_list[0]
    for i in range(1, len(subtract_image_list)):
        result = cv2.add(result, subtract_image_list[i])
    # cv2.imshow('mix_image', result)
    # cv2.waitKey()

    return result


# 显示过滤之后的sift图像
def show_good_matched_kp_image(good_match_array, img_read_origin, img_read_target, kp, kp_target):
    good_kp, good_kp_target = get_matched_kp(kp, kp_target, good_match_array)
    gray = cv2.cvtColor(img_read_origin, cv2.COLOR_BGR2GRAY)
    img = cv2.drawKeypoints(gray, good_kp, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('origin_good', img)
    gray = cv2.cvtColor(img_read_target, cv2.COLOR_BGR2GRAY)
    img = cv2.drawKeypoints(gray, good_kp_target, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('target_good', img)
    cv2.waitKey()


# 使用直方图统计过滤匹配
def histogram_filter_match_array(kp_origin_array, kp_target_array, match_array):
    augment_match_list = generate_augment_match(kp_origin_array, kp_target_array, match_array)
    scale_ratio_list = []
    rotate_angle_list = []
    filtered_augment_match_array = []

    match_length = len(augment_match_list)
    if match_length < 10:
        warnings.warn("匹配点数量({})太少，不进行直方图统计过滤".format(match_length))
        return augment_match_list
    for augment_match in augment_match_list:
        scale_ratio_list.append(augment_match.scale_ratio)
        rotate_angle_list.append(augment_match.rotate_angle)

    scale_ratio_series = pandas.Series(scale_ratio_list)
    min_scale_ratio = min(scale_ratio_list)
    max_scale_ratio = max(scale_ratio_list)
    # range_array = numpy.arange(min_scale_ratio, max_scale_ratio, )

    scale_ratio_length = len(scale_ratio_list)
    # 直方图数量
    cut_num = scale_ratio_length // 10
    if cut_num < 1:
        cut_num = 5
    cut_num = min(cut_num, 20)
    # print("直方图数量：{}".format(cut_num))
    # 缩放统计直方图
    scale_ratio_category = pandas.cut(scale_ratio_series, cut_num)
    scale_ratio_histogram = pandas.value_counts(scale_ratio_category)
    # 占最多数量的缩放直方图的坐标范围
    scale_ratio_histogram_max_interval = scale_ratio_histogram.index[0]
    scale_ratio_second_interval = pandas.Interval(0, 0)
    if len(scale_ratio_histogram) > 1:
        if scale_ratio_histogram[1] > scale_ratio_histogram[0] * 0.7:
            # 如果排第二的直方图跟第一的比差距不到0.7，也算在内
            scale_ratio_second_interval = scale_ratio_histogram.index[1]

    rotate_angle_series = pandas.Series(rotate_angle_list)
    # 旋转角度统计直方图
    max_rotate_category = pandas.cut(rotate_angle_series, cut_num)
    rotate_angle_histogram = pandas.value_counts(max_rotate_category)
    rotate_angle_histogram_max_interval = rotate_angle_histogram.index[0]
    # 占最多数量的直方图所包含的具体数量
    rotate_angle_histogram_second_interval = pandas.Interval(0, 0)
    # if len(rotate_angle_histogram) > 1:
    #     if rotate_angle_histogram[1] > rotate_angle_histogram[0] * 0.7:
    #         # 如果排第二的直方图跟第一的比差距不到0.7，也算在内
    #         rotate_angle_histogram_second_interval = rotate_angle_histogram.index[1]

    for augment_match in augment_match_list:
        scale_ratio = augment_match.scale_ratio
        rotate_angle = augment_match.rotate_angle
        if scale_ratio in scale_ratio_histogram_max_interval or scale_ratio in scale_ratio_second_interval:
            if rotate_angle in rotate_angle_histogram_max_interval or rotate_angle in rotate_angle_histogram_second_interval:
                filtered_augment_match_array.append(augment_match)

    return filtered_augment_match_array


def generate_augment_match(kp_origin_array, kp_target_array, match_array):
    augment_match_list = []
    match_array_length = len(match_array)
    for i in range(match_array_length):
        match = match_array[i]
        kp = kp_origin_array[match.queryIdx]
        kp_target = kp_target_array[match.trainIdx]

        (angle, point_x, point_y, size) = get_key_point_info(kp)
        # print("angle = ", angle, "size = ", size)
        (angle_target, point_x_target, point_y_target, size_target) = get_key_point_info(kp_target)
        # print("angle_target = ", angle_target, "size_target = ", size_target)

        rotate_angle = angle_target - angle
        scale_ratio = size_target / size
        # 统一旋转角度到0~180之间
        if rotate_angle < -180:
            rotate_angle = 360 + rotate_angle
        if rotate_angle > 180:
            rotate_angle = rotate_angle - 360
        augment_match = AugmentMatch(match, scale_ratio, rotate_angle, kp, kp_target)
        augment_match_list.append(augment_match)
    return augment_match_list


# 使用3个匹配点的坐标调用opencv的api生成仿射变换矩阵，作用在整个图像上
def deal_with_align(flag, image, image_target, kp_origin_array, kp_target_array, augment_match_array,
                    target_image_name):
    augment_match_length = len(augment_match_array)
    if augment_match_length < 3:
        warnings.warn("匹配数小于3")
        return None

    # 不同的match对应的两个kp的坐标有可能是完全相同的，只是他们的尺度和方向不同。对于后面使用坐标生成仿射变换的方法，需要在此手动过滤掉那些kp坐标完全相同的match
    filtered_match_array = []
    coordinate_array = []
    # 排序
    augment_match_array.sort(key=lambda aug_match: aug_match.get_key_point_distance())
    for augment_match in augment_match_array:
        key_point = augment_match.key_point
        key_point_target = augment_match.key_point_target
        (angle, point_x, point_y, size) = get_key_point_info(key_point)
        # print("angle = ", angle, "size = ", size)
        (angle_target, point_x_target, point_y_target, size_target) = get_key_point_info(key_point_target)
        coordinate = (point_x, point_y, point_x_target, point_y_target)
        if coordinate not in coordinate_array:
            coordinate_array.append(coordinate)
            filtered_match_array.append(augment_match.match)

    match_length = len(filtered_match_array)
    print("augment_match_length={},match_length={}".format(augment_match_length, match_length))
    if match_length < 3:
        warnings.warn("可使用的匹配数小于3")
        return None

    # affine_use_match = random.sample(filtered_match_array, 3)
    # 使用排序后的第一个，中间，和最后一个
    affine_use_match = [filtered_match_array[0], filtered_match_array[int(match_length / 2)],
                        filtered_match_array[match_length - 1]]
    # affine_use_match = [filtered_match_array[0], filtered_match_array[-1],
    #                     filtered_match_array[len(filtered_match_array) // 2]]
    imageUtils.save_to_used_match_dir(output_cache_image_file, image, kp_origin_array, image_target, kp_target_array,
                                      affine_use_match, flag,
                                      util.insert_str_in_file_name_end(target_image_name, "_used_match"))

    src_array = []
    dst_array = []
    for match in affine_use_match:
        kp = kp_origin_array[match.queryIdx]
        kp_target = kp_target_array[match.trainIdx]
        (angle, point_x, point_y, size) = get_key_point_info(kp)
        # print("angle = ", angle, "size = ", size)
        (angle_target, point_x_target, point_y_target, size_target) = get_key_point_info(kp_target)
        src_array.append(point_x)
        src_array.append(point_y)
        dst_array.append(point_x_target)
        dst_array.append(point_y_target)
    src = numpy.float32(src_array).reshape((3, 2))
    dst = numpy.float32(dst_array).reshape((3, 2))
    affine_M = cv2.getAffineTransform(src, dst)
    (h, w) = image_target.shape[:2]
    affine_result_image = cv2.warpAffine(image, affine_M, (w, h))
    # cv2.imshow("image_target", image_target)
    # cv2.imshow("result", affine_result_image)
    # cv2.waitKey()
    return affine_result_image


# 根据一组SIFT点的尺度，角度关系缩放，旋转图像返回一组变换之后的图像
def deal_with_align_manual(image, image_target, kp_origin_array, kp_target_array, augment_match_array):
    result_image_list = []
    match_length = len(augment_match_array)
    num = min(5, match_length)
    # 随机获取使用的匹配点
    used_augment_match = random.sample(augment_match_array, num)
    for i in range(len(used_augment_match)):
        filtered_augment_match = used_augment_match[i]
        kp = filtered_augment_match.key_point
        kp_target = filtered_augment_match.key_point_target

        (angle, point_x, point_y, size) = get_key_point_info(kp)
        # print("angle = ", angle, "size = ", size)
        (angle_target, point_x_target, point_y_target, size_target) = get_key_point_info(kp_target)
        # print("angle_target = ", angle_target, "size_target = ", size_target)
        factor = filtered_augment_match.scale_ratio
        rotate_angle = filtered_augment_match.rotate_angle

        if factor > 20:
            # 如果出现缩放比例特别高的情况
            warnings.warn("factor = {}".format(factor))
            continue

        rotate_image, M_rotate = imageUtils.rotate_bound(image, rotate_angle)
        origin_kp_array = numpy.array([[point_x], [point_y], [1]])
        coor_after_rotate = numpy.dot(M_rotate, origin_kp_array)
        scale_img, M_scale = imageUtils.scale_image(rotate_image, factor, factor)
        # coor_after_scale = numpy.dot(M_scale, [[coor_after_rotate[0][0]], [coor_after_rotate[1][0]], [1]])
        coor_after_scale = numpy.dot(M_scale, numpy.row_stack((coor_after_rotate, [1])))
        result, M_translate = imageUtils.translate_image(scale_img, point_x_target - coor_after_scale[0],
                                                         point_y_target - coor_after_scale[1])
        # cv2.imshow("match_img" + str(i), img_out)
        # cv2.imshow("rotate" + str(i), rotate_image)
        # cv2.imshow("scale" + str(i), scale_img)
        # cv2.imshow("res" + str(i), result)
        # cv2.waitKey()
        result_image_list.append(result)
        coor_result = numpy.dot(M_translate, numpy.row_stack((coor_after_scale, [1])))
        # print(M_rotate)
        # print("[point_x] = ", [point_x], ", [point_y] = ", [point_y])
        # print("[point_x_target] = ", [point_x_target], "[point_y_target] = ", [point_y_target])
        # print("coor_result = ", coor_result)
        # print("M_rotate = ", M_rotate)
        # print("coor_after_rotate = ", coor_after_rotate)
        # print("coor_after_scale = ", coor_after_scale)
    return result_image_list


def get_key_point_info(kp):
    angle = kp.angle
    point_x = kp.pt[0]
    point_y = kp.pt[1]
    size = kp.size
    return angle, point_x, point_y, size


def get_matched_kp(kp_origin, kp_target, match_array):
    good_kp = []
    good_kp_target = []
    for item in match_array:
        good_kp.append(kp_origin[item.queryIdx])
        good_kp_target.append(kp_target[item.trainIdx])
    return good_kp, good_kp_target


def get_average_sf(kp_list):
    length = len(kp_list)
    angle = 0
    point_x = 0
    point_y = 0
    size = 0
    for item in kp_list:
        angle += item.angle
        point_x += item.pt[0]
        point_y += item.pt[1]
        size += item.size
        print(item.angle)

    angle /= length
    point_x /= length
    point_y /= length
    size /= length
    return angle, (point_x, point_y), size


def sift_kp(image):
    # sift = cv2.SIFT()
    # kp, des = sift.detectAndCompute(image, None)
    # kp_image = cv2.drawKeypoints(image, kp, None)

    # 灰度图 shape 只有两个参数，但是这只是自己少量图片测出来的规律。不严谨
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # cv__sift = cv2.SIFT()
    cv__sift = cv2.SIFT_create()
    kp, descriptor = cv__sift.detectAndCompute(gray, None)
    # kp = cv__sift.detect(gray, None)
    # img = cv2.drawKeypoints(gray, kp, None)
    # 画出附带尺度大小，方向等信息的kp点
    # img = cv2.drawKeypoints(gray, kp, None, None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, descriptor

    # K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类。在进行特征点匹配时，一般使用KNN算法找到最近邻的两个数据点，
    # 如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good match（由Lowe在SIFT论文中提出）。


def get_good_match(des1, des2):
    good = []
    bf = cv2.BFMatcher()
    if des1 is None:
        warnings.warn("不存在原始描述子")
        return good
    if des2 is None:
        warnings.warn("不存在目标描述子")
        return good
    matches = bf.knnMatch(des1, des2, 2)
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)
    for i, pair in enumerate(matches):
        # 有可能不存在2对匹配
        if len(pair) < 2:
            good.append(pair[0])
        else:
            m, n = pair
            if m.distance < 0.7 * n.distance:
                good.append(m)
    # print("goodMatchedArray.length = ", len(good))

    # 删除matches里面的空list，并且根据距离排序
    # while [] in matches:
    #     matches.remove([])
    # good = sorted(good, key=lambda x: x[0].distance)
    return good


def divide_train_val_data(target_dir_path):
    align_similar_images = util.get_all_files_in_dir(constant.align_similar_image_dir)
    align_copy_images = util.get_all_files_in_dir(constant.align_copy_image_dir)
    similar_sub_length = len(align_similar_images)
    copy_sub_length = len(align_copy_images)
    if similar_sub_length == 0 or copy_sub_length == 0:
        return warnings.warn("指定的文件夹下没有数据")
    # 训练数据与测试数据7：3划分
    similar_divide_index = int(similar_sub_length * 0.7)
    copy_divide_index = int(copy_sub_length * 0.7)
    train_similar_images = align_similar_images[:similar_divide_index]
    val_similar_images = align_similar_images[similar_divide_index:]
    train_copy_images = align_copy_images[:copy_divide_index]
    val_copy_images = align_copy_images[copy_divide_index:]
    if os.path.exists(target_dir_path):
        shutil.rmtree(target_dir_path, True)
    train_similar_dir_path = target_dir_path + "/train/similar/"
    val_similar_dir_path = target_dir_path + "/val/similar/"
    train_copy_dir_path = target_dir_path + "/train/copy/"
    val_copy_dir_path = target_dir_path + "/val/copy/"
    util.copy_files_to_dir(train_similar_dir_path, *train_similar_images)
    util.copy_files_to_dir(val_similar_dir_path, *val_similar_images)
    util.copy_files_to_dir(train_copy_dir_path, *train_copy_images)
    util.copy_files_to_dir(val_copy_dir_path, *val_copy_images)

def train():
    # generate_subtract_images(r"D:\Work\academic\LoveLetterMaterial\CenterCropLoveLetterIFPSMaterialCopyRandom_noise_crop_scale_rotate",
    #                          r"D:\Work\academic\LoveLetterMaterial\CenterCropLoveLetterIFPSMaterialCopyRandom_noise_crop_scale_rotate",
    #                          copy_dir_start_index=0, copy_dir_sub_dir_num_limit=500, similar_dir_start_index=0,
    #                          similar_dir_sub_dir_num_limit=500)
    target_save_dir_path = r"D:\Work\dataset\LoveLetterMaterial\subtractCenterCropLoveLetterIFPSRandomRotateCropCopy"
    # divide_train_val_data(target_save_dir_path)
    CNNProcess.train(target_save_dir_path, 20)


def val():
    # subtract_image = get_subtract_image(origin_image_path, target_image_path, 0)
    # cv2.imshow("subtract_image", subtract_image)
    # cv2.waitKey()
    # CNNProcess.valuate(subtract_image, True)
    generate_subtract_images(r"E:\img_data\copydays\copydays_original_jpeg",
                             r"E:\img_data\copydays\copydays_original_jpeg",
                             copy_dir_start_index=0, copy_dir_sub_dir_num_limit=-1, similar_dir_start_index=0,
                             similar_dir_sub_dir_num_limit=-1)
    # CNNProcess.valuate(constant.align_image_dir, True)


if __name__ == '__main__':
    # train()
    val()
