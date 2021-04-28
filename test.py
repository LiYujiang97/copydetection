import torch
import numpy
import cv2

import imageUtils
import util

import os
import shutil
import dataAnalyze

if __name__ == '__main__':
    # running_corrects = 0
    # preds = torch.ones((2, 2))
    # array = numpy.array([[1, 1], [1, 2]])
    # labels = torch.from_numpy(array)
    #
    # data = preds == labels.data
    # data.numpy()
    # running_corrects += torch.sum(data)
    # print(running_corrects)

    # img_read_origin = cv2.imread(r"D://Work//academic//Challenging-Ground-truth//1//114_1.jpg", cv2.IMREAD_GRAYSCALE)
    # img_read_target = cv2.imread(r"D://Work//academic//Challenging-Ground-truth//1//114_1_crop.jpg", cv2.IMREAD_GRAYSCALE)
    # image = cv2.add(img_read_origin, numpy.zeros(numpy.shape(img_read_origin), dtype=numpy.uint8), mask=img_read_target)
    # cv2.imshow("image",image)
    # cv2.waitKey()

    # util.create_dir_for_every_file(r"E:\img_data\cifar10_data\automobile")
    # imageUtils.generate_copy_images(
    #     r"D:\Work\academic\myTestImage\Lenna\usualLena\Lenna_origin.jpg", 100, r"D:\Work\academic\myTestImage\Lenna\usualLena", True,
    #     True, True, True
    # )
    imageUtils.batch_generate_copy_images(r"E:\img_data\cifar10_data\automobile",
                                          r"E:\img_data\cifar10_data\automobileCopy_rotate_crop",
                                          per_copy_num=20, remove_exist_output_dir=False, add_gaussian_noise=False,
                                          add_salt_noise=False, add_crop=True, add_scale=False, add_rotate=True,
                                          random_use_attack=False)
    # imageUtils.batch_generate_copy_images(r"D:\Work\academic\copyImagesMaterial",
    #                                       r"D:\Work\academic\copyImagesMaterial_rotate",sub_dir_num_limit=50,
    #                                       per_copy_num=20, remove_exist_output_dir=False, add_noise=False,
    #                                       add_crop=False, add_scale=False, add_rotate=True, random_use_attack=False)
    # imageUtils.batch_center_crop_images(r"D:\Work\academic\ForrestGumpMaterial\ForrestGumpAllSimilarMaterial",
    #                                     r"D:\Work\academic\ForrestGumpMaterial\CenterCropForrestGumpAllSimilarMaterial")
    # imageUtils.batch_resize_images(r"D:\Work\academic\LoveLetterMaterial\CenterCropLoveLetterIFPSMaterial",
    #                                r"D:\Work\academic\LoveLetterMaterial\ResizeCenterCropLoveLetterIFPSMaterial", 512,
    #                                True)
    # imageUtils.batch_generate_copy_images(r"D:\Work\academic\myTestImage\mutualSimilarCopyGroundTruth\train",
    #                                       r"D:\Work\academic\myTestImage\mutualSimilarCopyGroundTruth\train",
    #                                       per_copy_num=20, remove_exist_output_dir=False, add_noise=False,
    #                                       add_crop=True, add_scale=False, add_rotate=True)
