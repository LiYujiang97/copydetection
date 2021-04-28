# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import similar_copy_detect
import imageUtils
import util
import numpy
import time
import CNNProcess


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    # start = time.time()
    # similar_copy_detect.generate_subtract_images()
    # end = time.time()
    # CNNProcess.resNet_train()
    # print(end-start)

    # CNNProcess.resNet_valuate(r"D:\Work\academic\LoveLetter\fps1\006559.jpg",
    #                           r"D:\Work\academic\LoveLetter\fps1\006561.jpg")
    # CNNProcess.resNet_valuate(r"D:\Work\academic\copyImagesMaterial\457_1\457_1.jpg",
    #                           r"D:\Work\academic\copyImagesMaterial\457_1\457_1_1.jpg")
    # in_dir = util.get_all_grouped_file_path_in_dir(r"D:\Work\academic\copyImagesMaterial\10_1")
    # similar_copy_detect.get_subtract_image_package_list(1, r"D:\Work\academic\copyImagesMaterial\10_1\10_1.jpg",
    #                                                     *in_dir)
    # similar_copy_detect.get_subtract_image_package_list(1, r"D:\Work\academic\copyImagesMaterial\11\11_1.jpg",
    #                                                     r"D:\Work\academic\copyImagesMaterial\19\117_1_9.jpg")

    # similar_copy_detect.test_model(1,
    #                                r'D:\Work\academic\myTestImage\Lenna\Lenna_origin.jpg',
    #                                r'D:\Work\academic\myTestImage\Lenna\Lenna_3.jpg')
    # imageUtils.generate_copy_images(r"D:\Work\academic\myTestImage\Lena\Lenna_origin.jpg", 6)

    # imageUtils.batch_generate_copy_images(r"D:\Work\academic\Challenging-Ground-truth",
    #                                       r"D:\Work\academic\copyImagesMaterial", 500, 20)
    # util.create_dir_for_every_file(r"D:\Work\dataset\copyday\copydays_original_strong")
    # util.rename_file_in_dir(r"D:\Work\dataset\copyday\copydays_jpeg_rename", "jpeg")
    util.copy_files_to_same_kind_dir(r"D:\Work\dataset\copyday\copydays_strong_rename",
                                     r"D:\Work\dataset\copyday\copydays_original_strong")
    # util.temp_rename_file_in_dir(r"D:\Work\dataset\copyday\copydays_strong_rename", "strong")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
