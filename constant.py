cache_dir = "cache/"
image_cache_dir = cache_dir + "image/"
# 用来查看图片对齐效果的
align_image_dir = cache_dir + "align_subtract/"
align_similar_image_dir = align_image_dir + "similar/"
align_copy_image_dir = align_image_dir + "copy/"
# 处理完之后用来转换成向量输入到模型的
input_feature_image_dir = image_cache_dir + "inputFeature/"
input_feature_similar_image_dir = input_feature_image_dir + "similar/"
input_feature_copy_image_dir = input_feature_image_dir + "copy/"
# 使用到的SIFT匹配点匹配
used_SIFT_match_dir = image_cache_dir + "usedSIFTMatch/"
used_SIFT_match_similar_dir = used_SIFT_match_dir + "similar/"
used_SIFT_match_copy_dir = used_SIFT_match_dir + "copy/"
# 用来查看图片变换后效果的
dealt_image_dir = image_cache_dir + "dealt/"
dealt_similar_image_dir = dealt_image_dir + "similar/"
dealt_copy_image_dir = dealt_image_dir + "copy/"

# 训练好的模型储存路径
model_save_path = r'cache/saveModel.pkl'

pr_data_dir = "PRData/"


copy_flag = 1
similar_flag = 0
