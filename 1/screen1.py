import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# 加载预训练的ResNet模型和特征
model = load_model(r'D:\DATA1\MRN\model\1.h5')
saved_features = np.load(r'D:\DATA1\MRN\model\1.npy')


# 设置目录路径
input_dir = r'D:\DATA1\MRN\MRN_cut'
output_dir = r'D:\DATA1\MRN\MRN_s'

# 定义特征提取函数
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

# 计算余弦相似度
def cosine_similarity(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# 遍历图像文件夹下的子文件夹
for root, dirs, files in os.walk(input_dir):
    for dir_name in dirs:
        sub_input_dir = os.path.join(root, dir_name)
        sub_output_dir = os.path.join(output_dir, dir_name)

        # 创建对应的子文件夹
        os.makedirs(sub_output_dir, exist_ok=True)

        # 遍历子文件夹内的图片
        for filename in os.listdir(sub_input_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(sub_input_dir, filename)

                # 提取特征
                img_features = extract_features(img_path)

                # 判断特征是否类似
                similarity_threshold = 0.016

                # 计算整体图片的相似度分数
                similarity_score = cosine_similarity(saved_features, img_features)
                average_similarity_score = np.mean(cosine_similarity(saved_features, img_features))

                # 打印特征之间的相似度分数
                print("Filename:", filename)
                print("Similarity Score:", average_similarity_score)

                if average_similarity_score > similarity_threshold:
                    # 将图片保存到目标文件夹，保留子文件夹目录和文件名
                    shutil.copy(img_path, os.path.join(sub_output_dir, filename))
