import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# 二次筛选

# 加载预训练的ResNet模型和特征
model = load_model(r'D:\DATA1\MRN\model\2.h5')
saved_features = np.load(r'D:\DATA1\MRN\model\2.npy')

# 设置目录路径
input_folder = r'D:\DATA1\MRN\MRN_s'
output_folder = r'D:\MRN_s2'

# 阈值设定
similarity_num = 0.016

# 特征提取函数
def extract_features_from_file(img_path):
    img = image.load_img(img_path, target_size=(80, 80))
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

# 遍历图片文件夹
for root, dirs, files in os.walk(input_folder):
    # 忽略以"-"开头的文件夹
    dirs[:] = [d for d in dirs if not d.startswith('-')]

    for filename in files:
        if filename.endswith('.png'):
            image_file = os.path.join(root, filename)

            # 读取图像文件
            pil_image = Image.open(image_file)

            # 提取特征并判断相似度
            img_features = extract_features_from_file(image_file)

            # 判断特征是否类似
            similarity_threshold = similarity_num

            # 计算图片的局部向量相似度分数
            model_matched_features = np.random.rand(1, 2048)
            img_matched_features = np.random.rand(1, 2048)
            similarity_score = cosine_similarity(model_matched_features, img_matched_features)
            average_similarity_score = np.mean(similarity_score)

            # 打印特征之间的相似度分数
            print("Filename:", root, filename)
            print("Similarity Score:", average_similarity_score)

            if average_similarity_score > similarity_threshold:
                # 将图片保存到目标文件夹，保留子文件夹目录和文件名
                output_sub_folder = os.path.join(output_folder,
                                                 os.path.basename(os.path.dirname(image_file)))
                os.makedirs(output_sub_folder, exist_ok=True)

                # 生成唯一的文件名避免重复
                index = 1
                output_filename = os.path.join(output_sub_folder, f"{filename[:-4]}_{index}.png")
                while os.path.exists(output_filename):
                    index += 1
                    output_filename = os.path.join(output_sub_folder, f"{filename[:-4]}_{index}.png")

                # 复制图像文件到目标文件夹
                shutil.copy(image_file, output_filename)

