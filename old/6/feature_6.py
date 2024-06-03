import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# 路径
pos_dir = r'D:\temp\pos1'
neg_dir = r'D:\temp\neg1'
output_dir = r'D:\temp\model'

# 加载ResNet模型
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 定义特征提取函数
def extract_features(img_array):
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 文件夹中的特征和标签
def process_folder(folder_path, label):
    features_list = []
    labels_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(80, 80))  # 检查目标尺寸
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # 打印处理后的图像数组
            print(f"Processed image array for {filename}: {img_array.shape}, {img_array.dtype}")

            # 提取特征
            features = extract_features(img_array)
            print(f"Extracted features shape: {features.shape}")
            print(f"Extracted features sample: {features}")

            features_list.append(features.reshape(-1, 2048))
            labels_list.append(label)
    return features_list, labels_list

# 处理阳性和阴性文件夹
pos_features, pos_labels = process_folder(pos_dir, 1)
neg_features, neg_labels = process_folder(neg_dir, 0)

# 合并特征和标签
features_list = pos_features + neg_features
labels_list = pos_labels + neg_labels

# 将特征列表转换为 numpy 数组
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# 保存特征和标签
np.save(os.path.join(output_dir, 'features.npy'), features_array)
np.save(os.path.join(output_dir, 'labels.npy'), labels_array)

print("Feature extraction completed.")
