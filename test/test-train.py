import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# 设置数据路径
data_dir = r'D:\temp\pos1'
output_dir = r'D:\temp\model'

# 加载预训练的ResNet模型
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 打印模型结构以确保加载正确
base_model.summary()

# 定义特征提取函数
def extract_features(img_array):
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 遍历图像文件夹并提取特征
features_list = []
for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(data_dir, filename)
        img = image.load_img(img_path, target_size=(224, 224))  # 检查目标尺寸
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # 打印处理后的图像数组
        print(f"Processed image array for {filename}: {img_array.shape}, {img_array.dtype}")

        # 提取特征
        features = extract_features(img_array)
        print(f"Extracted features shape: {features.shape}")
        print(f"Extracted features sample: {features}")

        if np.all(features == 0):
            print(f"Warning: extracted features are all zeros for {filename}")
        features_list.append(features)

# 将特征列表转换为 numpy 数组
features_array = np.array(features_list)

# 保存.npy及.h5文件
np.save(os.path.join(output_dir, 'features.npy'), features_array)
base_model.save(os.path.join(output_dir, 'model.h5'))

# 打印一些特征向量以供调试
print("Sample features from training data:")
print(features_array[:5])
