import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from scipy.spatial.distance import cosine

# 设置数据路径
data_dir = r'D:\temp\pos1'
model_dir = r'D:\temp\model'

# 加载保存的模型和特征
base_model = ResNet50(weights=None, include_top=False, pooling='avg')
base_model.load_weights(os.path.join(model_dir, 'model.h5'))

# 打印模型结构以确保加载正确
base_model.summary()

model_features = np.load(os.path.join(model_dir, 'features.npy'))

# 定义特征提取函数
def extract_features(img_array):
    img_array = img_array.astype('float32')
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 滑动窗口参数进行验算
for root, dirs, files in os.walk(data_dir):
    if files:
        for filename in files:
            print("load:", filename)
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                original_img = image.load_img(image_path, target_size=(224, 224))  # 检查目标尺寸
                original_img_array = image.img_to_array(original_img)
                original_img_array = np.expand_dims(original_img_array, axis=0)

                # 打印处理后的图像数组
                print(f"Processed image array for {filename}: {original_img_array.shape}, {original_img_array.dtype}")

                # 提取特征
                window_features = extract_features(original_img_array)

                if np.all(window_features == 0):
                    print(f"Warning: extracted features are all zeros for {filename}")

                max_similarity = 0
                # 计算相似性并确定颜色
                for model_feature in model_features:
                    similarity = 1 - cosine(window_features.flatten(), model_feature.flatten())

                    # 更新最大相似度
                    max_similarity = max(max_similarity, similarity)

                print("Max similarity:", max_similarity)
                # print("window_features:", window_features)
                # print("model_feature:", model_features[0])
                #
                # # 打印计算的相似度和特征向量进行调试
                # for i in range(5):
                #    similarity = 1 - cosine(window_features.flatten(), model_features[i].flatten())
                #     print(f"Similarity with model feature {i}: {similarity}")

