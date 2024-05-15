from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

# 训练图片特征

# 设置数据路径
data_dir = 'D:/DATA1/MRN/MRN_train'
output_dir = 'D:/DATA1/MRN/model'

# 加载预训练的ResNet模型，去除顶层全连接层
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 编辑模型
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 定义特征提取函数
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return base_model.predict(x)

# 遍历图像文件夹并提取特征
features_list = []
for i, filename in enumerate(os.listdir(data_dir)):
    if filename.endswith('.png'):
        img_path = os.path.join(data_dir, filename)
        features = extract_features(img_path)
        features_list.append(features)

# 将特征列表转换为 numpy 数组
features_array = np.array(features_list)

# 保存.npy及.h5文件
np.save(os.path.join(output_dir, '1.npy'), features_array)
base_model.save(os.path.join(output_dir, '1.h5'))
