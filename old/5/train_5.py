from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

# 数据增强参数
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 设置数据路径
data_dir = r'D:\DATA1\MRN\pos1'
output_dir = r'D:\DATA1\MRN\model'

# 增强数量
augmented_num = 5

# 加载预训练的ResNet模型
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 定义特征提取函数
def extract_features(img_array):
    img_array = img_array.astype('float32')
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 遍历图像文件夹并提取特征
features_list = []
for i, filename in enumerate(os.listdir(data_dir)):
    if filename.endswith('.png'):
        img_path = os.path.join(data_dir, filename)
        img = image.load_img(img_path, target_size=(80, 80))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # 使用数据增强生成器
        augmented_count = 0
        for augmented_img_array in datagen.flow(img_array, batch_size=1):
            features = extract_features(augmented_img_array)
            features_list.append(features)
            augmented_count += 1
            if augmented_count >= augmented_num:
                break

# 将特征列表转换为 numpy 数组
features_array = np.array(features_list)

# 保存.npy及.h5文件
np.save(os.path.join(output_dir, '5.npy'), features_array)
base_model.save(os.path.join(output_dir, '5.h5'))

# 打印一些特征向量以供调试
print("Sample features from training data:")
print(features_array[:5])
