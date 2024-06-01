from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

# 废弃，准确率低
# 训练阳性图片特征

# 设置数据路径
data_dir = r'D:\DATA1\MRN\pos1'
output_dir = r'D:\DATA1\MRN\model'

# 加载预训练的ResNet模型
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 编辑模型
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 图像增强
datagen = image.ImageDataGenerator(
    rotation_range=10,  # 旋转角度范围
    width_shift_range=0.1,  # 水平平移范围
    height_shift_range=0.1,  # 垂直平移范围
    shear_range=0.1,  # 错切变换范围
    zoom_range=0.1,  # 缩放范围
    horizontal_flip=True,  # 左右翻转
    preprocessing_function=preprocess_input  # 预处理函数
)

# 定义特征提取函数
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(80, 80))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    generator = datagen.flow(x, batch_size=1)
    augmented_images = [next(generator)[0] for _ in range(1)]
    return base_model.predict(np.array(augmented_images))

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
np.save(os.path.join(output_dir, '2.npy'), features_array)
base_model.save(os.path.join(output_dir, '2.h5'))
