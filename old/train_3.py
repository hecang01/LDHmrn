import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# 数据集路径
data_dir = r'D:\DATA1\MRN\MRN_s'
mask_dir = r'D:\DATA1\MRN\pos1'
save_dir = r'D:\DATA1\MRN\model'

# 保存特征
def save_features(features, save_dir):
    filename = '3.npy'
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, features)

# 定义数据加载器
def load_data(image_path, mask_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0

    mask = tf.keras.preprocessing.image.load_img(mask_path)
    mask = tf.keras.preprocessing.image.img_to_array(mask)
    mask = mask / 255.0

    return image, mask

# 定义数据增强器
datagen = ImageDataGenerator(
    rotation_range=10,  # 旋转角度范围
    width_shift_range=0.1,  # 水平平移范围
    height_shift_range=0.1,  # 垂直平移范围
    shear_range=0.1,  # 错切变换范围
    zoom_range=0.1,  # 缩放范围
    horizontal_flip=True,  # 左右翻转
    fill_mode='nearest'
)

# 定义残差网络架构
def create_model():
    inputs = layers.Input(shape=(256, 256, 3))

    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.MaxPooling2D((2, 2))(conv1)

    shortcut1 = conv1

    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.MaxPooling2D((2, 2))(conv2)

    shortcut2 = conv2

    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.MaxPooling2D((2, 2))(conv3)

    resnet = layers.add([conv3, shortcut2])
    resnet = layers.Conv2D(64, (1, 1), activation='relu')(resnet)

    resnet = layers.add([resnet, shortcut1])
    resnet = layers.Conv2D(32, (1, 1), activation='relu')(resnet)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(resnet)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 训练模型
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 进行训练
train_generator = datagen.flow_from_directory(data_dir, class_mode='binary', batch_size=32)

for epoch in range(10):
    for image_batch, mask_batch in train_generator:
        model.train_on_batch(image_batch, mask_batch)

# 保存模型和特征
model.save(os.path.join(save_dir, 'model.h5'))

# 提取特征
for image_path in os.listdir(data_dir):
    image, _ = load_data(os.path.join(data_dir, image_path), None)
    image = image[np.newaxis, ...]

    features = model.predict(image)
    save_features(features, save_dir)
