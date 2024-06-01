import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 小窗验证，废弃，加入阴性训练
# 文件目录
train_dir = r'D:\DATA1\MRN\MRN_s'
augmentation_dir = r'D:\DATA1\MRN\pos1'

# 数据增强
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_images = []
train_labels = []
# 使用 os.walk() 遍历整个目录树
for root, dirs, files in os.walk(train_dir):
    # 过滤子文件夹，只保留名称以1或2结尾的子文件夹
    dirs[:] = [d for d in dirs if d.endswith('1') or d.endswith('2')]
    for filename in files:
        if filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            image = Image.open(image_path).convert('L')
            image = image.resize((80, 80))  # 将图像调整为指定大小
            image = np.array(image) / 255.0  # 归一化像素值到 [0, 1] 范围内
            train_images.append(image)
            label = 1 if 'pos' in root else 0
            train_labels.append(label)

# 将列表转换为 numpy 数组
train_images = np.array(train_images)
train_images = np.expand_dims(train_images, axis=-1)
train_labels = np.array(train_labels)

print("Train images:", len(train_images))

# 特征文件夹数据增强
augmentation_images = []
augmentation_labels = []

for filename in os.listdir(augmentation_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(augmentation_dir, filename)
        image = Image.open(image_path).convert('L')
        image = image.resize((80, 80))  # 将图像调整为指定大小
        image = np.array(image) / 255.0  # 归一化像素值到 [0, 1] 范围内
        augmentation_images.append(image)
        augmentation_labels.append(1)

# 将列表转换为 numpy 数组
augmentation_images = np.array(augmentation_images)
augmentation_images = np.expand_dims(augmentation_images, axis=-1)
augmentation_labels = np.array(augmentation_labels)

print("Augmentation images:", len(augmentation_images))

train_generator = datagen.flow(train_images, train_labels, shuffle=True)
augmentation_generator = datagen.flow(augmentation_images, augmentation_labels, shuffle=True)

# 构建模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_generator, epochs=20, validation_data=augmentation_generator)

# 保存模型
model.save(r'D:\DATA1\MRN\model\4.h5')
