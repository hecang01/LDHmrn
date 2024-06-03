import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNNs模型，处理灰度图像

# 设置数据路径
pos_data_dir = r'D:/temp/pos1'  # 阳性样本目录
neg_data_dir = r'D:/temp/neg1'  # 阴性样本目录
output_dir = r'D:/temp/model'

# 数据预处理函数
def load_data(pos_data_dir, neg_data_dir):
    images = []
    labels = []
    # 加载阳性样本
    for filename in os.listdir(pos_data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(pos_data_dir, filename)
            img = image.load_img(img_path, target_size=(80, 80), color_mode="grayscale")
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(1)
    # 加载阴性样本
    for filename in os.listdir(neg_data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(neg_data_dir, filename)
            img = image.load_img(img_path, target_size=(80, 80), color_mode="grayscale")
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(0)
    images = np.array(images)
    labels = np.array(labels).reshape(-1, 1)  # 确保标签形状
    return images, labels

# 加载数据
images, labels = load_data(pos_data_dir, neg_data_dir)

# 分割数据为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 创建数据增强生成器
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)

# 创建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
batch_size = 64
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size

# 训练循环
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=steps_per_epoch,
          validation_data=(X_val, y_val),
          epochs=30)

# 保存模型
model.save(os.path.join(output_dir, '7.h5'))
