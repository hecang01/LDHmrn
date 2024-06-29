import os
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# CNNs模型，处理灰度图像

# 设置数据路径
pos_data_dir = r'/mnt/d/temp/pos1'  # 阳性样本目录
neg_data_dir = r'/mnt/d/temp/neg1'  # 阴性样本目录
output_dir = r'/mnt/d/temp/model'

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
            img_array = np.repeat(img_array, 3, axis=-1)
            images.append(img_array)
            labels.append(1)
    # 加载阴性样本
    for filename in os.listdir(neg_data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(neg_data_dir, filename)
            img = image.load_img(img_path, target_size=(80, 80), color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array = np.repeat(img_array, 3, axis=-1)
            images.append(img_array)
            labels.append(0)
    images = np.array(images)
    labels = np.array(labels).reshape(-1, 1)  # 确保标签形状
    return images, labels

# 加载数据
images, labels = load_data(pos_data_dir, neg_data_dir)

# 分割数据为训练集、验证集、测试集
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 创建数据增强生成器
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Resnet50提高准确率
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

# 创建CNN模型
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # 减少特征图维度
    BatchNormalization(),  # 批量归一化
    Dense(256, activation='relu'),  # 增加神经元数量
    Dropout(0.5),  # Dropout层防止过拟合
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# 冻结预训练模型的层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
batch_size = 64
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size

# 训练模型（第一阶段）
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=steps_per_epoch,
          validation_data=(X_val, y_val),
          epochs=10,
          callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5),
                     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# 解冻部分预训练层进行微调
for layer in base_model.layers[:50]:
    layer.trainable = True

# 再次编译模型
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型（第二阶段）
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
          steps_per_epoch=steps_per_epoch,
          validation_data=(X_val, y_val),
          epochs=20,
          callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
                     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# 保存模型
model.save(os.path.join(output_dir, '7.h5'))
