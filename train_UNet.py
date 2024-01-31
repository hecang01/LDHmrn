import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.utils import to_categorical

# 输入文件夹 A，包含DICOM文件的子文件夹
folder_a = r'D:\DATA1\MRN\MRN'

# 输入文件夹 B，包含类似的图像文件
folder_b = r'D:\DATA1\MRN\MRNt'

# 定义目标图像尺寸
target_size = (128, 128)

# 创建一个用于存储训练数据的列表
X_train = []
Y_train = []

# 记录处理的 DICOM 文件夹数量
dicom_folder_count = 0


def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # 编码器部分
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 中间部分
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    # 解码器部分
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    # 输出层
    outputs = Conv2D(2, 1, activation='softmax')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# 初始化UNet模型
model = unet_model(input_size=target_size + (1,))

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 创建 ProgbarLogger 回调
progbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

# 遍历文件夹 A 中的所有子文件夹
for root, dirs, files in os.walk(folder_a):
    # 忽略以"-"开头的文件夹
    dirs[:] = [d for d in dirs if not d.startswith('-')]

    for filename in files:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(root, filename)
            ds = pydicom.dcmread(dicom_file)

            # 检查是否包含有效的图像数据
            if hasattr(ds, 'pixel_array'):
                image_data = ds.pixel_array
                # 转成灰阶
                if len(image_data.shape) > 2:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
                resized_image = cv2.resize(image_data, target_size)
                X_train.append(resized_image)
                Y_train.append(1)  # DICOM文件标记为1

    # 显示 DICOM 文件夹名和计数
    current_folder = os.path.basename(root)
    dicom_folder_count += 1
    print(f"\n Folder: {current_folder}, Total {dicom_folder_count} .\n")

# 遍历文件夹 B 中的所有图像文件
for root, dirs, files in os.walk(folder_b):
    for filename in files:
        if filename.endswith('.png'):
            image_file = os.path.join(root, filename)
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, target_size)
            X_train.append(resized_image)
            Y_train.append(0)  # 图像文件标记为0

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# 灰阶额外维度
X_train = np.expand_dims(X_train, axis=-1)

# 划分训练集和验证集
validation_split = 0.2
split_index = int((1 - validation_split) * len(X_train))
X_train, X_val = X_train[:split_index], X_train[split_index:]
Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

# 将标签数据进行独热编码
Y_train_encoded = to_categorical(Y_train, num_classes=2)
Y_val_encoded = to_categorical(Y_val, num_classes=2)

# 训练模型
model.fit(X_train, Y_train_encoded, epochs=10, batch_size=8,
          validation_data=(X_val, Y_val_encoded), callbacks=[progbar])

# 保存模型
model.save(r'D:\DATA1\MRN\model\model_cut.h5')