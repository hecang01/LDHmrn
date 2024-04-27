# 模型全黑

import os
import cv2
import numpy as np
import pydicom
from tensorflow.keras.callbacks import ProgbarLogger
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, \
    MaxPooling2D, Concatenate
from tensorflow.keras.models import Model

# 输入文件夹 A，包含DICOM文件的子文件夹
folder_a = r'D:\DATA1\MRN\MRN'

# 输入文件夹 B，包含类似的图像文件
folder_b = r'D:\DATA1\MRN\MRN_train'

# 定义目标图像尺寸
target_size = (128, 128)

# 定义保存缩放比例和截取位置信息的列表
scale_factors = []
start_rows = []
start_cols = []

# 创建一个用于存储训练数据的列表
X_train = []
Y_train = []

# 记录处理的 DICOM 文件夹数量
dicom_folder_count = 0

# 超参数
epochs = 20
batch_size = 12

def resnet_block(inputs, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([x, inputs])
    return x


def resnet_model(input_size=(128, 128, 1), num_classes=1):
    inputs = Input(input_size)

    # 第一个卷积层
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # 添加残差块
    x = resnet_block(x, filters=64)
    x = resnet_block(x, filters=64)

    # 添加全局平均池化层
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层，输出为类别数量
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 初始化ResNet模型
model = resnet_model(input_size=target_size + (1,))

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

            # 选取指定序列
            protocol_name = ds.get('ProtocolName', '')
            if protocol_name in ['t2_de3d_we_cor_iso', 'PROSET']:
                # 检查是否包含有效的图像数据
                if hasattr(ds, 'pixel_array'):
                    image_data = ds.pixel_array

                    # 计算缩放比例
                    original_height, original_width = image_data.shape
                    scale_factor = target_size[0] / max(original_height, original_width)

                    # 缩放图像
                    resized_image = cv2.resize(image_data, None, fx=scale_factor, fy=scale_factor)

                    # 如果图像是多通道的，转换为灰度图
                    if len(resized_image.shape) > 2:
                        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

                    # 将图像数据添加到训练集中
                    X_train.append(resized_image)
                    Y_train.append(1)  # DICOM文件标记为1

                    # 保存缩放比例和截取位置信息
                    scale_factors.append(scale_factor)
                    start_rows.append(max(0, (resized_image.shape[0] - target_size[0]) // 2))
                    start_cols.append(max(0, (resized_image.shape[1] - target_size[1]) // 2))

    # 显示 DICOM 文件夹名和计数
    current_folder = os.path.basename(root)
    dicom_folder_count += 1
    print(f"\n Folder: {current_folder}, Total {dicom_folder_count}.\n")

# 遍历文件夹 B 中的所有图像文件
for root, dirs, files in os.walk(folder_b):
    for filename in files:
        if filename.endswith('.png'):
            image_file = os.path.join(root, filename)
            # 以彩色模式读取图像
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, target_size)
            X_train.append(resized_image)
            Y_train.append(0)  # 图像文件标记为0

# 转换为 numpy 数组
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# 灰阶额外维度
X_train = np.expand_dims(X_train, axis=-1)

# 划分训练集和验证集
validation_split = 0.2
split_index = int((1 - validation_split) * len(X_train))
X_train, X_val = X_train[:split_index], X_train[split_index:]
Y_train, Y_val = Y_train[:split_index], Y_train[split_index:]

# 训练模型
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[progbar],
          sample_weight=[scale_factors[:split_index], start_rows[:split_index], start_cols[:split_index]])

# 保存模型
model_dir = r'D:\DATA1\MRN\model'
model_name = f'model_cut_ResNet_{epochs}_{batch_size}.h5'

# 检查是否已存在同名文件
if os.path.exists(os.path.join(model_dir, model_name)):
    # 寻找可用的文件名
    i = 1
    while os.path.exists(os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}_{i}.h5")):
        i += 1
    # 添加序号并保存模型
    model.save(os.path.join(model_dir, f"{os.path.splitext(model_name)[0]}_{i}.h5"))
    print(f"Saved: {os.path.join(model_dir, f'{os.path.splitext(model_name)[0]}_{i}.h5')}")
else:
    # 直接保存模型
    model.save(os.path.join(model_dir, model_name))
    print(f"Saved: {os.path.join(model_dir, model_name)}")

print("\n Done.")
