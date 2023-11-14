import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import ProgbarLogger

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

# 初始化 TensorFlow 模型或使用自定义模型
model = tf.keras.Sequential([
    # 添加你自己的模型层
    tf.keras.layers.Flatten(input_shape=target_size + (1,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 创建 ProgbarLogger 回调
progbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

# 遍历文件夹 A 中的所有子文件夹
for root, dirs, files in os.walk(folder_a):
    for filename in files:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(root, filename)
            ds = pydicom.dcmread(dicom_file)

            # 检查是否包含有效的图像数据
            if hasattr(ds, 'pixel_array'):
                image_data = ds.pixel_array
                # Convert to grayscale if the image is not already
                if len(image_data.shape) > 2:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
                resized_image = cv2.resize(image_data, target_size)
                X_train.append(resized_image)
                Y_train.append(1)  # DICOM文件标记为1

    # 显示 DICOM 文件夹名和计数
    current_folder = os.path.basename(root)
    dicom_folder_count += 1
    print(f"\nProcessed {dicom_folder_count} DICOM folders. Current folder: {current_folder}\n")

# 遍历文件夹 B 中的所有图像文件
for root, dirs, files in os.walk(folder_b):
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_file = os.path.join(root, filename)
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, target_size)
            X_train.append(resized_image)
            Y_train.append(0)  # 图像文件标记为0

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# 灰阶额外维度
X_train = np.expand_dims(X_train, axis=-1)

# 训练模型
model.fit(X_train, Y_train, epochs=10, validation_split=0.2, callbacks=[progbar])

# 保存模型
model.save(r'D:\DATA1\MRN\model\model1.h5')
