import os
import numpy as np
import cv2
import pydicom
import tensorflow as tf
from tensorflow.keras.callbacks import ProgbarLogger

# 输入文件夹 A，包含DICOM文件的子文件夹
folder_a = r'D:\DATA1\MRN\MRN'

# 定义目标图像尺寸
target_size = (128, 128)

# 创建一个用于存储训练数据的列表
X_train = []
Y_train = []

# 记录处理的 DICOM 文件夹数量
dicom_folder_count = 0

# 超参数
epochs = 20
batch_size = 16

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
                    # 转成灰阶
                    if len(image_data.shape) > 2:
                        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
                    resized_image = cv2.resize(image_data, target_size)
                    X_train.append(resized_image)
                    Y_train.append(1)  # DICOM文件标记为1

    # 显示 DICOM 文件夹名和计数
    current_folder = os.path.basename(root)
    dicom_folder_count += 1
    print(f"\n Folder: {current_folder}, Total {dicom_folder_count}.\n")

# 将图像展平以适应 Dense 层
X_train_reshaped = np.array(X_train).reshape(len(X_train), 128, 128, 1)  # 调整图像形状
Y_train = np.array(Y_train)

# 初始化 TensorFlow 模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 创建 ProgbarLogger 回调
progbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

# 训练模型
model.fit(X_train_reshaped, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[progbar])

# 保存模型
model_dir = r'D:\DATA1\MRN\model'
model_name = f'model_cut_VGG16_{epochs}_{batch_size}.h5'

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
