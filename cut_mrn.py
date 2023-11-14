import os
import numpy as np
import cv2
import tensorflow as tf
import pydicom

# 输入文件夹 A，包含DICOM文件的子文件夹
folder_a = r'D:\DATA1\MRN\MRN'

# 输入文件夹 B，包含类似的图像文件
# folder_b = r'D:\DATA1\MRN\MRNt'

# 输出文件夹 C
folder_c = r'D:\DATA1\MRN\MRNi'

# 加载已训练的模型
model = tf.keras.models.load_model(r'D:\DATA1\MRN\model\model1.h5')

# 定义目标图像尺寸
target_size = (128, 128)

# 创建空列表，用于保存提取的图像
extracted_images = []

# 定义参数以调整切割图像的数量
overlap = 0.2  # 重叠比例，控制切割区域的重叠程度
min_size = 64  # 切割区域的最小尺寸，避免切割得太小

# 遍历文件夹 A 中的所有子文件夹
for root, dirs, files in os.walk(folder_a):
    for filename in files:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(root, filename)
            ds = pydicom.dcmread(dicom_file)
            image_data = ds.pixel_array
            resized_image = cv2.resize(image_data, target_size)
            predicted_class = model.predict(np.expand_dims(resized_image, axis=0)).argmax()

            if predicted_class == 0:
                # 如果模型预测为类似的图像文件，保存图像到文件夹 C
                output_filename_prefix = os.path.join(folder_c, f'{os.path.basename(root)}_{filename}')

                # 获取切割区域的坐标
                x_center = resized_image.shape[1] // 2
                y_center = resized_image.shape[0] // 2
                half_size = min(resized_image.shape[0], resized_image.shape[1]) // 2
                x_start = max(0, x_center - half_size)
                x_end = min(resized_image.shape[1], x_center + half_size)
                y_start = max(0, y_center - half_size)
                y_end = min(resized_image.shape[0], y_center + half_size)

                # 切割图像
                step_size_x = int((1 - overlap) * (x_end - x_start))
                step_size_y = int((1 - overlap) * (y_end - y_start))

                for x in range(x_start, x_end - min_size, step_size_x):
                    for y in range(y_start, y_end - min_size, step_size_y):
                        x_slice = slice(x, x + min_size)
                        y_slice = slice(y, y + min_size)

                        # 获取切割后的图像
                        cut_image = resized_image[y_slice, x_slice]

                        # 保存切割后的图像为 128x128 大小
                        output_filename = f'{output_filename_prefix}_{x}_{y}.png'
                        cv2.imwrite(output_filename, cv2.resize(cut_image, target_size))
