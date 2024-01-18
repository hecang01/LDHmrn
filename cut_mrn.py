import os
import cv2
import pydicom
import numpy as np
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model(r'D:\DATA1\MRN\model\model_cut.h5')

# 定义文件夹A，包含DICOM文件的子文件夹
folder_a = r'D:\DATA1\MRN\MRN'

# 定义输出文件夹C，用于保存提取的图像
folder_c = r'D:\DATA1\MRN\MRNi'


# 函数：从DICOM图像中选择适当的正方形
def select_square(image, size=128):
    height, width = image.shape
    # 选择图像中心位置开始提取
    start_row = max(0, (height - size) // 2)
    start_col = max(0, (width - size) // 2)
    square = image[start_row:start_row + size, start_col:start_col + size]
    return square


# 遍历文件夹A中的DICOM文件
for root, dirs, files in os.walk(folder_a):
    for filename in files:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(root, filename)
            ds = pydicom.dcmread(dicom_file)

            # 检查DICOM文件是否包含有效的图像数据
            if hasattr(ds, 'pixel_array'):
                image_data = ds.pixel_array
                # 如果图像不是灰度图，则转换为灰度图
                if len(image_data.shape) > 2:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)

                # 对DICOM图像进行模型预测
                input_data = np.expand_dims(np.expand_dims(image_data, axis=-1), axis=0)
                prediction = model.predict(input_data)[0]

                # 如果预测为DICOM文件（类别为1）
                if np.argmax(prediction) == 1:
                    # 从DICOM图像中选择适当的正方形
                    square = select_square(image_data)

                    # 调整正方形大小为128x128像素
                    resized_square = cv2.resize(square, (128, 128))

                    # 保存图像到输出文件夹，使用唯一的名称
                    output_filename = f"{os.path.basename(root)}_{filename[:-4]}.png"
                    output_path = os.path.join(folder_c, output_filename)
                    cv2.imwrite(output_path, resized_square)


print("提取和分类完成。图像保存到:", folder_c)
