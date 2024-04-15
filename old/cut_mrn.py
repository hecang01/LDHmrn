import os
import cv2
import pydicom
import numpy as np
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model(r'D:\DATA1\MRN\model\model_cut_VGG16_20_16.h5')

# 定义文件夹A，包含DICOM文件的子文件夹
folder_a = r'D:\DATA1\MRN\MRN'

# 定义输出文件夹C，用于保存提取的图像
folder_c = r'D:\DATA1\MRN\MRNi'

# 定义目标图像尺寸
target_size = (128, 128)


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
    # 忽略以"-"开头的文件夹
    dirs[:] = [d for d in dirs if not d.startswith('-')]

    # 输出当前文件夹名
    print("当前文件夹:", os.path.basename(root))

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

                    # 调整DICOM图像大小为模型期望的大小
                    resized_image = cv2.resize(image_data, target_size)

                    # 对DICOM图像进行模型预测
                    input_data = np.expand_dims(np.expand_dims(resized_image, axis=-1), axis=0)
                    prediction = model.predict(input_data)[0]

                    # 如果预测为DICOM文件（类别为1）
                    if np.argmax(prediction) == 1:
                        # 从DICOM图像中选择适当的正方形
                        square = select_square(resized_image)

                        # 调整正方形大小为128x128像素
                        resized_square = cv2.resize(square, target_size)

                        # 保存图像到输出文件夹，使用唯一的名称
                        output_filename = f"{os.path.basename(root)}_{filename[:-4]}.png"
                        output_path = os.path.join(folder_c, output_filename)
                        cv2.imwrite(output_path, resized_square)

                        print(f"保存图像: {output_filename}")
