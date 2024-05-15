import os
import pydicom
import numpy as np
from PIL import Image

# 切割原始的dicom文件为多个128*128的图片

# 指定输入和输出文件夹路径
input_folder = r'D:\DATA1\MRN\MRN'
output_folder = r'D:\DATA1\MRN\MRN_cut'

def resize_and_save_as_png(dicom_path, output_path):
    # 读取Dicom文件
    ds = pydicom.dcmread(dicom_path)

    # 获取图像数据
    image_data = ds.pixel_array

    # 将图像数据重新缩放到0到255的范围内
    scaled_image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 255))

    # 将图像转换为灰度模式（模式为'L'）
    pil_image = Image.fromarray(scaled_image_data.astype(np.uint8), mode='L')

    # 缩放图像到500*500
    resized_image = np.array(pil_image.resize((500, 500), Image.LANCZOS))

    # 获取子文件夹名
    folder_name = os.path.basename(os.path.dirname(dicom_path))

    # 确保输出文件夹存在
    os.makedirs(os.path.join(output_path, folder_name), exist_ok=True)

    # 切割并保存图像
    for i in range(200, 350, 25):
        # 截取正方形图像
        cropped_image = resized_image[i-64:i+64, 186:314,]

        # 创建输出文件名
        index = 1
        output_filename = os.path.join(output_path, folder_name, f"{index}_{folder_name}_{i}.png")

        # 如果输出文件已存在，则添加序号
        while os.path.exists(output_filename):
            output_filename = os.path.join(output_path, folder_name, f"{index}_{folder_name}_{i}.png")
            index += 1

        # 保存图像为PNG文件
        Image.fromarray(cropped_image.astype(np.uint8)).save(output_filename)

def process_dicom_folders(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        # 忽略以"-"开头的文件夹
        dirs[:] = [d for d in dirs if not d.startswith('-')]

        for filename in files:
            if filename.endswith('.dcm'):
                dicom_file = os.path.join(root, filename)
                ds = pydicom.dcmread(dicom_file)

                # 选取指定序列
                protocol_name = ds.get('ProtocolName', '')
                if protocol_name in ['t2_de3d_we_cor_iso', 'PROSET']:
                    print(dicom_file),
                    # 检查是否包含有效的图像数据
                    if hasattr(ds, 'pixel_array'):
                        # 处理图像并保存为PNG
                        resize_and_save_as_png(dicom_file, output_folder)

# 处理Dicom文件夹
process_dicom_folders(input_folder, output_folder)
