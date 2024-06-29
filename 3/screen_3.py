import os
import numpy as np
import pydicom
from PIL import Image
from tensorflow.keras.models import load_model

# 筛选保存为灰度图像

# 加载CNN模型
model = load_model(r'D:\DATA1\MRN\model\MRN_model.h5')

# 路径
input_dicom_folder = r'D:\DATA1\MRN\MRN'
output_folder = r'D:\temp\MRN_s'

# 预测阈值
prediction_num = 0.5

# 预测类别
def predict_image_class(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]  # 获取预测值
    return prediction

# 将灰度图像转换为三通道
def convert_to_rgb(img_array):
    return np.stack((img_array,) * 3, axis=-1)

# 窗宽和窗位
def apply_windowing(image, window_width, window_level):
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / (upper_bound - lower_bound) * 255
    return image

# 遍历DICOM文件夹
for root, dirs, files in os.walk(input_dicom_folder):
    dirs[:] = [d for d in dirs if not d.startswith('-')]
    for filename in files:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(root, filename)
            ds = pydicom.dcmread(dicom_file)
            protocol_name = ds.get('ProtocolName', '')
            if protocol_name in ['t2_de3d_we_cor_iso', 'PROSET']:
                print(dicom_file)
                if hasattr(ds, 'pixel_array'):
                    image_data = ds.pixel_array

                    # 默认窗宽窗位t2_de3d_we_cor_iso
                    if protocol_name == 't2_de3d_we_cor_iso':
                        if 'WindowWidth' in ds and 'WindowCenter' in ds:
                            window_width = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
                            window_level = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
                        else:
                            window_width = image_data.max() - image_data.min()
                            window_level = (image_data.max() + image_data.min()) / 2
                    # 窗宽窗位PROSET
                    elif protocol_name == 'PROSET':
                        window_width = 1500
                        window_level = 700

                    scaled_image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 255))
                    pil_image = Image.fromarray(scaled_image_data.astype(np.uint8), mode='L')
                    # 原始1024x1024的图像
                    original_image = np.array(pil_image.resize((1024, 1024), Image.LANCZOS))
                    # 创建256x256的副本
                    resized_image = np.array(pil_image.resize((256, 256), Image.LANCZOS))
                    resized_image_rgb = convert_to_rgb(resized_image)  # 转换为三通道
                    prediction = predict_image_class(resized_image_rgb)  # 预测
                    print("Filename:", filename)
                    print("Prediction:", prediction)
                    if prediction > prediction_num:
                        output_sub_folder = os.path.join(output_folder, os.path.basename(os.path.dirname(dicom_file)))
                        os.makedirs(output_sub_folder, exist_ok=True)
                        output_filename = os.path.join(output_sub_folder, f"{filename[:-4]}_1.png")
                        Image.fromarray(original_image.astype(np.uint8), mode='L').save(output_filename)
