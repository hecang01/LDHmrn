import os
import numpy as np
import openpyxl
import pydicom
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# 加载预训练的ResNet模型和特征
model = load_model(r'D:\DATA1\MRN\model\1.h5')
saved_features = np.load(r'D:\DATA1\MRN\model\1.npy')

# 设置目录路径
input_dicom_folder = r'D:\DATA1\MRN\MRN'
output_folder = r'D:\MRN_s'

# 阈值设定
similarity_num = 0.016

# 错误列表
wb = openpyxl.Workbook()
ws = wb.active
ws.append(["Filename", "Error Message"])

# 定义特征提取函数
def extract_features_from_file(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

# 计算余弦相似度
def cosine_similarity(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# 遍历DICOM文件夹
for root, dirs, files in os.walk(input_dicom_folder):
    # 忽略以"-"开头的文件夹
    dirs[:] = [d for d in dirs if not d.startswith('-')]

    for filename in files:
        if filename.endswith('.dcm'):
            dicom_file = os.path.join(root, filename)
            ds = pydicom.dcmread(dicom_file)

            # 选取指定序列
            protocol_name = ds.get('ProtocolName', '')
            if protocol_name in ['t2_de3d_we_cor_iso', 'PROSET']:
                print(dicom_file)
                # 检查是否包含有效的图像数据
                if hasattr(ds, 'pixel_array'):
                    # 获取图像数据
                    image_data = ds.pixel_array

                    # 将图像数据重新缩放到0到255的范围内
                    scaled_image_data = np.interp(image_data, (image_data.min(), image_data.max()), (0, 255))

                    # 将图像转换为灰度模式（模式为'L'）
                    pil_image = Image.fromarray(scaled_image_data.astype(np.uint8), mode='L')

                    # 缩放图像到500*500
                    resized_image = np.array(pil_image.resize((500, 500), Image.LANCZOS))

                    # 标志是否找到符合条件的图像区域
                    found = False

                    # 切割并判断图像是否符合特征
                    for i in range(200, 350, 25):
                        for j in range(200, 350, 25):
                            # 截取正方形图像
                            cropped_image = resized_image[i - 64:i + 64, j - 64:j + 64]

                            try:
                                # 将截取后的图像保存为临时文件
                                temp_img_path = os.path.join(output_folder, 'temp.png')
                                Image.fromarray(cropped_image.astype(np.uint8)).save(temp_img_path)

                                # 提取特征并判断相似度
                                img_features = extract_features_from_file(temp_img_path)
                                similarity_score = cosine_similarity(saved_features, img_features)
                                average_similarity_score = np.mean(similarity_score)

                                print("Filename:", filename)
                                print("Similarity Score:", average_similarity_score)

                                if average_similarity_score > similarity_num:
                                    # 如果找到符合条件的图像区域，缩放并保存整个原始图像
                                    found = True
                                    output_sub_folder = os.path.join(output_folder,
                                                                     os.path.basename(os.path.dirname(dicom_file)))
                                    os.makedirs(output_sub_folder, exist_ok=True)

                                    # 缩放整个原始图像到1280*1280
                                    large_resized_image = np.array(pil_image.resize((1280, 1280), Image.LANCZOS))
                                    output_filename = os.path.join(output_sub_folder, f"{filename[:-4]}_1280x1280.png")
                                    Image.fromarray(large_resized_image.astype(np.uint8)).save(output_filename)
                                    break

                            except Exception as e:
                                print(f"Error processing file: {filename}")
                                print(f"Error message: {str(e)}")
                                ws.append([filename, str(e)])

                        if found:
                            break  # 跳过并处理下一个文件

# 保存Excel文件
wb.save(r'D:\MRN_s\error.xlsx')
