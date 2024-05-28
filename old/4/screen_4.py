from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cosine

# 设置路径
data_dir = r'D:\DATA1\MRN\pos1'
model_dir = r'D:\DATA1\MRN\model'
image_dir = r'D:\DATA1\MRN\MRN_s'
output_dir = r'D:\DATA1\MRN\results'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 滑动窗口参数
window_size = (50, 50)  # 大小
step_size = 5  # 步长

# 加载特征模型和特征数据
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
known_features = np.load(os.path.join(model_dir, '5.npy'))

# 定义特征提取函数
def extract_features(img_array):
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 遍历图像文件夹并提取特征
for root, dirs, files in os.walk(image_dir):
    for filename in files:
        if filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            original_img = image.load_img(image_path)
            original_img_array = image.img_to_array(original_img)
            original_height, original_width, _ = original_img_array.shape

            results = []

            for y in range(0, original_height - window_size[1] + 1, step_size):
                for x in range(0, original_width - window_size[0] + 1, step_size):
                    # 裁剪窗口
                    window = original_img_array[y:y + window_size[1], x:x + window_size[0]]
                    window = np.expand_dims(window, axis=0)

                    # 提取特征
                    window_features = extract_features(window)

                    # 与已知特征进行比较
                    for known_feature in known_features:
                        similarity = 1 - cosine(window_features.flatten(), known_feature.flatten())
                        if similarity > 0.8:  # 可以根据需求调整相似度阈值
                            results.append({'x': x, 'y': y, 'similarity': similarity})
                            break  # 如果只需要记录一个匹配

            # 将结果保存到 Excel 文件
            if results:
                output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.xlsx')

                # 创建DataFrame
                df = pd.DataFrame(results)

                # 示例数据：Name列的生成，实际情况请根据需要调整
                df['Name'] = 'name_h_w'
                split_names = df['Name'].str.split('_', expand=True)
                df['name'] = split_names[0]
                df['h'] = split_names[1] if split_names.shape[1] > 1 else ''
                df['w'] = split_names[2] if split_names.shape[1] > 2 else ''

                # 重新排列列顺序，将新的列放在前面
                df = df[['name', 'h', 'w', 'x', 'y', 'similarity']]

                # 保存到Excel文件
                df.to_excel(output_path, index=False)

                print(f"Results saved in {output_path}")
