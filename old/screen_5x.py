from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cosine

# 路径
model_dir = r'D:\DATA1\MRN\model'
image_dir = r'D:\1'
output_dir = r'D:\2'

# 滑动窗口参数
window_size = (80, 80)  # 大小
step_size = 20  # 步长

# 阈值
similarity_num = 0.8

# 特征模型和特征数据
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
known_features = np.load(os.path.join(model_dir, '5.npy'))

# 特征提取函数
def extract_features(img_array):
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 滑动窗口参数进行验算
for root, dirs, files in os.walk(image_dir):
    if files:
        results = []
        for filename in files:
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                original_img = image.load_img(image_path)
                original_img_array = image.img_to_array(original_img)
                original_height, original_width, _ = original_img_array.shape
                print("FIle:", filename)

                for y in range(0, original_height - window_size[1] + 1, step_size):
                    for x in range(0, original_width - window_size[0] + 1, step_size):
                        # 裁剪窗口
                        window = original_img_array[y:y + window_size[1], x:x + window_size[0]]
                        window = np.expand_dims(window, axis=0)

                        # 提取特征
                        window_features = extract_features(window)

                        # 与已知特征进行比较
                        for known_feature in known_features:
                            similarity = 1 - cosine(window_features, known_feature)
                            print("similarity:", similarity, x, y)
                            if similarity > similarity_num:  # 阈值
                                if results:
                                    output_path = os.path.join(output_dir, os.path.basename(root) + '.xlsx')

                                    # 创建DataFrame
                                    df = pd.DataFrame(results)

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
