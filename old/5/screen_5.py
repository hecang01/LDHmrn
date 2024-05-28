from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import cv2
from scipy.spatial.distance import cosine

# 路径
model_dir = r'D:\DATA1\MRN\model'
image_dir = r'D:\temp\1'
output_dir = r'D:\temp\2'

# 滑动窗口参数
window_size = (120, 120)  # 大小
step_size = 30  # 步长

# 标记点参数
point_size = 1  # 点的大小
start_color = np.array([0, 255, 0])  # BGR 绿色
end_color = np.array([0, 0, 255])  # BGR 红色

# 相似度大小
similarity_min = 0.0
similarity_max = 0.9

# 特征模型和特征数据
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_features = np.load(os.path.join(model_dir, '5.npy'))

# 特征提取函数
def extract_features(img_array):
    img_array = img_array.astype('float32')
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 滑动窗口参数进行验算
for root, dirs, files in os.walk(image_dir):
    if files:
        for filename in files:
            print("load:", filename)
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                original_img = image.load_img(image_path)
                original_img_array = image.img_to_array(original_img)
                original_height, original_width, _ = original_img_array.shape
                marked_img = original_img_array.copy()

                for y in range(0, original_height - window_size[1] + 1, step_size):
                    for x in range(0, original_width - window_size[0] + 1, step_size):
                        # 裁剪窗口
                        window = original_img_array[y:y + window_size[1], x:x + window_size[0]]
                        window = np.expand_dims(window, axis=0)

                        # 提取特征
                        window_features = extract_features(window)

                        # 计算窗口中心点
                        center_x = x + window_size[0] // 2
                        center_y = y + window_size[1] // 2

                        max_similarity = 0
                        # 计算相似性并确定颜色
                        for model_feature in model_features:
                            similarity = 1 - cosine(window_features.flatten(), model_feature.flatten())
                            # 更新最大相似度
                            max_similarity = max(max_similarity, similarity)

                        print("Max similarity:", max_similarity, x, y)

                        if similarity_min <= max_similarity <= similarity_max:
                            ratio = (max_similarity - similarity_min) / (similarity_max - similarity_min)
                            point_color = (1 - ratio) * start_color + ratio * end_color
                            point_color = tuple(map(int, point_color))

                            # 标记点
                            cv2.circle(marked_img, (center_x, center_y), point_size, point_color, -1)

                # 保存标记后的图像
                marked_img = cv2.cvtColor(marked_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                output_path = os.path.join(output_dir, os.path.relpath(root, image_dir))
                os.makedirs(output_path, exist_ok=True)
                cv2.imwrite(os.path.join(output_path, filename), marked_img)
                print(f"Marked image saved at {os.path.join(output_path, filename)}")
