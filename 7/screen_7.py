import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 路径
model_path = 'D:/temp/model/7.h5'
image_dir = 'D:/temp/1'
output_dir = 'D:/temp/2'
window_output_dir = 'D:/temp/windows'  # 保存小方块的目录

# 滑动窗口参数
window_size = (80, 80)  # 大小
step_size = 10  # 步长

# 标记点参数
point_size = 1  # 点的大小
start_color = np.array([0, 255, 0])  # BGR 绿色
end_color = np.array([0, 0, 255])  # BGR 红色

# 相似度阈值
similarity_threshold = 0.5

# 加载预训练的CNN模型
model = load_model(model_path)

# 创建保存小方块的目录
os.makedirs(window_output_dir, exist_ok=True)

# 滑动窗口参数验算
for root, dirs, files in os.walk(image_dir):
    if files:
        for filename in files:
            print("load:", filename)
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
                original_img = cv2.resize(original_img, (1280, 1280))  # 确保图像大小为1280x1280
                marked_img = original_img.copy()

                for y in range(0, 1280 - window_size[1] + 1, step_size):
                    for x in range(0, 1280 - window_size[0] + 1, step_size):
                        # 裁剪窗口
                        window = original_img[y:y + window_size[1], x:x + window_size[0]]
                        window = (window * 255).astype(np.uint8)

                        # 保存当前窗口的小方块
                        window_filename = f"{filename}_x{x}_y{y}.png"
                        window_path = os.path.join(window_output_dir, window_filename)
                        cv2.imwrite(window_path, window)

                        window = np.expand_dims(window, axis=0)  # 增加批量维度
                        window = np.expand_dims(window, axis=-1)  # 添加通道维度
                        window = window.astype(np.float32)  # 不再进行归一化

                        # 使用CNN模型预测窗口的相似性
                        similarity = model.predict(window)[0][0]
                        print(filename, similarity, x, y)

                        # 计算窗口中心点
                        center_x = x + window_size[0] // 2
                        center_y = y + window_size[1] // 2

                        # 判断相似性是否超过阈值
                        if similarity >= similarity_threshold:
                            ratio = (similarity - similarity_threshold) / (1 - similarity_threshold)
                            point_color = (1 - ratio) * start_color + ratio * end_color
                            point_color = tuple(map(int, point_color))

                            # 标记点
                            cv2.circle(marked_img, (center_x, center_y), point_size, point_color, -1)

                # 保存标记后的图像
                output_path = os.path.join(output_dir, os.path.relpath(root, image_dir))
                os.makedirs(output_path, exist_ok=True)
                cv2.imwrite(os.path.join(output_path, filename), marked_img)
                print(f"Marked image saved at {os.path.join(output_path, filename)}")
