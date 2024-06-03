import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# 路径
data_dir = r'D:\temp\pos1'
model_dir = r'D:\temp\model'

# 加载模型和分类器
base_model = ResNet50(weights=None, include_top=False, pooling='avg')
cnn_model = load_model(os.path.join(model_dir, 'cnn_classifier.h5'))

# 特征提取函数
def extract_features(img_array):
    img_array = img_array.astype('float32')
    x = preprocess_input(img_array)
    return base_model.predict(x)

# 遍历图像文件夹并进行验证
y_true = []
y_pred = []
for filename in os.listdir(data_dir):
    if filename.endswith('.png'):
        image_path = os.path.join(data_dir, filename)
        original_img = image.load_img(image_path, target_size=(80, 80))
        original_img_array = image.img_to_array(original_img)
        original_img_array = np.expand_dims(original_img_array, axis=0)

        # 打印处理后的图像数组
        print(f"Processed image array for {filename}: {original_img_array.shape}, {original_img_array.dtype}")

        # 提取特征
        features = extract_features(original_img_array)
        features = features.reshape(1, 2048)

        # 使用CNN模型预测
        prediction = cnn_model.predict(features)
        predicted_label = np.argmax(prediction, axis=1)[0]

        y_true.append(1)
        y_pred.append(predicted_label)

# 计算验证准确率
accuracy = accuracy_score(y_true, y_pred)
print(f"Validation accuracy: {accuracy}")
