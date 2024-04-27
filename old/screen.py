import os
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import regularizers
import shutil

# 设置文件夹路径
source_folder = r'D:\DATA1\MRN\MRN_cut'
standard_folder = r'D:\DATA1\MRN\MRN_train'
destination_folder = r'D:\MRN_s'

# 设置参数
prediction_rate = 0.95
epochs = 20
batch_size = 32

# 加载PNG图像并将其转换为数组
def load_image(path):
    img = image.load_img(path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    return img_array

# 构建更复杂的CNN模型
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))  # 新增一层卷积层
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  # 增加全连接层的神经元数量
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))  # 增加全连接层的神经元数量
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 输出层使用softmax激活函数，输出两个类别的概率
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 加载标准文件夹内的PNG图像并生成特征向量
def load_and_extract_features(standard_folder):
    images = []
    labels = []
    for root, _, files in os.walk(standard_folder):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                img_array = load_image(img_path)
                images.append(img_array)
                labels.append(1)  # 将标签改为1，表示符合标准
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# 加载源文件夹内的PNG图像并预测是否符合标准
def predict_images(model, source_folder, destination_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".png"):
                img_path = os.path.join(root, file)
                img_array = load_image(img_path)
                img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
                prediction = model.predict(img_array)
                if np.argmax(prediction) == 1:  # 判断预测概率最高的类别是否为1，表示符合标准
                    relative_path = os.path.relpath(img_path, source_folder)
                    destination_path = os.path.join(destination_folder, relative_path)
                    destination_subfolder = os.path.dirname(destination_path)
                    os.makedirs(destination_subfolder, exist_ok=True)
                    shutil.copy(img_path, destination_path)

# 加载数据并训练模型
standard_images, standard_labels = load_and_extract_features(standard_folder)
model = build_model()
model.fit(standard_images, standard_labels, epochs=epochs, batch_size=batch_size)

# 使用训练好的模型预测第一个文件夹内的PNG图像是否符合标准，并移动符合标准的图像到目标文件夹
predict_images(model, source_folder, destination_folder)
