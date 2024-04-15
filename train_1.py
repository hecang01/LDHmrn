import os
import numpy as np
import cv2
import pydicom
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# 设置文件夹路径
folder_png_path = r'D:\DATA1\MRN\MRNt'
folder_dicom_path = r'D:\DATA1\MRN\MRN'
folder_model_path = r'D:\DATA1\MRN\model'

# 加载文件夹png中的PNG图像，并转换为灰度图像
def load_png(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        images.append(img)
    return np.array(images)

# 从DICOM文件中提取图像，并转换为灰度图像
def load_dicom(folder):
    images = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.dcm'):
                dcm_file = os.path.join(subdir, file)
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'PixelData'):
                    pixel_array = ds.pixel_array
                    if pixel_array is not None:
                        pixel_array = cv2.resize(pixel_array, (128, 128))
                        images.append(pixel_array)
    return np.array(images)

# 加载并处理数据
images_A = load_png(folder_png_path)
images_B = load_dicom(folder_dicom_path)

# 构建训练集和测试集
X = np.concatenate((images_A, images_B), axis=0)
y = np.concatenate((np.zeros(len(images_A)), np.ones(len(images_B))), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将图像形状转换为 (样本数, 128, 128, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 构建ResNet模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 1))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 冻结ResNet的前几层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 保存模型
model.save(os.path.join(folder_model_path, 'resnet_model.h5'))
