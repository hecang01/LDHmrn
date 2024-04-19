import os
import numpy as np
import cv2
import pydicom
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 设置文件夹路径
folder_A_path = r'D:\DATA1\MRN\MRNt'
folder_B_path = r'D:\DATA1\MRN\MRN'
folder_C_path = r'D:\DATA1\MRN\model'

# 超参数
epochs = 10
batch_size = 8

# 加载文件夹A中的PNG图像，并转换为具有三个通道的灰度图像
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 将灰度图像转换为具有三个通道的图像
        images.append(img)
    return np.array(images)

# 从DICOM文件中提取图像，并转换为灰度图像
def extract_images_from_dicom(folder):
    images = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.dcm'):
                dcm_file = os.path.join(subdir, file)
                ds = pydicom.dcmread(dcm_file)
                if hasattr(ds, 'PixelData'):
                    pixel_array = ds.pixel_array
                    if pixel_array is not None:
                        # 调整图像大小
                        pixel_array = cv2.resize(pixel_array, (128, 128))
                        # 检查图像的通道数
                        if len(pixel_array.shape) == 3:  # 如果图像已经是三通道的，不需要转换
                            images.append(pixel_array)
                        else:  # 否则将图像转换为具有三个通道的格式
                            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
                            images.append(pixel_array)
    return np.array(images)

# 加载并处理数据
images_A = load_images_from_folder(folder_A_path)
images_B = extract_images_from_dicom(folder_B_path)

# 打印 images_A 和 images_B 的形状
print("Shape of images_A:", images_A.shape)
print("Shape of images_B:", images_B.shape)

# 构建训练集和测试集
X = np.concatenate((images_A, images_B), axis=0)
y = np.concatenate((np.zeros(len(images_A)), np.ones(len(images_B))), axis=0)

# 打印 X 和 y 的形状
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将图像形状转换为 (样本数, 128, 128, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# 构建ResNet模型
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# 在输入层之后添加一个通道数为 3 的扩展层，将灰度图像转换为具有三个通道的图像
input_layer = base_model.layers[0].input
expanded_input = input_layer
expanded_input = Reshape((128, 128, 3))(expanded_input)

x = base_model(expanded_input)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input_layer, outputs=predictions)

# 冻结ResNet的前几层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# 保存模型
model.save(os.path.join(folder_C_path, 'resnet_model.h5'))
