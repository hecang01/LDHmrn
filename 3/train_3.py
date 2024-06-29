import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 使用CNNs训练。注意力集中在中轴

# 设置数据路径
pos_data_dir = 'D:/DATA1/MRN/MRN_train_pos'  # 阳性样本目录
neg_data_dir = 'D:/DATA1/MRN/MRN_train_neg'  # 阴性样本目录
output_dir = 'D:/temp/model'


# 数据预处理函数
def load_data(pos_data_dir, neg_data_dir):
    images = []
    labels = []
    # 加载阳性样本
    for filename in os.listdir(pos_data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(pos_data_dir, filename)
            img = image.load_img(img_path, target_size=(1024, 1024))
            resized_image = np.array(img.resize((256, 256), Image.LANCZOS))
            img_array = image.img_to_array(resized_image)
            images.append(img_array)
            labels.append(1)
    # 加载阴性样本
    for filename in os.listdir(neg_data_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(neg_data_dir, filename)
            img = image.load_img(img_path, target_size=(1024, 1024))
            resized_image = np.array(img.resize((256, 256), Image.LANCZOS))
            img_array = image.img_to_array(resized_image)
            images.append(img_array)
            labels.append(0)
    images = np.array(images)
    labels = np.array(labels).reshape(-1, 1)  # 确保标签形状
    return images, labels


# 加载数据
images, labels = load_data(pos_data_dir, neg_data_dir)


# 生成掩码
def create_center_mask(height, width, center_fraction=0.25):
    mask = np.zeros((height, width), dtype=np.float32)
    center_width = int(width * center_fraction)
    start = (width - center_width) // 2
    mask[:, start:start + center_width] = 1.0
    return mask


# 生成掩码
img_height, img_width = 256, 256
mask = create_center_mask(img_height, img_width)

# 扩展掩码维度以匹配输入
mask = np.expand_dims(mask, axis=0)  # (1, 256, 256)
mask = np.expand_dims(mask, axis=-1)  # (1, 256, 256, 1)
mask = np.tile(mask, (images.shape[0], 1, 1, 1))  # 将掩码应用到所有图像

# 分割数据为训练集和验证集
X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(images, labels, mask,
                                                                        test_size=0.2, random_state=42)

# 创建数据增强生成器
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False
)


# 自定义损失函数
def custom_loss(y_true, y_pred, mask):
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_loss = bce_loss * mask
    return tf.reduce_mean(weighted_loss)


# 自定义数据生成器以包含掩码
def data_generator(images, labels, masks, batch_size, datagen):
    gen = datagen.flow(images, labels, batch_size=batch_size)
    while True:
        img_batch, label_batch = gen.next()
        mask_batch = masks[:len(img_batch)]
        yield img_batch, label_batch, mask_batch


# 创建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
batch_size = 64
steps_per_epoch = len(X_train) // batch_size
validation_steps = len(X_val) // batch_size


# 定义训练步骤，包括掩码
@tf.function
def train_step(img_batch, label_batch, mask_batch):
    with tf.GradientTape() as tape:
        predictions = model(img_batch, training=True)
        loss = custom_loss(label_batch, predictions, mask_batch)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# 训练循环
for epoch in range(30):  # 训练30个epoch
    print(f"Epoch {epoch + 1}/30")
    for step in range(steps_per_epoch):
        img_batch, label_batch, mask_batch = next(data_generator(X_train, y_train, mask_train, batch_size, datagen))
        loss = train_step(img_batch, label_batch, mask_batch)
        print(f"Step {step + 1}/{steps_per_epoch}, Loss: {loss.numpy()}")

    # 验证步骤
    val_loss = 0
    for step in range(validation_steps):
        img_batch, label_batch, mask_batch = next(data_generator(X_val, y_val, mask_val, batch_size, datagen))
        val_predictions = model(img_batch, training=False)
        val_loss += custom_loss(label_batch, val_predictions, mask_batch).numpy()
    val_loss /= validation_steps
    print(f"Validation Loss: {val_loss}")

# 保存模型
model.save(os.path.join(output_dir, 'MRN_model.h5'))
