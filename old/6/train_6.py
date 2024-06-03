import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 重写特征版本，准确率低，重置

# 设置数据路径
model_dir = r'D:\temp\model'

# 加载特征和标签
features_array = np.load(os.path.join(model_dir, 'features.npy'))
labels_array = np.load(os.path.join(model_dir, 'labels.npy'))

# 重塑特征数组
features_array = features_array.reshape((features_array.shape[0], 2048))

# 将标签转换为分类格式
labels_array = to_categorical(labels_array, num_classes=2)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)

# 构建CNN模型
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 编译模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# 保存模型
model.save(os.path.join(model_dir, 'cnn_classifier.h5'))

print("CNN training completed.")
