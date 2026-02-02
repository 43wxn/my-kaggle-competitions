'''import zipfile, os

zip_path = "../../competitions/digit-recognizer/digit-recognizer.zip"
extract_dir = "../data/"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ 解压完成")
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from datetime import datetime

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
print("✅ 数据加载完成")

y_train = train['label']
#Drop 'label' column
X_train = train.drop(labels = ['label'],axis=1) #axis=1 refers to columns
# free some space
del train

#check the data
print("训练集维度: ", X_train.shape)
print("测试集维度: ", test.shape)

# normalization
# 灰度化图像减少光照的影响
# 将图像的像素值缩放到0-1之间，CNN在0-1范围内收敛更快
X_train = X_train/255.0
test = test/255.0
print("✅ 数据归一化完成")

# reshape -> 28x28x1
# CNN需要4D张量作为输入，分别是样本数、高度、宽度、通道数
X_train = X_train.values.reshape(-1,28,28,1) # -1 表示Numpy根据其他维度自动计算该维度的值
test = test.values.reshape(-1,28,28,1)
print("✅ 数据重塑完成")

# labels encoding
y_train = to_categorical(y_train, num_classes=10)

# split training and validation set
random_seed = 2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)
print("✅ 训练集和验证集划分完成")

# set the CNN model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same',activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))    
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))  

#Define the optimizer
optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)     
epochs = 30
batch_size =86 # 86 is a magic number

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std         
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

# fit the model
history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, 
                              validation_data = (X_val,y_val),      
                                verbose = 2,
                                steps_per_epoch=X_train.shape[0] // batch_size,
                                callbacks=[learning_rate_reduction])

results = model.predict(test)
results = np.argmax(results,axis=1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,results.shape[0]+1),name = "ImageId"),results],axis = 1)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"cnn_mnist_{timestamp}.csv"

submission.to_csv(f"../submissions/{filename}",index=False)
print(f"✅ 预测结果已保存到 ../submissions/{filename}")


