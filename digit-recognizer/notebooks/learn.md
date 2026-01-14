# 1.怎么数据预处理的
## 归一化
MNIST数据集本身是一个灰度图像,每张图片是一个784个像素的一维数组。0~255
```python
X_train = X_train/255.0
test = test/255.0
```
把图像像素缩放到0~1之间，CNN在0~1之间收敛更快

## 数据重塑
CNN需要4D张量（tensor）作为输入,(样本书，高度，宽度，通道数) --> CNN一次处理的是一批图片,每次都向前传播一个batch
```python
X_train = X_train.values.reshape(-1,28,28,1)
```
## labels encoding
在MNIST数据集中 label是 0,1,2,3,4,5,6,7,8,9这样的形式
输出层softmax输出的是0-9的9个概率。是在给10个类别进行打分,所以label也需要是10维的
one-hot encoding 会将原来的y_train转化为[1,0,0,0,0,0,0,0,0,0]这样的形式
```python
y_train = to_categorical(y_train,num_classes=10)
```
**one-hot encoding 还跟损失函数的选择有关系** 

# 2.怎么构建网络
**step 1:**
创建一个Sequential对象-model --> 顺序模型
**step 2:**
model.add(某一层) --> 把一层网络按顺序接到当前模型的最后一层后面

Dense --> 全连接层:
```python
model.add(Dense(
    units=128,
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform'
))
```
units：神经元个数
activation:激活函数
| 激活函数      | 用途        |
| --------- | --------- |
| `relu`    | 中间层（默认首选） |
| `sigmoid` | 二分类输出     |
| `softmax` | 多分类输出     |
| `tanh`    | 老模型，少用    |
use_bias:是否使用偏置项：表达式中的b
kernel_initializer：权重W的初始化方式
| 初始化              | 用途      |
| ---------------- | ------- |
| `glorot_uniform` | 默认，通用   |
| `he_normal`      | ReLU 网络 |
| `zeros`          | 不推荐     |

Conv2D --> 卷积层
```python
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    activation='relu',
    input_shape=(28, 28, 1)
))
```
filters：卷积核个数，每个filter学习一种特征，数量越多，模型越强，但训练的也越慢
kernel_size：卷积核大小 （3，3）绝对主流
strides：滑动步长 （1,1）：保留细节 （2,2）下采样(类似pooling)
padding：
| padding | 含义       |
| ------- | -------- |
| `valid` | 不补零，尺寸变小 |
| `same`  | 补零，尺寸不变  |
> 新手几乎永远用same
input_shape：只在第一层写

MaxPooling2D --> 下采样
```python
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
))
```
pool_size：窗口大小
strides：通常等于 pool_size；不写就默认等于 pool_size

Flatten
```python
model.add(Flatten())
```
把 CNN 的三维输出摊平成一维

Dropout --> 防止过拟合
```python
model.add(Dropout(0.5))
```
训练时随机丢弃 50% 神经元
0.25：卷积层后
0.5：全连接层后



