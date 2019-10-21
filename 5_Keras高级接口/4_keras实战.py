import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import os
import ssl
# 全局取消ssl证书验证
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 声明一个预处理函数
def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 导入cifar10数据集.这个数据集，是有10类图片。每个图片像素32*32 有RGB三个通道
# x[50k,32,32,3]   y[50k,1]  x_test[10k,32,32,3] y_test[10k,1]
(x, y),(x_test, y_test) = datasets.cifar10.load_data()
print(x_test.shape)
print(y_test.shape)
# 经过squeeze变换，去掉维度为1的。
y = tf.squeeze(y)  #y[50k,1]->[50k]
y_test = tf.squeeze(y_test) #y_test[10k,1] ->[10k]

# 对y和y_test进行onehot处理。因为这个数据集有10类，所以深度为10
y = tf.one_hot(y, depth=10) #[50k]->[50k,10]
y_test = tf.one_hot(y_test, depth=10) #[10k]->[10k,10]
print('datasets:',x.shape,y.shape,x_test.shape,y_test.shape,x.min(),x.max())


batchsize = 128 # batch的大小，
# 构造数据集。处理训练的数据集
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 预处理 shuffle是打乱顺序。将数据集进过preprocess处理后，映射(map)到原来的数据集中
# batch.将数据分成一分一分的，然后喂入神经网络
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsize)

# 将测试的数据集也处理一下
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(batchsize) #测试数据集不用打乱


# 迭代器，查看下batch后的shape.即一次喂入神经网络的shape
sample = next(iter(train_db))
print('batch',sample[0].shape, sample[1].shape)


# ---  新建网络 --
# 自定义层
class MyDense(layers.Layer):
    # init 方法
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        # 自定义参数kernel,并指定维度。
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])

        # 自定义，我们不要bias这个参数
        # self.bias = self.add_variable('b', [outp_dim])

    # call函数,前向传播函数
    def call(self, inputs, training=None):
        # 计算方法，即 input @ w
        x = inputs @ self.kernel
        return x

# 自定义网络(模型）
class MyNetwork(keras.Model):
    # init方法
    def __init__(self):
        super(MyNetwork, self).__init__()
        # 自定义5层网络
        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    # call方法
    def call(self, inputs, training=None):
        # x[b, 32, 32, 3]->[b, 32*32*3]
        x = tf.reshape(inputs, [-1, 32*32*3])
        # [b, 32*32*3]->[b,256]
        x = self.fc1(x)
        x = tf.nn.relu(x)
        # [b, 256]->[b,128]
        x = self.fc2(x)
        x = tf.nn.relu(x)
        # [b, 128]->[b,64]
        x = self.fc3(x)
        x = tf.nn.relu(x)
        # [b, 64]->[b,32]
        x = self.fc4(x)
        x = tf.nn.relu(x)
        # [b, 32]->[b,10]
        x = self.fc5(x)

        return x


# 实例化MyDense
network = MyNetwork()
# 利用自定义网络进行compile,即计算梯度和测试
#optimizer 指定优化器
#loss   指定loss
#metrics 指定测试方式
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
# fit,开始执行compile
# 参数1，train_db是训练的数据集，
# 参数epochs 训练多少次
# 参数validation_data 测试的数据集
# 参数validation_freq.训练多少次epoch，做一次测试，这里是训练一次测试一次
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)

# 最后在来一次最终测试结果
network.evaluate(test_db)


#------ 保存模型 -----------
# 这里保存一下模型
network.save_weights('ckpt/weights.ckpt')
# 删除模型
del network
print('saved to ckpt/weights.ckpt')

# 新建网络模型
network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
# 加载保存的模型
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file.')

# 测试结果，看看与保存前的一致不
network.evaluate(test_db)
