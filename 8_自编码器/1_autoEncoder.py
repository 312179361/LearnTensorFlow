import  os
import  tensorflow as tf
import  numpy as np
from  tensorflow import keras
from  tensorflow.keras import Sequential, layers
from  PIL import Image
from matplotlib import pyplot as plt

import ssl
import time

# 全局取消ssl证书验证
ssl._create_default_https_context = ssl._create_unverified_context

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


# 把多张image，拼接成一个image
def save_images(imgs, name):
    new_im = Image.new('L',(280,280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            print('+++',im.shape)
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)


h_dim = 20 # 需要降维到20
batch_size = 512
lr = 1e-3 # 衰减因子

# 加载数据集
(x_train, y_train),(x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# astype强制类型转换，然后在除以255，这样就x_train和x_test就在0~1中
x_train, x_test = x_train.astype(np.float32) / 255. , x_test.astype(np.float32) / 255.
# 因为是无标签学习，即没有真实值,所以不用y_train和y_test
#构建数据集
db_train = tf.data.Dataset.from_tensor_slices(x_train)
# drop_remainder 表示在少于batch_size元素的情况下是否应删除最后一批，True是删除，这样保证每个batch都是128，
db_train = db_train.shuffle(batch_size*5).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices(x_test)
db_test = db_test.batch(batch_size)

#y=1是好评，y=0是差评
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


#创建AE类
class AE(keras.Model):
    # 初始化方法
    def __init__(self):
        super(AE, self).__init__()

    # encoder方法，编码.即降维
        self.encoder = Sequential([
            # 创建三层
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim) #最终降维到20维
        ])


        # decoder，解码。即升维，还原
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784) # 28*28
        ])



    # call函数，前向传播
    def call(self, inputs, training=None):
        # 将input通过encoder降维 [b, 784]->[b, 10]
        h = self.encoder(inputs)
        # 升维，还原 [b, 10]->[b, 784]
        x_hat = self.decoder(h)

        return x_hat


# 实例化模型
model = AE()
# 输入的维度是784，即28*28
model.build(input_shape=(None, 784))
model.summary()

# 创建优化器
optimizer = tf.optimizers.Adam(lr=lr)

# 开始训练
for epoch in range(100):
    for step , x in enumerate(db_train):

        # [b, 28, 28] -> [b, 784]
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            # 通过调用AE 的call方法，重现图片数据
            x_rec_logits = model(x)

            # loss
            rec_loss = tf.losses.binary_crossentropy(x, x_rec_logits, from_logits=True)
            rec_loss = tf.reduce_mean(rec_loss)

        # 计算梯度
        grads = tape.gradient(rec_loss, model.trainable_variables)
        # 更新参数
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

        # 测试
        x = next(iter(db_test)) # 从testDB中取一个数据

        logits = model(tf.reshape(x, [-1, 784])) # 将数据喂入训练的模型中
        x_hat = tf.sigmoid(logits)
        # 还原数据。  x_hat[b, 784]->[b,28,28]
        x_hat = tf.reshape(x_hat, [-1,28,28])

        # [b,28,28]->[2b,28,28]
        # x_concat = tf.concat([x, x_hat],axis=0)
        x_concat = x
        x_concat = x_concat.numpy() * 255.
        x_concat = x_concat.astype(np.uint8)

        print('===x_concat:',x_concat.shape)

        save_images(x_concat,'ae_image/aarec_epoch_%d.png'%epoch)












