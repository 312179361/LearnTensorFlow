import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import layers



# 生成器
class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # z:[b, 100]->[b, 3*3*512]->[b, 3, 3, 512]->[b,64,64,3]

        self.fc = layers.Dense(3*3*512) #[b,100]->[b,3*3*512]

        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')



    def call(self, inputs, training=None):
        # [b, 100]->[b, 3*3*512]
        x = self.fc(inputs)

        x = tf.reshape(x, [-1, 3, 3, 512]) #[b,100]->[b,3,3,512]
        x = tf.nn.leaky_relu(x)

        # [b,3,3,512]->[b,9,9,256]
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        # [b,9,9,256]->[b,21,21,128]
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))

        # [b,21,21,128]->[b,63,63,3]
        x = self.conv3(x)

        x = tf.tanh(x)  # x∈(-1,1)

        return x


# 作为验证的，其实就是一个分类器
class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b,64,64,3]->[b, 20,20,64]
        # Conv2D(filters, kernel_size, strides, padding)
        # filters  输出的空间维度，即卷积中的滤波器数
        # kernel_size  卷积核的大小。2个整数的整数元组/列表,指定2D卷积窗口的高度和宽度.可以是单个整数,以指定所有空间维度的相同值。
        # 这里的5就是5*5
        # strides 2个整数的整数或元组/列表,指定卷积沿高度和宽度的跨度.可以是单个整数,以指定所有空间维度的相同值.指定任何步幅值！= 1与指定任何dilation_rate值！= 1都不相容
        self.con1 = layers.Conv2D(64, 5, 3, 'valid')

        # [b,20,20,3]->[b, 20,20,128]
        self.con2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        # [b,20,20,128]->[b, 20,20,256]
        self.con3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # 打平 [b, h, w, c]->[b, -1] 就保留第0维度，其余的维度相乘，如[b, 2, 2, 3]->[b, 2*2*3]
        self.flatten = layers.Flatten()

        # 全连接层[b, -1]->[b, 1]
        self.fc = layers.Dense(1)



    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu( self.con1(inputs))

        x = tf.nn.leaky_relu(self.bn2(self.con2(x), training=training))

        x = tf.nn.leaky_relu(self.bn3(self.con3(x), training=training))

        # 打平 [b, h, w, c]->[b, -1]
        x = self.flatten(x)
        print('abcde:::',x.shape)
        # [b, -1]->[b, 1]
        logits = self.fc(x)


        return logits



def main():
    # 实例化验证器
    d = Discriminator()
    # 实例化生成器
    g = Generator()


    # 假数据
    x = tf.random.normal([2, 64, 64,3])

    z = tf.random.normal([2, 100])
    # 将图片放入验证器
    prob = d(x)
    print(prob)


    # 将参数放入生成器，生产图片
    x_hat = g(z)
    print(x_hat.shape)



if __name__ == '__main__':
    main()