import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets

# 去掉无关打印的信息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 加载一个数据集
# x:[60k, 28, 28]  y:[60k]
(x, y), _ = datasets.mnist.load_data()

# 转换成tensor
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  #这样x的数值范围0~1
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape)

print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))

# 创建一个数据集train_db。将上面的60k分为128一组。
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
# 迭代器iter
train_iter = iter(train_db)
# next()，可以得到下一组，即一次128,没next一次，就是进入下一个128分组。
# sample[0]表示128分组中的x。sample[1]128分组中的y
sample = next(train_iter)
print(sample[0].shape, sample[1].shape)



#因为要降维，[b, 784]->[b, 256]->[b, 128]->[b, 10]
# w@x+b
# 创建w1。利用裁剪后的正态分布
w1 = tf.Variable(tf.random.truncated_normal([784, 256],stddev=0.1)) # 要从784降成256，矩阵相乘后，就会是256
b1 = tf.Variable(tf.zeros([256])) #因为w1@x后，是256。所以b1要创建成256


w2 = tf.Variable(tf.random.truncated_normal([256, 128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))


w3 = tf.Variable(tf.random.truncated_normal([128, 10],stddev=0.1)) # 要从784降成256，矩阵相乘后，就会是256
b3 = tf.Variable(tf.zeros([10])) #因为w3@x后，是10。所以b3要创建成10

# 衰减因子
lr = 1e-3 # 科学计数法。10的-3次方

# 多循环几次，减少loss值，对整个数据集迭代
for epoch in range(10):

    # 对每个batch进行迭代 ，其中x是一个矩阵
    for step, (x, y) in enumerate(train_db):
        # 这里的x:[128, 28, 28]
        # y:[128]

        # 进行维度变换
        # [b, 28, 28]->[b, 28*28] 降维.其中b=128
        x = tf.reshape(x, [-1, 28 * 28])

        # GradientTape 为了计算梯度
        with tf.GradientTape() as tape:  # 由于只能跟踪variable类型，所以要将w1w2w3b1b2b3包装成variable

            # h1 = x@w1 + b1
            # [b, 784]@[784, 256] + [256]  -> [b, 256]  其中b=128
            h1 = x @ w1 + b1
            # 加上非线性因子
            h1 = tf.nn.relu(h1)

            # [b, 256] - > [b, 128]  其中b=128
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            # [b, 128] - > [b, 10]  其中b=128
            h3 = h2 @ w3 + b3

            # 计算误差
            # h3:[b, 10]  而 y:[b]  其中b=128

            # one_hot(), depth代表尺寸，y:[b]->[b,10]
            y_onehot = tf.one_hot(y, depth=10)

            # 做一个均方差 mse = mean(sum(y-h3)^2)
            loss = tf.square(y_onehot - h3)  # loss:[b, 10]
            # reduce_mean是求均值
            loss = tf.reduce_mean(loss)

        # 梯度的计算 即对loss分别求w1, b1, w2, b2, w3, b3的偏导
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # print(grads)
        # 因为w1 = w1 - lr*w1_grads 这样不断就越来越准确了
        # assign_sub()是原地更新，保持类型不变，功能和w1 - lr*w1_grads一样.
        # 利用公式计算，会返回tensor类型，而不是variable类型
        w1.assign_sub(lr * grads[0])  # grads[0]代表w1的梯度
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(step, 'loss:', float(loss))












