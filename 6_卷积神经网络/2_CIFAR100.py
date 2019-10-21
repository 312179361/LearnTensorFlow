import  tensorflow as tf
from  tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import  os
import ssl

# 全局取消ssl证书验证
ssl._create_default_https_context = ssl._create_unverified_context

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

# 创建conv层，将其放在一起。其目的是让二维shape变小，但是深度增加，
conv_layers = [ # 5个units .每个unit包含2个conv+1个max pooling
    # unit 1 两个conv+1个max pooling
    # layers.Conv2D 卷积层 conv。
    # 参数1 64。代表有64个kernel,即有64个kernel去扫描
    # kernel_size卷积层kernel的大小，kernel的深度是通过输入来判定的
    # padding="same"代表，输入和输出的参数维度不变,否则参数会略小与原来的
    # activation
    layers.Conv2D(64, kernel_size=[3, 3], padding="same",activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # max pooling
    # pool_size的大小
    # strides是步长。
    layers.MaxPool2D(pool_size=[2, 2],strides=2, padding="same"),# 步长是2,这一步会让x的横纵维度减半
    # [b, 16, 16, 64]

    # unit2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same",activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2],strides=2, padding="same"),
    # [b, 8, 8, 128]

    # unit3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same",activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2],strides=2, padding="same"),
    # [b, 4, 4, 256]

    # unit4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same",activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2],strides=2, padding="same"),
    # [b, 2, 2, 512]

    # unit5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),
    # [b, 1, 1, 512]

]


# 数据集的预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255. # 转换到0~1的范围
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 加载数据集 x[50k,32,32,3]  y[50k,1] x_test[10k,32,32,3]  y_test[10k,1]
# x是50k张图片。每个图片是32*32的，有RGB三个通道
# y是50个数字，每个数字的值代表着一类，这个数据集中一共有100类，即y的最大值是100
(x, y), (x_test, y_test) = datasets.cifar100.load_data()
# 因为y的的第二个维度是1，所有可以用squeeze去挤压掉，y[50k, 1] -> [50k];y_test[10k, 1] -> [10k]
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

# 构造数据集
# 训练数据集
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(64) #每次喂入数据是64个
# 测试数据集
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = train_db.map(preprocess).batch(64)

sample = next(iter(train_db))


print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

def main():
    # 将创建的卷积层，放到Sequential中
    conv_net = Sequential(conv_layers)
    # conv_net.build(input_shape=[None, 32, 32, 3])
    # x = tf.random.normal([4, 32, 32, 3])
    # out = conv_net(x)
    # print(out.shape)


    # 创建全连接层
    fc_net = Sequential([
        # layers.Dense(256, activation=tf.nn.relu),
        # layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),

    ])
    conv_net.build(input_shape=[None, 32, 32, 3])
    # 这里全连接层的512是卷积层的输出，
    fc_net.build(input_shape=[None, 512])

    # 声明优化器
    # 他的作用是更新优化参数，即w = w - lr*grad这样更新
    optimizer = optimizers.Adam(lr=1e-4)

    # 声明一下参数列表，因为有卷积层和全连接层，所以将两个的参数拼接起来。在计算梯度使用
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    # ---开始训练---
    # 训练50次
    for epoch in range(50):
        # 遍历训练数据集.由于db一共是60k个数据，而batch是128,所以一共循环60k/128次
        # enumerate 可将一个可遍历的对象，列出数据和数据下标。即step就可以列出循环的次数
        for step,(x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                # x先经过卷积层 [b, 32, 32, 3] -> [b, 1, 1, 512]
                out = conv_net(x)
                # 维度变换 [b, 1, 1, 512]->[b, 512]
                out = tf.reshape(out, [-1, 512])

                # 接着进入全连接层
                #[b, 512]->[b, 100]
                logits = fc_net(out)

                # 计算loss
                y_onehot = tf.one_hot(y, depth=100) #[b]->[b,100]
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)


            #计算梯度
            # 参数是卷积层和全连接层的参数之和
            grads = tape.gradient(loss, variables)
            # 更新参数，即更新trainable_variables
            optimizer.apply_gradients(zip(grads, variables))


            if step % 5 == 0:

                print(epoch, step,'loss:',float(loss))

   
        total_num = 0
        total_correct = 0
        # ----------  开始测试 -----
        for x, y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            # 变换成概率
            prob = tf.nn.softmax(logits, axis=1)
            # 去最大概率的下标
            pred = tf.argmax(prob,axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            #和真实的y，进行比较.将true->1 将false->0
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            # 相加，算结果分数
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print(epoch,'acc:',acc)
        




if __name__ == '__main__':
    main()

