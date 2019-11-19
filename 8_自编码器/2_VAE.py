import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
from    matplotlib import pyplot as plt
import ssl

# 全局取消ssl证书验证
ssl._create_default_https_context = ssl._create_unverified_context

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 把多张image，拼接成一个image
def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


h_dim = 20 # 需要降维到20
batchsz = 512
lr = 1e-3  # 衰减因子

# 加载数据集
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
# astype强制类型转换，然后在除以255，这样就x_train和x_test就在0~1中
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# 因为是无标签学习，即没有真实值,所以不用y_train和y_test
#构建数据集
train_db = tf.data.Dataset.from_tensor_slices(x_train)
# drop_remainder 表示在少于batch_size元素的情况下是否应删除最后一批，True是删除，这样保证每个batch都是128，
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

z_dim = 10

class VAE(keras.Model):

    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim) # 均值Mean
        self.fc3 = layers.Dense(z_dim) # 方差Variance

        # Decoder
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        # 将输入x，放入fc1中，然后激活后，得到h
        h = tf.nn.relu(self.fc1(x))
        # 将h分别放入fc2和fc3中
        mu = self.fc2(h) #得到均值mean
        log_var = self.fc3(h) # 得到方差Variance

        return mu, log_var

    def decoder(self, z):
        # 经过fc4和fc5，即decoder完毕
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)

        return out

    def reparameterize(self, mu, log_var):

        eps = tf.random.normal(log_var.shape)

        std = tf.exp(log_var*0.5)

        z = mu + std * eps
        return z

    def call(self, inputs, training=None):

        # 执行encoder,得到mu和log_var
        # [b, 784] -> [b, z_dim],[b, z_dim]
        mu, log_var = self.encoder(inputs)
        # reparameterization trick
        z = self.reparameterize(mu, log_var)
        # 执行decoder
        x_hat = self.decoder(z)

        return x_hat, mu, log_var


model = VAE()
model.build(input_shape=(4, 784))
optimizer = tf.optimizers.Adam(lr)

for epoch in range(1000):

    for step, x in enumerate(train_db):
        # [b, 28, 28] -> [b, 784]
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits, mu, log_var = model(x)
            # loss
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

            # compute kl divergence (mu, var) ~ N (0, 1)
            # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            kl_div = -0.5 * (log_var + 1 - mu**2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]

            loss = rec_loss + 1. * kl_div
        #计算参数
        grads = tape.gradient(loss, model.trainable_variables)
        #更新梯度
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if step % 100 == 0:
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))


    # 测试
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() *255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_image/sampled_epoch%d.png'%epoch)

    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() *255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_image/rec_epoch%d.png'%epoch)

