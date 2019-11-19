import  os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from PIL import Image
import  glob
from    GAN import Generator, Discriminator

from    dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)

    image = Image.fromarray(final_image)
    image.save(image_path)
    # toimage(final_image).save(image_path)



# 真图片的loss计算
def celoss_ones(logits):
    # tf.ones_like(tensor) 返回一个和给定tensor形状(shape)一样的，但是数值都是1
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))

    return tf.reduce_mean(loss)

# 假图片的loss计算
def celoss_zeros(logits):
    # tf.zeros_like(tensor) 返回一个和给定tensor形状(shape)一样的，但是数值都是0
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


# 验证器的loss函数
def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    #1、将真实的图片，当做真的
    #2、将生成的图片，当做假的

    # 将batch_z输入generator中，得到生成的图片，即假图片
    fake_image = generator(batch_z, is_training)
    print('假图片',fake_image.shape)

    # 将假图片放入discriminator函数中
    d_fake_logits = discriminator(fake_image, is_training)
    # 将真图片放入discriminator函数中
    d_real_logits = discriminator(batch_x, is_training)

    # 将上面的两个经过验证的输出，分别做loss计算
    d_loss_real = celoss_ones(d_real_logits) # loss越小，说明越真
    d_loss_fake = celoss_zeros(d_fake_logits) # loss越小，说明越假

    # 验证的目的是 输入真的图片，返回时真。输入假的图片返回是假。即真的loss最小，假的loss也最小
    # 所以将两个相加，如果越小越好
    loss = d_loss_fake + d_loss_real

    return loss



# 生成器的loss函数
def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 目的是 把生成的图片 当做真的

    fake_image = generator(batch_z, is_training)
    # 把生成的假图片，放入验证器中
    d_fake_logits = discriminator(fake_image, is_training)

    # 希望验证结果是真的.
    loss = celoss_ones(d_fake_logits) # 如果越像真的，loss就越小

    return loss




def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    # hyper parameters
    z_dim = 100
    epochs = 3000000
    batch_size = 512
    learning_rate = 0.002
    is_training = True

    # 数据集的加载
    img_path = glob.glob('/Users/tongli/Desktop/Python/TensorFlow/faces/*.jpg')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)

    print(dataset, img_shape)

    sample = next(iter(dataset))

    print(sample.shape, tf.reduce_max(sample).numpy(),
          tf.reduce_min(sample).numpy())

    dataset = dataset.repeat()

    db_iter = iter(dataset)


    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    # 优化器
    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        # genrator的输入是随机输入的.生成一张图片

        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        # 真实的图片
        batch_x = next(db_iter)

        # train D 训练验证器
        with tf.GradientTape() as tape:
            d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)

        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))


        # train G 训练生成器
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, ',d-loss:',float(d_loss), ',g-loss:',float(g_loss))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')



if __name__ == '__main__':
    main()
