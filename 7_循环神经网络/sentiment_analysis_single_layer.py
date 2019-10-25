# 情感分类
import  os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import ssl


# 全局取消ssl证书验证
ssl._create_default_https_context = ssl._create_unverified_context

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

batchsz = 128 # batchSize
embedding_len = 100

# 导入数据集，是一个电影评价的数据集.x是评语，y是0或1,是差评或好评
total_words = 10000 # 规定总单词数量，其实表示常见的单词，如果超出1万的单词就用一个特殊的统一标识，因为常见单词才是语义的主要意思
max_review_len = 80 # 规定句子中的单词长度，如果超过80个，就截取。如果不够，就补零。这样后期方便处理
(x_train, y_train),(x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)

# 将数据集的数据都进行80长度的限定
# x_train [b, 80] b是多少个句子，80是有80个单词
# x_test [b, 80]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)

#构建数据集
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# drop_remainder 表示在少于batch_size元素的情况下是否应删除最后一批，True是删除，这样保证每个batch都是128，
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
#y=1是好评，y=0是差评
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


# 构建循环神经网络
class MyRNN(keras.Model):
    def __init__(self, unit):
        super(MyRNN, self).__init__()

        # embedding层，将字符串转换为数据 embedding_len就是embedding的长度，即一个单词用一个100维的向量表示
        # [b, 80](b句话，每句话80个单词)->[b, 80, 100]（b句话，每句话80个单词，每个单词是100的维度）
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        # 循环网络层.创建两层
        self.rnn = keras.Sequential([
            layers.SimpleRNN(unit, dropout=0.5,return_sequences=True, unroll=True),
            layers.SimpleRNN(unit, dropout=0.5,unroll=True)
        ])
        # [b, 80, 100] -> [b, 64]

        # 全连接层,分类
        self.outlayer = layers.Dense(1)


    def call(self, inputs, training=None):

        '''
        :param inputs: 原始句子[b, 80]
        :param training: 表示是训练还是test
        :return:
        '''
        # x[b,80]
        x = inputs
        # embedding:[b,80]->[b,80,100]
        x = self.embedding(x)

        # rnn 层
        x = self.rnn(x)
        # 经过rnn，x[b,80,100]->x[b,64]

        # 将x送入全连接层  x[b,64]->[b,1]
        x = self.outlayer(x)
        # 将数据压缩到0~1，即得到概率
        prob = tf.sigmoid(x)
        return prob



def main():
    units = 64
    epochs = 4
    # 实例化RNN
    model = MyRNN(units)
    # 指定优化器和loss
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss = tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    # 测试
    model.evaluate(db_test)



if __name__ == '__main__':
    main()
