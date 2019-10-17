'''
三种模式
save/load weights       最轻量级，只保存网络参数，不保存状态

save/load entire model  保存所有状态和参数。可完美的恢复

saved_model             保存的格式是通用的，可以用其他语言进行解析等操作
'''


import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
# 第一种保存加载

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()
print('datasets:', x.shape, y.shape, x.min(), x.max())



db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz) 

sample = next(iter(db))
print(sample[0].shape, sample[1].shape)


network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))
network.summary()




network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
# 执行3次训练
network.fit(db, epochs=3, validation_data=ds_val, validation_freq=2)
 # 做一次测试
network.evaluate(ds_val)
# 保存模型，save_wights， 只保存网络参数
network.save_weights('weights.ckpt')
print('saved weights.')
# 将网络删除
del network

# 重建网络，注意重建的网络要和上面的网络一模一样，才能复原
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),
		loss=tf.losses.CategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
# 加载保存的网络参数，这样就可以复原了。因为值保存了网络参数，不会完美的复原，经过测试发现结果可能不会一模一样，会很接近
network.load_weights('weights.ckpt')
print('loaded weights!')
# 测试，
network.evaluate(ds_val)



'''
第三种保存方式 saved_model
# 保存
tf.saved_model.save(model, '路径')

# 加载
imported = tf.saved_model.load(path)
f = imported.signatures["serving_default"]

'''
