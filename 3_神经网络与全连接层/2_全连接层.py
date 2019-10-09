import tensorflow as tf
from tensorflow import keras

x = tf.random.normal([2, 3])

# sequential容器，可以将多个Dense层包含起来，可以逐次进行
model = keras.Sequential([
        # 第一层 输出维度是2，推断，kernel(即w)为[3, 2]，这样x@w才是[2, 2]
        keras.layers.Dense(2, activation='relu'),
        # 第二层 输出维度是2，推断，kernel(即w)为[2, 2]
        keras.layers.Dense(2, activation='relu'),
        # 第三层 输出维度是2，推断，kernel(即w)为[2, 2]
        keras.layers.Dense(2)
    ])

# build，必须是3，这样kernel才是[3, 2]。其中2是第一层的维度指定的
model.build(input_shape=[None, 3])
model.summary()

for p in model.trainable_variables:

    print(p.name, p.shape)