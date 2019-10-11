import tensorflow as tf
# 单输出感知机，就是多个输入节点，最后输出是一个节点
x = tf.random.normal([1, 3]) # 三个节点。经过运算后，变成一个节点
w = tf.ones([3, 1])
b = tf.ones([1])


y = tf.constant([1])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.sigmoid(x@w+b) # 经过运算shape=[1, 1]
    print(prob)
    # 这里y不用onehot是因为就一个节点，
    loss = tf.reduce_mean(tf.losses.MSE(y, prob))

# 计算梯度
grads = tape.gradient(loss, [w, b])
print(grads)

