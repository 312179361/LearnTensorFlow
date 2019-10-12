import tensorflow as tf
# 多输出感知机，就是多个输入节点，输出也是多个节点

x = tf.random.normal([2, 4]) # 2个样本，输入4个节点。经过计算得到3个节点
w = tf.random.normal([4, 3])
b = tf.zeros([3])

y = tf.constant([2, 0])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x@w+b, axis=1) # 计算后shape=[2,3]
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

# 计算梯度
grads = tape.gradient(loss, [w, b])

print(grads[0])
print(grads[1])

