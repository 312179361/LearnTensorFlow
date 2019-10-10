# 损失函数就是loss 具体函数看笔记（p4）

import tensorflow as tf

#-------- MSE 损失函数   ---------------
'''
# 拿出x值
x = tf.random.normal([2, 4]) # 相当于两个样本，
# 构建预测的w和b
w = tf.random.normal([4, 3])
b = tf.zeros([3])

# 计算的结果w@x+b 的shape为[2, 3],即两个样本，三种可能，

# 取到真实的y值
y = tf.constant([2, 0])



with tf.GradientTape() as tape:
    # 跟踪w和b
    tape.watch([w, b])
    print(x@w+b)
    # 先矩阵线性运算，然后通过softmax函数，得到预测值prob
    prob = tf.nn.softmax(x@w+b, axis=1) # 保证每个样本的三个可能概率相加都是1，即每一行的概率相加为1
    print(prob)

    # loss表达式 MSE
    # one_hot，depth=3是深度为3，[2, 0] ->[[0,0,1],[1,0,0]]
    # losses.MSE(真实值，预测值)就是求mse的loss
    # reduce_mean是均值。求两个loss的均值
    # reduce_mean(losses.MES())这两步加起来，效果就是MSE的loss，即1/N * Σ(y真-y预)^2
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

# 求loss对[w, b]的梯度
grads = tape.gradient(loss, [w, b])
# grads[0]就是loss对w的梯度，即loss对w的偏导
print(grads[0])
# grads[1]就是loss对b的梯度，即loss对b的偏导
print(grads[1])

'''


#-------- crossEntropy 损失函数   ---------------

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])
b = tf.random.normal([3])

y = tf.constant([2, 0])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    # 预测值
    logits = x@w+b
    print(logits)
    # losses.categorical_crossentropy(真实值，预测值)就是求crossEntropy的loss
    # 这里不用softmax,而是from_logits=True，这样让TensorFlow自动处理。自己用softmax处理可能会出现数据不稳定
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y, depth=3), logits, from_logits=True))

# 求梯度
grads = tape.gradient(loss, [w, b])
# grads[0]就是loss对w的梯度，即loss对w的偏导
print(grads[0])
# grads[1]就是loss对b的梯度，即loss对b的偏导
print(grads[1])