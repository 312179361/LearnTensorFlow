import tensorflow as tf

# 具体的函数解释看笔记(p3)
#--------------- sigmoid激活函数 --------------
'''


# linspace(start, end ,num) 返回一个等差数列,start是第一个数字，end是最后一个数字，num是一共有多少个数字
a = tf.linspace(-10., 10., 10)
print(a)


with tf.GradientTape() as tape:
    # 跟踪a变量
    tape.watch(a)
    # 将a带入sigmoid中，构造出y和a的联系。即得出loss
    y = tf.sigmoid(a)
    print(y)

# 计算y对a的梯度
grads = tape.gradient(y, [a])
print(grads)

'''
#--------------- tanh 激活函数 --------------

a1 = tf.linspace(-5., 5., 10)
y1 = tf.tanh(a1)
print(y1)


#--------------- ReLU 及leaky_relu 激活函数 --------------
a2 = tf.linspace(-1., 1., 10)
y2 = tf.nn.relu(a2)
print(y2)

y3 = tf.nn.leaky_relu(a2)
print(y3)
