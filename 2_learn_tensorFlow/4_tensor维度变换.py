'''
维度

shape 维度，(2, 3)， 即2x3的矩阵维度，是2维的
ndim 具体的维度值，如2维，3维

'''

# shape的顺序。[2, 3]是2行3列、[1, 2, 3]是深度是1，2行3列。多维的以此类推

import tensorflow as tf


a = tf.random.normal([4, 28, 28, 3])
print(a.shape) # (4, 28, 28, 3)
print(a.ndim)  # 4，即4维

#------------维度变换---------

# 维度变换reshape，要保证变换前后维度乘积数值一样。即4*28*28*3 = 4*784*3
a1 = tf.reshape(a, [4, 784, 3])  # 将[4, 28, 28, 3] -> [4, 784, 3]，即将中间的两个维度合起来了。784=28*28
a2 = tf.reshape(a, [4, -1, 3])   #-1会自动算出28*28，即上面的简写
a3 = tf.reshape(a, [4,28*28*3])  # 将[4, 28, 28, 3] -> [4, 2352]，即将4维，变换成2维
print(a1.shape) #(4, 784, 3)
print(a1.ndim)

a5 = tf.reshape(a,[2, 2, 28, 28, 3]) #将[4, 28, 28, 3] -> [2, 2, 28, 28, 3],即将4维，变换成5维
print(a5.shape)
print(a5.ndim)

#------------维度转置---------

# 维度的转置，,即。transpose
b = tf.random.normal([4, 3, 2, 1])
# transpose 矩阵的转置，即a4321->a1234 , a3321->a1233,
b1 = tf.transpose(b)  # shape从(4,3,2,1) -> (1,2,3,4)
print(b1.shape)


# 指定某些维度转置，传入参数perm。[0, 1, 3, 2]代表原来的维度编号
# 原先0号维度是4，1号维度是3，2号维度是2，3号维度是1
# 如[0, 1, 3, 2] 即第0个位置放原来的0号维度，即4
#                   第1个位置放原来的1号维度，即3
#                   第2个位置放原来的3号维度，即1
#                   第3个位置放原来的2号维度，即2
#    最后结果是，[4, 3, 1, 2]

b2 = tf.transpose(b, perm=[0, 1, 3, 2]) # 最后结果是4,3,1,2，也就是将最后两个维度转置了一下，前两个维度不变
print(b2.shape)
b3 = tf.transpose(b, perm=[1, 3, 0, 2]) #结果shape为(3,1,4,2)
print(b3.shape)


#------------维度的添加---------
# 添加一个维度expand_dims,参数axis是维度添加的位置，
c = tf.random.normal([4, 3, 2])
c1 = tf.expand_dims(c, axis=0) #在第0号位，加一个维度， [4, 3, 2] -> [1, 4, 3, 2]。这样3维变4维
print(c1.shape)
c2 = tf.expand_dims(c, axis=3) #在第3号位，加一个维度， [4, 3, 2] -> [4, 3, 2, 1]
print(c2.shape)
c3 = tf.expand_dims(c, axis=-1) #在第-1号位,即最后，加一个维度， [4, 3, 2] -> [4, 3, 2, 1]
print(c3.shape)


#------------维度的减小---------
# squeeze,维度的减小。
# 当某个维度shape的值是1的时候，就可以把这个维度去掉，即减小
d = tf.zeros([1, 2, 1, 1, 3])
print(d.shape)
d1 = tf.squeeze(d) # 将shape为1的全部去掉，即(2,3)
print(d1.shape)

# 减小某个指定维度，加上参数axis，这个是shape的编号
# 维度不为1的不能减少
d2 = tf.squeeze(d, axis=0) #将shape标号为0的去掉，即(2,1,1,3)
print(d2.shape)

d3 = tf.squeeze(d, axis=[0, 2])#将shape标号为0和2的去掉，即(2,1,3)
print(d3.shape)

d4 = tf.squeeze(d, axis=-2) #将shape标号为-2的去掉，即(1,2,1,3)
print(d4.shape)









