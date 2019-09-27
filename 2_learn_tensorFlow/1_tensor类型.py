import tensorflow as tf
import numpy as np

# tf.constant 创建变量

print(tf.constant(1))

print(tf.constant(1.))

# dtype 指定类型为double
print(tf.constant(2., dtype=tf.double))

print(tf.constant([True, False]))

print(tf.constant('hello world'))

'''
with tf.device("cpu"):
    a = tf.constant([1])

with tf.device("gpu"):
    b = tf.range([1])

print(a.device)
print(b.device)

# 转换为gpu
aa = a.gpu()
# 转换为cpu
bb = b.cpu()

# tensor 转换为 numpy
b.numpy()

# 查看tensor的维度，
a.shape # 查看维度。[[1,2,3],[2,4,4]] --> (2,3)
a.ndim # 查看具体维度值。[[1,2,3],[2,4,4]] --> 2,即2维
tf.rank(brank(a)) # 查看具体维度值，用tensor类型表示

'''


'''
with tf.device("cpu"):
    # a = tf.constant([[[1,2,3,4],[2,4,4,4],[2,4,4,4]],[[2,4,4,4],[1,4,2,3],[2,5,4,4]]])
    a = tf.constant([[[1,2,3],[4,4,4],[4,4,4],[4,4,4]]])

print(a.ndim)
print(a.shape)
print(tf.rank(a))

# tf.ones(维度, 类型) 创建一个全是1的tensor
aaa = tf.ones([3,2,2], tf.int32)
print(aaa)


# 判断是不是tensor, tf.is_tensor
print(tf.is_tensor(a))
# 查看类型
print(a.dtype)



# 创建一个numpy
a1 = np.arange(5)
print(a1.dtype)

# 将numpy转换为tensor类型
aa1 = tf.convert_to_tensor(a1)
aa2 = tf.convert_to_tensor(a1, dtype=tf.int32) # 转换为tensor,并制定类型
print(aa1)
print(aa2)

# cast类型转换，
aa3 = tf.cast(aa2, dtype=tf.float32) # 将aa2从int32转换为float32
print(aa3)

aa4 = tf.cast(aa2, dtype=tf.double) # 将aa2从int32转换为double
print(aa4)



# 整型转bool型
b = tf.constant([0,1])
b1 = tf.cast(b, dtype=tf.bool)
print(b1)

# bool转整型
b2 = tf.cast(b1, dtype=tf.int32)
print(b2)

'''


# Variable 经过Variable是一个特殊的Tensor类型，有可求导的特性，可以自动记录梯度
a = tf.range(5)
b = tf.Variable(a)

print(b)
# 只读属性。可训练的。表名这个变量可被优化更新
print(b.trainable)
# 判断是不是Tensor类型，因为是Variable是一个特殊的tensor，所以返回true
print(tf.is_tensor(b))



# 将tensor转换为numpy
b1 = b.numpy()
print(b1)

# 如果是标量，即0维的，也就是一个数，可以用int 或 float转换
aaa = tf.ones([]) # tf.ones创建一个全是1的tensor。这里是创建了个0维的
print(aaa)
# 将tensor转换为int。前提是0维的tensor
print(int(aaa))
# 将tensor转换为float。前提是0维的tensor
print(float(aaa))


