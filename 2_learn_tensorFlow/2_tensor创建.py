'''
创建tensor 有很多种方法
1、从numpy 或者 list 中创建
2、利用zeros 和 ones
3、fill 方式
4、random方式  随机化的初始化
5、constant 方式

注* 1和5都是传递data，即具体的矩阵值
    2，3，4都是传递矩阵维度，即shape
'''

import numpy as np
import tensorflow as tf

'''


# 1、从numpy 或者 list 中创建
# 创建一个numpy， np.ones() 创建一个全1的numpy
np1 = np.ones([2,3])
# 利用convert_to_tensor转换
tf1 = tf.convert_to_tensor(np1)
print(tf1)

np2 = np.zeros([2, 3])
tf2 = tf.convert_to_tensor(np2)
print(tf2)


# 利用convert_to_tensor将list转换为tensor
tf3 = tf.convert_to_tensor([1,2])
print(tf3)

tf4 = tf.convert_to_tensor([[1],[2]])
print(tf4)



# 2、通过ones和zeros的方式创建tensor  。ones和zero括号内传递的是shape即维度
tf5 = tf.ones([])     # 0维的  标量
print(tf5)

tf6 = tf.ones([1])     # 1维的 1x1
print(tf6)

tf7 = tf.zeros([2, 2])       # 2维度 2x2
print(tf7)



#3、利用fill 创建tensor  fill(shape, 要填充的数值)
tf8 = tf.fill([2, 1], 8)  # 创建一个2x1 的矩阵，数值都是8
print(tf8)

tf9 = tf.fill([], 9)    # 创建一个标量，数值为9
print(tf9)



#4、随机化的初始化 random
# random.normal(shape, mean=正态分布的均值,默认0, stddev=正态分布的标准差,默认1)是正态分布，
tf10 = tf.random.normal([2, 2], mean=1, stddev=1)  #得出一个2*2的矩阵，每个元素符合正态分布
print(tf10)

#random.truncated_normal 截取一部分的正态分布
tf11 = tf.random.truncated_normal([2, 2], mean=0, stddev=1)
print(tf11)


#random.uniform  均匀分布
# 在0到1之间的一个均匀分布采样，构成的2x2的矩阵
tf12 = tf.random.uniform([2, 2], minval=0, maxval=1)
print(tf12)

# 在0到100之间的一个均匀分布采样，构成的2x2的矩阵
tf13 = tf.random.uniform([2, 2], minval=0, maxval=100)
print(tf13)

 

# 5、通过constant创建tensor 。tf.constant(具体的值，不是shape)
# constant和convert_to_tensor 基本上一样的功能

tf14 = tf.constant(1)     # 创建一个数值为1的标量
print(tf14)

tf15 = tf.constant([1])    # 创建一个1x1，数值为1的tensor
print(tf15)

tf16 = tf.constant([2, 3])    # [2, 3]的tensor
print(tf16)


# [1, 2]
# [3, 4] 的2x2的tensor
tf17 = tf.constant([[1, 2],[3, 4]])
print(tf17)

'''


 







