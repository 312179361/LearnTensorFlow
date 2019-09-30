import tensorflow as tf
'''
a = tf.random.normal([3, 3])
print(a)

mask = a > 0 # 每个元素和0比较，大于0的是true，小于0的是False
print(mask) #[3, 3]

a1 = tf.boolean_mask(a, mask) # boolean_mask。得到是true的值，即过滤掉false后
print(a1)

indices = tf.where(mask)  # where(), 得到mask是true的索引
print(indices)

# 通过gather_nd函数，利用true的索引，可以得到全是true的值，和boolean_mask的结果一样
a2 = tf.gather_nd(a, indices)
print(a2)


#---------------------
# 如果where(cond, A, B)有三个参数

A = tf.ones([3, 3])
B = tf.zeros([3, 3])
# where(mask, A, B)。 如果mask为true,那么就在A中选择对应索引的值。如果mask为False,就在B中选择对应索引的值
C = tf.where(mask, A, B)
print(C)

#---------------------
#scatter_nd(indices, updates, shape) 。shape是一个底板，通过indices得知每个元素的下标，然后将updates更新到底板上

indices = tf.constant([[4],[3],[1],[6]]) # [4,1] 位置下标
print(indices)
updates = tf.constant([9,10,11,12]) # 要更新的值
shape = tf.constant([9]) # 指定底板的shape ，更新的时候，会变成[0,0,0...]

res = tf.scatter_nd(indices, updates, shape) # 将updates的每一个值，对应的位置是indeces,然后更新到shape上
print(res) #[ 0 11  0 10  9  0 12  0  0]

# 三维的更新
# 指定下标是0和2，即代表第0层和第2层
indices1 = tf.constant([[0],[2]])
# updates1.shape  [2, 4, 4,]
updates1 = tf.constant([ [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],

                         [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]] ] )

shape1 = tf.constant([4, 4, 4]) # 指定底板的shape ，更新的时候，会变成4*4*4全是0的底板


res1 = tf.scatter_nd(indices1, updates1, shape1) # 将updates1的两层数据，更新到底板上。即第0层和第2层有数据，其他两层全是0
print(res1)

'''

'''
#---------------------------------------

# 得到一个5*5的坐标，每个坐标有x和y 两个值，所以，整体是5*5*2的矩阵
# linspace(start, end, num) 从-2到2中间一共取5个点，平均分配
x = tf.linspace(-2., 2, 5)
print(x)
y = tf.linspace(-2., 2, 5)
print(y)

# meshgrid将x和y合并成25个点，然后将全部的x和y分别返回
point_x, point_y = tf.meshgrid(x, y)

print(point_x) # [5, 5] ,即全部25个点的x值
print(point_y) # [5, 5] ,即全部25个点的y值

# stack 还原成25个点的坐标， 即[5, 5, 2]
points = tf.stack([point_x, point_y], axis=2)
print(points)

'''


#--------------   一个小例子  -----------------

import matplotlib.pyplot as plt

def func(x):
    # x [500, 500, 2]

    z = tf.math.sin(x[...,0]) + tf.math.sin(x[...,1]) # x[...,0]就是取每个点的x值，x[...,1]就是取每个点的y值
    return z

# 取500个点
x = tf.linspace(0., 2*3.14, 500)
y = tf.linspace(0., 2*3.14, 500)

# 得到坐标
point_x, point_y = tf.meshgrid(x, y)
points = tf.stack([point_x, point_y], axis=2) #[500, 500, 2]

z = func(points)


# 画图的操作
plt.figure('plot 2d func value')
plt.imshow(z, origin='lower', interpolation='none')
plt.colorbar()

plt.figure('plot 2d func contour')
plt.contour(point_x, point_y, z)
plt.colorbar()
plt.show()





