import tensorflow as tf

a = tf.range(10)
print(a)
# maximum(),取较大的数
a1 = tf.maximum(a, 7) # a中的元素和7相比较，取较大的，即比7小的都替换成7
print(a1)
# minimum(),取较小的数
a2 = tf.minimum(a, 3) # a中的元素和7相比较，取较小的，即比3大的都替换成7
print(a2)
# clip_by_value(),取这个范围内的数
a3 = tf.clip_by_value(a, 2,8) #a中的元素小于2的替换成2，大于8的替换成8
print(a3)


# relu函数，x小于0的y是0,x大于0的y=x
a = a-5
print(tf.nn.relu(a))

# 自己实现relu函数
tempRelu = tf.maximum(a, 0)
print(tempRelu)


#clip_by_norm() ...根据一个范数进行裁剪

b = tf.constant([[2.,5.],[8.,3.]])
print(tf.norm(b)) # 求范数

# 根据范数，来裁剪数值
b1 = tf.clip_by_norm(b, 7) # 将范数规定在7左右，然后对每个数值进行同步裁剪
print(b1)
print(tf.norm(b1))


#clip_by_global_norm() ...也是一个范数进行裁剪，对所有的tensor的所有数值都进行同步裁剪



