# 张量合并 concat()。concat操作不会增加维度，dim不会改变。几维还是几维
import tensorflow as tf

a = tf.ones([4, 35, 8])
b = tf.ones([2, 35, 8])
# concat,将a和b两个张量，合并成一个张量。axis是在哪个维度合并。注意，除了合并的维度，其他维度大小要一样
c = tf.concat([a, b], axis=0)  # 在第0个维度上合并--> [4+2, 35, 8]
print(c.shape)

a1 = tf.ones([4, 32, 8])
b1 = tf.ones([4, 3, 8])
c1 = tf.concat([a1, b1], axis=1)  # 在第0个维度上合并--> [4, 32+3, 9]
print(c1.shape)

# 张量合并stack,会增加维度，dim会改变。保证合并的张量的shape要相等，即a2和b2的shape要相等
a2 = tf.ones([4, 35, 8])
b2 = tf.ones([4, 35, 8])

c2 = tf.stack([a2, b2], axis=0)  # 会在第0个位置前，增加一个维度[2, 4, 35, 8]
print(c2.shape)
c3 = tf.stack([a2, b2], axis=2)  # 会在第2个位置前，增加一个维度[4, 35, 2, 8]
print(c3.shape)

# 切割张量unstack，按照axis这个维度上打散。这样就降维了
# 将[2, 4, 35, 8]-> 两个[4, 35, 8]
res = tf.unstack(c2, axis=0)
print(len(res))
print(res[0].shape)
res1 = tf.unstack(c2, axis=3)  # 打成8个[2,4,35]
print(len(res1))
print(res1[1].shape)

# split,切割张量.可以在某个维度上灵活切割
# num_or_sizr_splits=2将其维度平均分成两份。[2,4,35,8]->[2,4,35,8/2]
res2 = tf.split(c2, axis=3, num_or_size_splits=2)
print(len(res2))
print(res2[0].shape)
print(res2[1].shape)

# num_or_sizr_splits=[2,2,4]将其维度分成2,2,4三份。[2,4,35,8]->[2,4,35,2]和[2,4,35,2]和[2,4,35,4]
res3 = tf.split(c2, axis=3, num_or_size_splits=[2, 2, 4])
print(len(res3))
print(res3[0].shape)
print(res3[2].shape)
