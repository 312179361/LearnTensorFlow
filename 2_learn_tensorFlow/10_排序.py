import tensorflow as tf


'''
a = tf.random.shuffle(tf.range(5))
print(a)

#sort 全排序
a1 = tf.sort(a) # direction默认是升序，
a2 = tf.sort(a, direction='DESCENDING') #direction='DESCENDING'是降序排列
print(a1)
print(a2)

# argsort
idx = tf.argsort(a) # 返回元素从小到大的下标
idx1 = tf.argsort(a, direction='DESCENDING')  # 返回元素从大到小的下标
print(idx)
print(idx1)

# 利用gather 可以将乱序的tensor和下标，进行排序
a3 = tf.gather(a, idx) # 通过从小到大的下标，可以进行排序
print(a3)

# 高纬度排序
a4 = tf.random.uniform([3, 3],maxval=10, dtype=tf.int32) # 创建一个平均分布的3x3矩阵，最大值是9
print(a4)

a5 = tf.sort(a4) # 对其最后一个维度，即每一行内进行升序排序。
print(a5)
a6 = tf.sort(a4, direction='DESCENDING') # 每一行内进行降序排序。
print(a6)

id2 = tf.argsort(a4) # 每一行内，从小到大的下标
print(id2)
id3 = tf.argsort(a4, direction='DESCENDING') # 每一行内，从大到小的下标
print(id3)

print('--------------------')


print(a)

#  top_k 可以得到最大的几个值，或者最小的几个值。多维的就是在一行内进行排序
res = tf.math.top_k(a, 2) # 返回最大的两个值
print(res.indices) # indices是得到索引值，即最大两个值的索引值
print(res.values) # value得到的是具体的值，即最大两个值具体是多少


#----------  一个小例子 top_k Accuracy --------------
def accuracy (output, target, top_k=(1,)): # 参数：样本， 目标值， top_k，即需要返回top_k的值

    maxk = max(top_k) # 去top_k的最大值，就得到需要取前几项
    batch_size = target.shape[0] #一共有多少种可能，即6种可能

    # 得到样本从大到小，前n项值的下标
    preidx = tf.math.top_k(output, maxk).indices # [10, k]
    # 对其转置。第一行就是每个样本概率最大的。第二行是概率次大的，依次类推
    preidx = tf.transpose(preidx, perm=[1, 0]) #[k, 10]

    # 将样本和目标值进行比较
    # 由于目标值是[10]，所以需要broadcast [10]->[k, 10]
    target_ = tf.broadcast_to(target, preidx.shape)
    # 比较
    correct = tf.equal(preidx, target_) #[2, 10]

    print('correct:',correct)

    res = []
    for k in  top_k:

        # cast将true变为1，false变为0。reshape,维度变换reshape([:k],[-1])就是先切片，得到前k行(由于含头不含尾，所以k=1取的是第0行。k=2取的是0和1行)。然后将前k行变换成一行
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        print('corr_k：',correct_k)
        correct_k = tf.reduce_sum(correct_k) # 前k行求和
        acc = float(correct_k / batch_size) # 加和后/总可能数。就是top_k的可能性
        res.append(acc)

    return res









# 随机生成一个10*6的矩阵，当做样本，即有10个样本。6代表每个样本是这6种可能性的概率
output = tf.random.normal([10, 6])
# softmax代表在第1号这个维度上几个值，加起来等于1.
output = tf.math.softmax(output, axis=1)  # 这里为了保证每个样本的6种可能性概率相加为1
print('样本:',output)

# 目标值，即真实值.随机生成一个目标值，最大是5。因为有10个样本，所以shape为10
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
print('目标值：',target)

# 调用函数accuracy，来计算top-k的值

acc = accuracy(output, target, (1,2,3,4,5,6))
print('top-1-6:',acc)




'''

output = tf.random.normal([10, 6])


print(output)
print(output[0])
print()


