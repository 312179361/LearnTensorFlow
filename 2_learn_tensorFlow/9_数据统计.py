'''
    tf.norm 范数
    tf.reduce_min/max 最小最大值
    tf.reduce_min/max 最小最大值位置
    tf.equal 张量比较
    tf.unique
'''
import tensorflow as tf
# tf.norm 范数。
# 二范数。。平方和再开根号||x||2 = [Σx^2]^1/2
# 一范数。。绝对值的和  ||x||1 = Σ|x|
# 无穷范数。。取绝对值中最大的 ||x||∞ = max|x|

a = tf.ones([2, 3])
# a= [1,1
#     1,1]
res = tf.norm(a) # res = (1^2 + 1^2 + 1^2 + 1^2 + 1^2 + 1^2)^1/2 = 根号下6
print(res)

#axis=1, 就是将3当成一个整体去求范数，即第一行是一个整体，其范数是根号3，第二行是一个整体，也是根号3
res2 = tf.norm(a, axis=1) #返回一个list.
print(a)
print(res2)

# ord范数的值，ord=1就是1范数。ord=2就是2范数(默认)
res3 = tf.norm(a, ord=1) # |1|+|1|+|1|+|1|+|1|+|1| = 6
print(res3)

#
res4 = tf.norm(a, ord=1, axis=1) #将a位置为1的shape当成一个整体，求1范数
# 第一行|1|+|1|+|1|=3。第二行一样.结果[3, 3]
print(res4)



# reduce_min/max/mearn 最大值，最小值，均值

b = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) #[3, 4]
res5 = tf.reduce_min(b) # 求b中的最小值->1
res6 = tf.reduce_max(b) # 求b中的最小值->12
res7 = tf.reduce_mean(b) # 求b中的平均值
print(res5, res6, res7)

print(b)
# 加上维度，即将这个维度当成一个整体，算其中的最大最小均值。然后有多少个这样的整体。就返回多少个结果，即返回list
res8 = tf.reduce_min(b, axis=1) #将每行当成整体了，分别算每行的最小值
print(res8)

# reduce_sum 将元素加和。
print(tf.reduce_sum(b,axis=1)) # 将每行当成整体了，分别算每行和


# argmax()  argmin() 最大值、最小值的位置.
res9 = tf.argmin(b) # 没有写axis,即axis=0。将每一列当成一个整体，然后算最小值，然后返回一个降维的位置矩阵。一共有4列，即[0,0,0,0]
print(res9)
res10 = tf.argmin(b, axis=1) # 将每一行当成一个整体，然后算最小值，共有3行，即[0,0,0]
print(res10)

res11 = tf.argmax(b, axis=1) # 最大值的位置[3,3,3]
print(res11)

# tf.equal比较
a1 = tf.constant([1,2,3,4,5])
a2 = tf.constant([1,2,3,4,5])
a3 = tf.constant([0,1,2,3,4])

res12 = tf.equal(a1, a2) # a1和a2逐位比较，都相等，返回全是true
print(res12)
res13 = tf.equal(a1, a3) # a1和a3逐位比较，都不相等，返回全是False
print((res13))

# true代表1，false代表0。cast是类型转换，所以true->1。然后加和为5
print(tf.reduce_sum(tf.cast(res12,dtype=tf.int32)))

# ---------   一个小例子  ---------
# 2个样本，数值代表每个位置的概率，如预测1号位的概率是0.1 。。。
aa = tf.constant([[0.1, 0.2, 0.7],[0.9, 0.05, 0.05]]) # shape=(2,3)
# 通过预测，知道样本1，三号最大0.7。样本2,一号最大0.9
# 按照每行一组，算最大值的位置，然后类型转换。得到预测的结果，
pred = tf.cast(tf.argmax(aa, axis=1), dtype=tf.int32) #[2, 0]
print(pred)

# 真实y值，
y = tf.constant([2,1]) # y=[2, 1]

# 比较y和pred。即比较真实的y和预测的值
ress = tf.equal(y, pred) #[true, false]
print(ress)
# 转换一下比较结果.加和
correct = tf.reduce_sum(tf.cast(ress, dtype=tf.int32)) #结果是1
print(correct)

print(correct / 2) # 结果就是预测准确度，即50%

# ---------   一个小例子，完毕  ---------

# unique去除重复
a4 = tf.range(5) #[0,1,2,3,4]
print(a4)
# 没有重复，即输入本身
a41 = tf.unique(a4)
print(a41)

a5 = tf.constant([4,2,2,3,4,9])
a51 = tf.unique(a5) # 返回去除重复后的值：y=[4,2,3,9]和 每个值对应的位置:idx=[0, 1, 1, 2, 0, 3]。即0->4,1->2,2->3,4->9。这样[0,1,1,2,0,3]->[4,2,2,3,4,9]
print(a51)

# tf.gather(unique, idx) 通过去重后的 和 idx即可恢复原来的张量
# a52 = tf.gather(a51.y, a51.idx)
a52 = tf.gather([4,2,3,9], [0,1,1,2,0,3])
print(a52)
