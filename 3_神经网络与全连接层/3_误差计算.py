'''
MSE 平方差的和， loss = Σ(y-out)^2 / N 。即输出y和真实y相减,的平方和，在除以个数
                L2-norm = √￣Σ(y-out)^2 。即输出y和真实y相减,的平方和。在整体开一个根号


Cross Entropy Loss

Hinge Loss

'''
import tensorflow as tf

'''
y = tf.constant([1, 2, 3, 0, 2])
y = tf.one_hot(y, depth=4) # 分出4类， 即[1000],[0100],[0010],[0001]
print(y)
# cast()类型转换，将其转换为int32
y = tf.cast(y, dtype=tf.int32)
print(y)

# 模拟out
out = tf.random.normal([5, 4])
# 三种方式求MSE
#square平方。reduce_mean平均值
loss1 = tf.reduce_mean(tf.square(y-out))
print(loss1)
# norm就是√￣Σ(y-out)^2
loss2 = tf.square(tf.norm(y-out))/(5*4)
# tf.losses.MSE
loss3 = tf.reduce_mean(tf.losses.MSE(y, out))

'''

#-------------------------
# 熵Entropy 熵越小--信息越多(越不稳定)
# H(p) = - ΣP(i)logP(i) = ∫PlogP

a = tf.fill([4], 0.25) # 创建4个0.25
print(a)

# log是以2为底的，所以要转换一下 tf.math.log(a)/tf.math.log(2.)
b = a*tf.math.log(a)/tf.math.log(2.)  # PlogP
print(b)
# 加合,求负号，
res = -tf.reduce_sum(b)
print(res)    #熵是2，最大的。最稳定

a1 = tf.constant([0.1,0.1,0.1,0.7])
b1 = a1*tf.math.log(a1)/tf.math.log(2.)
res1 = -tf.reduce_sum(b1)
print(res1)      # 熵是1.3567797

a2 = tf.constant([0.01,0.01,0.01,0.97])
b2 = a2*tf.math.log(a2)/tf.math.log(2.)
res2 = -tf.reduce_sum(b2)
print(res2) # 熵是0.24194068就很小了，最不稳的


# 交叉熵.针对两个，一个p真实值,一个q预测值
# H(p,q) = -Σp(i)logq(i) = H(p) + Dkl(p|q)
# 其中当p=q时，Dkl(p|q)=0

# p如果是one-hot encoding
# H(p:[0,1,0]) = -1log1 = 0
# H([0,1,0],[p0,p1,p2]) = 0 + Dkl(p|q) = -1logqi
# 此时H(p|q)就是Dkl(p|q)了，如果Dkl(p|q)最小，说明误差越小，如果p=q就是没有误差
# 所以H(p|q)可以作为loss函数，作为误差函数

# 例子 只有cat和dog两种可能 所以P(dog) = 1 - P(cat)
# H(p,q) = -Σp(i)logq(i) =  -P(cat)logQ(cat) - P(dog)logQ(dog)
#        = -P(cat)logQ(cat) - (1 - P(cat))log(1-Q(cat))
#     令y = P(cat),Q(cat) = p
#        上式 = -ylogp - (1-y)log(1-p) = - (ylogp + (1-y)log(1-p))



'''
如P1 = [1 0 0 0 0],即一张图片的是第一类的
Q1 = [0.4 0.3 0.05 0.05 0.2] 通过预测，得到这个图片在每个分类的概率

H(P1, Q1) = -ΣP(i)logQ(i) 
          = -(1log0.4 + 0log0.3 + 0log0.05 + 0log0.05 +0log0.2)
          = -log0.4 ≈0.916   这个熵离0还有一定的距离
          
Q2 = [0.98 0.01 0 0 0.01] 

H(P1, Q2) = -ΣP(i)logQ(i) 
          = -(1log0.98 + 0log0.01 + 0log0 + 0log0 +0log0.01)
          = -log0.98 ≈0.02   这个熵离0就很接近了，这样误差loss就小很多
          
'''

# 利用tf.losses.categorical_crossentropy(p,q),p是真实值，q是预测值,求交叉熵。
a = tf.losses.categorical_crossentropy([0,1,0,0], [0.25,0.25,0.25,0.25])
print(a)
a1 = tf.losses.categorical_crossentropy([0,1,0,0], [0.1,0.1,0.8,0.1])
print(a1)
a2 = tf.losses.categorical_crossentropy([0,1,0,0], [0.1,0.7,0.1,0.1])
print(a2)
a3 = tf.losses.categorical_crossentropy([0,1,0,0], [0.01,0.97,0.01,0.01])
print(a3)

# 利用tf.losses.CategoricalCrossentropy()(p,q),p是真实值，q是预测值,求交叉熵。 和上面的效果一样
b = tf.losses.CategoricalCrossentropy()([0,1,0,0], [0.01,0.97,0.01,0.01])
print(b)

# 对于二进制，即两种选择的话可以用BinaryCrossentropy或者binary_crossentropy
c = tf.losses.BinaryCrossentropy()([1], [0.1]) # 代表预测正确的概率是0.1
print(c)
c1 = tf.losses.binary_crossentropy([1], [0.1])
print(c1)




'''
正常流程
1、得到原始数据.即x值
2、通过w和b,利用线性回归x@w+b得到logits值，即得到y值
3、利用softmax变换到0~1中，且各个概率相加为1
4、利用crossEntropy和真实值(one-hot)交叉熵，计算loss

其中3和4 两个步骤TensorFlow可以合并，多一个参数 from_logits=True。即tf.losses.categorical_crossentropy([0,1], logits, from_logits=True)
这样可以将logits直接当做参数传递，并且不会产生数据不稳定

如果自己利用3和4，理论上是正确的，但是可能会产生数据不稳定的情况，所以不建议
'''
x = tf.random.normal([1, 784])
w = tf.random.normal([784, 2])
b = tf.zeros([2])
logits = x@w + b
print(logits)

# 加上from_logits=True就可以避免数据不稳定，并且参数直接传未经softmax处理的logits
res3 = tf.losses.categorical_crossentropy([0,1], logits, from_logits=True)
print(res3)

# 这样理论可以，但是不建议。可能会产生数据不稳定的情况
prob = tf.math.softmax(logits,axis=1)
print(prob)
res4 = tf.losses.categorical_crossentropy([0,1], prob)
print(res4)


