'''
+ - * / 加减乘除

**多少次方  pow  square平方

sqrt平方根

//整除  %取余

exp指数  log对数

@  matmul 矩阵相乘

'''
import tensorflow as tf

# + - * / % //  这些都是对应项相运算，
a = tf.ones([2, 2])  # 2x2,数值都是2的矩阵
b = tf.fill([2,2], 2.)  # 2x2,数值都是1的矩阵

print(a + b)
print(a - b)
print(a * b)
print(a / b)

print(b // a) # 整除，舍去余数
print(b % a) #取余

# exp指数，log对数
print(tf.math.log(b)) # 相当于矩阵的每一项，都以e为底数，求对数。lneb

#tf.math.exp()和tf.exp()一样
print(tf.math.exp(a)) # 相当于矩阵的每一项，都以e为底数，求指数 e^a
print(tf.exp(a))

# 如果想求对数是其他底数，就用公式得出，

# pow  ** 都是求n次方的
print(tf.pow(b, 3)) # 给b矩阵的每一项求3次方
print(b**3)

# sqrt开平方根
print(tf.sqrt(b))  # 给b矩阵的每一项开平方

#---------以上运算都是对应项进行运算----------------------
c = tf.constant([[1,2],[3,4]])
d = tf.constant([[1,2],[3,4]])
# @ 和 matmul  都是矩阵相乘，即 行乘列相加
print(c@d)
print(tf.matmul(c,d))


# 三维的@或者matmul,是后两维进行矩阵操作，前面的当成长度
e = tf.ones([4,2,3])
f = tf.fill([4,3,5],2.)
# 将e的后两维[2,3]和f的后两维[3,5]进行运算
# 得到的是[4,2,5]
print(e@f)

# 如果前面的不一样，但是有一个是1，就可以通过broadcasting进行补充，然后进行运算
e1 = tf.ones([1,2,3])
f1 = tf.fill([4,3,5],2.)
# 得到的是[4,2,5]
print(e1@f1)

e2 = tf.ones([2,1,2,3])
f2 = tf.fill([1,4,3,5],2.)
# 得到的是[2,4,2,5]
print(e2@f2)

e3 = tf.ones([2,1,2,3])
f3 = tf.fill([4,3,5],2.)
# e3 [2,1,2,3]->[2,4,2,3]
# f3 [4,3,5]->[1,4,3,5]->[2,4,3,5]
# 得到的是[2,4,2,5]
print(e3@f3)


# 例子：Y = X@W + b

x = tf.ones([4, 2])
w = tf.ones([2, 1])
b = tf.constant(0.1)
# x@w的shape是[4, 1]。b通过自动broadcasting变成[4, 1]。然后相加
print(x@w + b)



