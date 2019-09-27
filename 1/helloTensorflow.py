import tensorflow as tf



'''
hello = tf.constant('hello tensorflow')
sess = tf.Session()
print(sess.run(hello))
'''


'''
# 常量
tf.constant()
# 占位符
tf.placeholder
# 变量--一种特殊的张量，

**  普通的张量的生命周期依赖计算完成而结束，内存也释放
    变量会常驻内容，在每一步训练时不断更新其值，实现模型的参数更新 
tf.Variable
'''


'''
# 0阶张量--标量
# 定义变量 tf.Variable(值, name(命名，可选))
mammal = tf.Variable("Elephant")

ignition = tf.Variable(451, tf.int16) # 还可以指定类型，也会自动判断，tf.int16
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

sess.run(tf.global_variables_initializer())


# 给变量赋值，assign或者assign_xxx
ignition = ignition.assign(ignition + 5)
# ignition = ignition.assign_add(6)

print(sess.run(mammal))
print(sess.run(ignition))
print(sess.run(floating))
print(sess.run(its_complicated))


# 1阶张量--矢量
mammal2 = tf.constant([3.14, 2.71])
print(sess.run(mammal2))

# 2阶张量--矩阵
mammal3 = tf.constant([[1,2],[3,4]])
# 获取对象的阶数
# r = tf.rank(mammal3)
# 切片，即获取某一单元。0阶不需要索引，1阶是一个索引，2阶是两个索引。如果2阶传递一个索引，就会返回一个1阶的
# my_scalar = mammal3[1, 2]

print(sess.run(mammal3))
# print(r)


# 创建变量
w = tf.Variable([[1, 2], [3, 4], [5, 6]], name='w')
# 创建会话
sess = tf.compat.v1.Session()
# 使用global_variables_initializer初始化变量w
sess.run(tf.global_variables_initializer())

print(sess.run(w))

# 更新变量w的值.一下两种方式都可以
 w1 = tf.assign_add(w, [[1,2],[3,4],[5,6]])
# w1 = w.assign_add([[1,2],[3,4],[5,6]])

print(sess.run(w1))

'''
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('GPU',tf.test.is_gpu_available())

a = tf.constant(2)
b = tf.constant(3)
print(a+b)