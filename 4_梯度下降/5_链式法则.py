# 链式法则就是复合函数的求导法则
# (f⭕️ n)'(x) = f'(u)n'(x)
import tensorflow as tf

x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, b1, w2, b2])
    # 复合函数，y2 = f(y1) y1 = g(x)
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

# y2对y1的偏导.结果是2
dy2_dy1 = tape.gradient(y2, [y1])[0]
print(dy2_dy1)
# y1对w1的偏导。结果是1
dy1_dw1 = tape.gradient(y1, [w1])[0]
print(dy1_dw1)

# 直接y2对w1的偏导。结果是2。符合链式法则。dy2_dw1=dy2_dy1 * dy1_dw1
dy2_dw1 = tape.gradient(y2, [w1])[0]
print(dy2_dw1)

