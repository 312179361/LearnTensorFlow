import numpy as np
from    mpl_toolkits.mplot3d import Axes3D
from    matplotlib import pyplot as plt
import  tensorflow as tf

# 求f(x,y) = (x^2 +y-11)^2 + (x+y^2 -7)^2的最小值

# 声明出来这个函数
def himmelblau(x):
    #
    return (x[0] ** 2 +x[1] - 11) ** 2 +(x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)
print('X,Y maps:',X.shape, Y.shape)
Z = himmelblau([X, Y])
# 画出图像
fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()



# 开始求f(x,y)的最小值
# 给坐标一个初值
x = tf.constant([4, 0.])

# 让梯度下降这个过程重复200次，
for step in range(200):

    with tf.GradientTape() as tape:
        tape.watch([x])
        # loss函数，即函数本身
        y = himmelblau(x)
    # 求梯度
    grads = tape.gradient(y, [x])[0]
    # 通过梯度，去逼近最小值的坐标
    x -= 0.01*grads

    # 输出结果
    if step % 20 == 0:

        print('step{}:grads={},x = {},f(x) = {}'.format(step,grads , x.numpy(), y.numpy))



