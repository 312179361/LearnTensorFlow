import numpy as np

'''
计算误差，即 compute loss
将给定的x,y值，带入(wx+b-y)^2中，平方是为了保持正数,其中w,b最开始是0，
然后通过不断学习，w,b会越来越趋近真实值

将所有的点(x,y)带入后，然后累加，除以个数，就可以了
详情见笔记1
'''
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        # 从数据源中得到第i个点
        x = points[i, 0]
        y = points[i, 1]
        # 将点i带入(wx+b-y)^2中，然后累加. **代表平方
        totalError = ((w * x + b) - y) ** 2

    return totalError / float(len(points))



# 计算梯度和更新，即计算w 和 b。详情见笔记2
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        # 计算偏微分 loss对b的偏微分
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        # 计算偏微分 loss对w的偏微分
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)

    # 更新w 和 b
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


'''
循环学习，
将得到的新w和新b,当做w和b,代入，计算更新的w和b，一直不断的逼近极小值
'''

def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):

    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        # 执行迭代函数，然后返回的b和w，在代入，进行迭代
        # print(points)
        # b, w = step_gradient(b, w, np.array(points), learning_rate)
        b, w = step_gradient(b, w, points, learning_rate)

    return [b, w]


def run():

    points = np.genfromtxt("data.csv", delimiter=",")
    # 衰减因子，前进步长
    learning_rate = 0.0001

    # 初始化b 和 w
    init_b = 0
    init_w = 0

    # 迭代次数
    num_iterations = 1000

    print("开始梯度下降 b={0}, w={1}, error={2}".format(init_b, init_w, compute_error_for_line_given_points(init_b, init_w, points) ))

    # 开始学习
    print("Running...")
    [b, w] = gradient_descent_runner(points, init_b, init_w, learning_rate, num_iterations)


    #迭代后的结果,即迭代后，loss的误差值

    print("迭代后{0}次， b={1}, w={2}, error={3}".format(num_iterations,b,w, compute_error_for_line_given_points(b, w, points)))



if __name__ == '__main__':
        run()