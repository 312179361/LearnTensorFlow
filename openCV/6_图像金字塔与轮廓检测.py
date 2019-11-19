import cv2 as cv
import numpy as np
'''
img = cv.imread('images/1.jpg')



# 高斯金字塔，原理见笔记(p8)


print(img.shape)

# 利用高斯金字塔，上采样。即放大，shape变大了
up = cv.pyrUp(img)
print(up.shape)

# 利用高斯金字塔，下采样。即缩小，shape变小了
down = cv.pyrDown(img)
print(down.shape)



# --- 拉普拉斯金字塔
# 先做下采样
down = cv.pyrDown(img)
# 再做上采样
down_up = cv.pyrUp(down)
# 和原始数据 做差
llres = img - down_up

'''

# 获取图像轮廓。

# 读取图片
img = cv.imread('images/blackWord.png')
# 变换成灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 变换成二值图。利用二值图，容易提高准确率
# 阈值操作。超过阈值(127)的部分取maxVal(255)，小于127的取0。
ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

# 利用边缘检测
# findContours(图像, mode, method)
# mode轮廓检索模式
#    RETR_EXTERNAL 只检索最外面的轮廓
#    RETR_LIST     检索所有轮廓，并将其保存到一条链表中
#    RETR_CCOMP    检测所有轮廓，并将他们组织为两层，顶层是外部边界，第二层是空洞的边界
#    RETR_TREE     检测所有轮廓，并重构检讨轮廓的整个层次(最常用)
#
# method轮廓逼近方法(最常用的两种)
#    CHAIN_APPROX_NONE      以freeman链吗的方式输出轮廓，所有其他方法输出多边形(顶点序列)
#    CHAIN_APPROX_SIMPLE    压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分

# 三个返回值，
# contours  轮廓信息，是list类型
# hierarchy 层级
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# 绘制轮廓
draw_img = img.copy() # 不copy下面的操作就会修改原图
# 绘制，drawContours(原图, 要绘制的点信息, 要绘制多少(-1是绘制所有轮廓), 颜色 ,线条宽度)
res = cv.drawContours(draw_img, contours, -1, (0,0,255),3) # 用红色的绘制，线条宽度为2，绘制所有的轮廓信息



res2 = np.hstack((img, res))


cv.imshow('girl', res2)
cv.waitKey(0)
cv.destroyAllWindows()