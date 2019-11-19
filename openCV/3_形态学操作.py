import cv2 as cv
import numpy as np

img = cv.imread('images/whiteWord.png')
'''

# erode腐蚀。 在腐蚀内核中，取最小值，这样可以把边缘的有0，有255 的区域全部变成0，这样黑色区域就变多了，相反白色字就瘦身了
kernel = np.ones((3,3), np.uint8)
print(kernel)
# kernel是指定腐蚀内核， iterations是腐蚀次数
img = cv.erode(img, kernel, iterations=1)
'''

'''

# dilate膨胀。 和腐蚀正好相反
kernel2 = np.ones((3,3), np.uint8)
img = cv.dilate(img, kernel2, iterations=3)
'''


'''
kernel = np.ones((3,3), np.uint8)
# 开运算，先腐蚀 再膨胀
# img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# 闭运算，先膨胀 再腐蚀
img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
'''


'''
# 梯度 = 膨胀的图片 - 腐蚀的图片
kernel = np.ones((7,7), np.uint8)

img2 = cv.dilate(img, kernel, iterations=1)
img3 = cv.erode(img, kernel, iterations=1)


# 梯度，就是将两个图片相减，这样就得到边框
res = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# 将他们拼接在一起，作对比
res2 = np.hstack((img2, img3, res))

'''

''' 
# 礼帽操作 = 原始图片 - 开运算操作 
kernel = np.ones((7,7), np.uint8) 
res1 = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)


# 黑帽操作 = 闭运算 - 原始图片
res2 = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

# 将他们拼接在一起，作对比
res3 = np.hstack((res1, res2))


'''



cv.imshow('word',res3)


cv.waitKey(0)
cv.destroyAllWindows()

