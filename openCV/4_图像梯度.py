import cv2 as cv
import numpy as np


'''
img= cv.imread('images/whiteRound.png')


# ------ 梯度 ----
# 通过梯度，可以得出图像的轮廓
# sobel算子，具体解释见笔记(P7)

# Sobel(src, ddepth, dx, dy, ksize)
# src是图片， ddepth是图像深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
# dx,dy代表是水平方向还是竖直方向的梯度， ksize是sobel算子的大小，即梯度核的大小

print(cv.CV_64F)
# 因为梯度可能有负值，或者大于255的值，这里就是用一个64位的类型。
# 计算x方向的梯度，
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3) # 计算水平方向的梯度，
# 取绝对值，并转化为原来的uint8(8位无符号)
sobelX = cv.convertScaleAbs(sobelX)


# 计算y方向的梯度
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
sobelY = cv.convertScaleAbs(sobelY)

# 将x和y方向上的梯度相加,不要用Sobel(img, cv.CV_64F, 1, 1, ksize=3)一下计算x和y,效果不如分别计算x和y,然后加和
sobelXY = cv.addWeighted(sobelX, 0.5, sobelY, 0.5, 0) # 0.5是对应的权值.最后的0是偏执项，一般为0


'''


'''


# ---  小例子 得出图像轮廓  --
img= cv.imread('images/1.jpg',cv.IMREAD_GRAYSCALE)

sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobelX = cv.convertScaleAbs(sobelX)

sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
sobelY = cv.convertScaleAbs(sobelY)

sobelXY = cv.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

'''

# 三种算子的效果差异
# --- sobel算子
img= cv.imread('images/1.jpg',cv.IMREAD_GRAYSCALE)
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobelX = cv.convertScaleAbs(sobelX)

sobelY = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
sobelY = cv.convertScaleAbs(sobelY)

sobelXY = cv.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

# --- scharr算子
scharrX = cv.Scharr(img, cv.CV_64F, 1, 0)
scharrX = cv.convertScaleAbs(scharrX)

scharrY = cv.Scharr(img, cv.CV_64F, 0, 1)
scharrY = cv.convertScaleAbs(scharrY)

scharrXY = cv.addWeighted(scharrX, 0.5, scharrY, 0.5, 0)

# --- laplacian算子
laplacian = cv.Laplacian(img, cv.CV_64F)
laplacian = cv.convertScaleAbs(laplacian)

res = np.hstack((sobelXY, scharrXY, laplacian))










cv.imshow('round', res)
cv.waitKey(0)
cv.destroyAllWindows()









