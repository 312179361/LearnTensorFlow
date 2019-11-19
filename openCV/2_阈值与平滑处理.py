import cv2 as cv
import numpy as np


img = cv.imread('images/1.jpg')
'''


# --------     阈值操作   ---------
# threshold(src, thresh, maxval, type)
# 参数1：图片
# 参数2：阈值
# 参数3：当像素超过了阈值(或者小于阈值，根据type判断)，所赋予的值
# 参数4：类型 有五种
#       THRESH_BINARY 超过阈值的部分去maxval，否则取0
#       THRESH_BINARY_INV  上面的反转
#       THRESH_TRUNC    大于阈值的部分设为阈值， 否则不变
#       THRESH_TOZERO   大于阈值的部分不变， 否则取0
#       THRESH_TOZERO_INV  上面的反转

# 返回值，第一个是阈值，第二个是变换后的图片
ret,img2 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
print(img2)

'''

# -----      滤波操作   --------
# ----- 均值滤波
# 就是将图片的每一个m*m的小矩阵中，求的均值，这样可以使图像更平滑
blurImg = cv.blur(img, (3,3)) # 每个3*3的矩阵中的值做平均。


# ----- 方框滤波
# 第二个参数， ddepth一般为-1，结果和原始普通的颜色通道保持一致
# normalize归一化，True方框滤波和均值是一样的，False即小矩阵中不做均值，只做加和，越界就去255
boxFilterImg = cv.boxFilter(img, -1, (3,3), normalize=False)

# ----- 高斯滤波(正太分布滤波)
# 离小矩阵中间那个越近的权值越大，离得越远的权值越小
gaussImg = cv.GaussianBlur(img, (3, 3), 1) # 1是方差

# ----- 中值滤波
# 取小矩阵中m*m个数值，取中值hstack
medImg = cv.medianBlur(img, 3)

# 可以将多种结果拼接在一起
# res = np.hstack((img,blurImg, gaussImg, medImg)) # hstack横向拼接
res = np.vstack((img,blurImg, gaussImg, medImg))   # vstack纵向拼接


cv.imshow('girl',res)
cv.waitKey(0)
cv.destroyAllWindows()

