import  cv2 as cv
import numpy as np
'''
Canny边缘检测 原理
1、使用高斯滤波器，以平滑图像，滤除噪声
2、计算图像中每个像素点的梯度强度和方向
3、应用非极大值抑制，以消除边缘检测带来的杂散响应。即丢弃非极大值的，取最大值的
4、应用双阈值检测确定真实的和潜在的边缘。
   梯度值 > maxVal，处理为边界
   minVal < 梯度值 < macVal，连有边界的则保留。否则舍弃
   梯度值 < minVal，则舍弃，不是边界
5、通过抑制孤立的弱边缘最终完成边缘检测
'''

img = cv.imread('images/1.jpg', cv.IMREAD_GRAYSCALE)

# 利用Canny边缘检测。 Canny(图片,双阈值minVal, maxVal)
v1 = cv.Canny(img, 100, 250) # 双阈值较高，说明要求高，即梯度比较高的才会被当做边缘
v2 = cv.Canny(img, 50, 100) # 双阈值较低，说明要求高，即梯度不是很高的就会被当做边缘

res = np.hstack((v1, v2))

cv.imshow('girl', res)
cv.waitKey(0)
cv.destroyAllWindows()



