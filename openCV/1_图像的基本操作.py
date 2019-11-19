import cv2 as cv
import numpy as np
import matplotlib as plt

# 封装的图片显示函数
def cv_show(name, img):
    print('222')
    print(img.size)
    # 显示图像，第一个参数是窗口的名字，第二个是图像
    cv.imshow(name, img)
    # 等待时间，毫秒级，0代表键盘的任意键消失
    cv.waitKey(0)
    # 销毁
    cv.destroyAllWindows()


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)

# 视频的读取
def cv_video_show():
    # 如果传0或1就是获取摄像头，如果填写路径，就是视频文件
    capture = cv.VideoCapture('images/daomeixiong001.mp4')
    # 查看是否可以打开
    if capture.isOpened():
        # 读入视频的一帧.返回两个值，open是bool值，表示是否这一帧可以读入
        # frame是这一帧的图像
        open, frame = capture.read()

        # 循环每一帧，得到视频
        while open:
            ret, fram = capture.read()
            if fram is None: # 如果这一帧为空，就break
                break

            # 如果可以读取
            if ret == True:
                # 进行图像处理
                gray = cv.cvtColor(fram, cv.COLOR_BGR2GRAY) # 将这一帧转换成灰度
                # 显示
                cv.imshow('result',gray)

                # cv.wawaitKey(10)代表这一帧多久执行完，进入下一帧，越小视频播放速度越快
                # 27是退出键的编码
                if cv.waitKey(1) & 0xFF == 27:
                    break
        cv.destroyAllWindows()


# 对图片进行切片
def imgCut(img):
    img = img[0:200, 0:200]



# 颜色通道的提取 。注意顺序是b,g,r
def colorSplit(img):
    b, g, r = cv.split(img)
    print(b)
    print(b.shape)


def changeColor(img):

    # 将第三个bgr通道做修改
    img[:,:,0] = 0 # b通道设为0
    img[:,:,1] = 0 # g通道设为0


def mergeColor(b, g, r):
    # merge就是将b,g,r三个通道结合到一起
    img = cv.merge((b, g, r))
    print(img.shape)



# 边界扩展填充
def slce(img):

    # BORDER_REFLICATE　　　  # 直接用边界的颜色填充， aaaaaa | abcdefg | gggg
    # BORDER_REFLECT　　　　  # 倒映，abcdefg | gfedcbamn | nmabcd
    # BORDER_REFLECT_101　　 # 倒映，和上面类似，但在倒映时，会把边界空开，abcdefg | egfedcbamne | nmabcd
    # BORDER_WRAP　　　　  　# 额。类似于这种方式abcdf | mmabcdf | mmabcd
    # BORDER_CONSTANT　　　　# 常量，增加的变量通通为value色 [value][value] | abcdef | [value][value][value]

    img2 = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_REPLICATE)
    print('111')

    return img2

# 图像融合
def fusionImg(img1, img2):
    print(img1.shape)
    print(img2.shape)

    # 两个图片想融合，需要将shape变换成一样的
    # resize,shape变换
    img2 = cv.resize(img2, (600,400))

    # resize。如果前面写(0,0) 后面写fx=n,fy=m。即x*n y*m
    # img1 = cv.resize(img1, (0,0),  fx=3,fy=1) # 即x变换成原来的3倍，y变换成原来的1倍

    print(img2.shape)

    # 图像融合 cv.addWeighted(img1, a, img2, b, c) a是img1的权重，b是img2的权重，c是偏执。
    # 利用公式计算 res = a*img1 + b*img2 + b

    res = cv.addWeighted(img1, 0.4, img2, 0.6, 0)

    return res





# 读取图片。第二个参数是图像类型cv.IMREAD_COLOR,是彩色图(默认)。cv.IMREAD_GRAYSCALE是灰度图
img = cv.imread('images/1.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('images/2.png', cv.IMREAD_COLOR)
# print(img)
print('-----')

'''

# 数学运算，是给每个像素点的值增加或者减小。如果超出255，就继续从0开始计算。如果小于0，就继续从255开始计算
img = img-10  # 给每个点的值都减10
print(img)

# add函数，可以将两个相同维度的img相加。与上面的+号运算不同的是，如果超出255，就停止了，不继续从0开始加了
img = cv.add(img, img)

'''

# 图像融合
res = fusionImg(img, img2)



# 调用图片显示
cv_show('gi', res)

# 调用视频显示
# cv_video_show()


'''
# 图像的保存
cv.imwrite('girl.png',img)
'''





