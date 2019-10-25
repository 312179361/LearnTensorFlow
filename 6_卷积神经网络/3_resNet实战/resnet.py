import tensorflow as tf
from tensorflow import keras
from    tensorflow.keras import layers, Sequential

# 创建一个BasicBlock
class BasicBlock(layers.Layer):

    # 初始化。filter_num就是kernel的数量，strids步长
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # 一个卷积层
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        # 一个batchNorm
        self.bn1 = layers.BatchNormalization()
        # 一个激活函数
        self.relu = layers.Activation('relu')

        # 卷积层 batchNorm 激活函数，这三层是一个Unti

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()


        # 如果strids步长不为1，就需要downsample(identity)层
        if stride != 1:
            # identity层
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))

        else:
            # 如果为1，就原状返回
            self.downsample = lambda x:x



    # call 函数
    def call(self, inputs, training=None):

        # 将input经过第一个unit
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        # 经过第二个unit
        out = self.conv2(out)
        out = self.bn2(out)

        # 将input经过identity层
        identity = self.downsample(inputs)

        # 将卷积层和identity层加和
        output = layers.add([out, identity])
        # 经历一个激活函数
        output = tf.nn.relu(output)

        return output


# 创建ResNet类。resNet的基本单元是resblock。
class ResNet(keras.Model):
    '''
    如：layer_dims=[2,2,2,2]。代表
    一共有4个resBlock
    每一个resBlcok分别有2个basicBlock

    num_classes  全连接层输出。代表多少类，默认100个类
    '''
    def __init__(self, layer_dims, num_classes=100):

        super(ResNet, self).__init__()

        #第一步：预处理层

        self.stem = Sequential([layers.Conv2D(64, (3,3),strides=(1,1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                # 池化层
                                layers.MaxPool2D(pool_size=(2, 2),strides=(1,1), padding='same')
                                ])



        #第二步：创建中间的resBlock，调用build_resblock函数。一共创建4个resblock
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1],stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2],stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3],stride=2)


        # 第三步:全链接层
        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc = layers.Dense(num_classes)




    def call(self, inputs, training=None):
        # 运算
        # 第一步,预处理
        x = self.stem(inputs)
        # 第二步，四个resblock
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 第三步
        x = self.avgpool(x) #[b, c]
        x = self.fc(x) #[b, 100]

        return x


    #多个basicBlock组成resBlock。 blocks是一个resblock有多少个basicblock
    def build_resblock(self, filter_number, blocks, stride=1):

        res_blocks = Sequential()
        # 在res_blocks容器中添加basicBlock
        # 第一个basicBlock是下采样，即strid不为1
        res_blocks.add(BasicBlock(filter_number, stride))

        # 其他basicBlock就是普通的，即strid=1
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_number, stride=1))

        return res_blocks


'''
# 创建新函数，resnet18
resnet18 是有
一个预处理层，，，，，，1层
一个全链接层，，，，，，1层
4个resBLock，每个resBlock有两个BasicBlock,而每个basic中有两个Unit(即卷积功能)，，，4*2*2=16

一共18层，即resnet18
'''
def resnet18():
    # 这里的参数代表，一共有4个resBlock，每个resBlock有2个basicblock
    return ResNet([2, 2, 2, 2])

# 创建新函数，resnet34
def resnet34():
    # 这里的参数代表，一共有4个resBlock，每个resBlock分别有3，4，6，3个basicblock
    return ResNet([3, 4, 6, 3])