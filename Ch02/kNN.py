'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)

Output:     the most popular class label

@author: pbharrin
'''

'''
    k近值算法示例
'''
import matplotlib.pyplot as plt
import operator

from numpy import *
from numpy.ma.core import *

'''
使用k近邻算法改进约会网站的配对效果,算法步骤如下:
(1) 收集数据：提供文本文件。
(2) 准备数据：使用Python解析文本文件。
(3) 分析数据：使用Matplotlib画二维扩散图。
(4) 训练算法：此步骤不适用于k-近邻算法。
(5) 测试算法：使用海伦提供的部分数据作为测试样本。测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
(6) 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
'''


def dating_class_test():
    hoRatio = 0.50  # hold out 10%
    # 从指定文件中载入数据,载入数据为每年获得的飞行常客里程数,玩视频游戏所耗时间百分比,每周消费的冰淇淋公升数
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # print(normMat[0:20])
    # print(ranges)
    # print(minVals)
    # exit()

    m = normMat.shape[0] # 取矩阵行数,rage函数返回包含行列数的元组对象
    numTestVecs = int(m * hoRatio)  # 取测试行数
    errorCount = 0.0
    for i in range(numTestVecs):
        # 取出矩阵每一行,非测试行,非测试行的分类标签
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)  # 求预测分类值
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0  # 预测分类与实际类别不同，则标记为一个错误
    print("the total error rate is: %f, error count is %d" % (errorCount / float(numTestVecs), errorCount))


# 分类器方法,求预测分类值
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 求出测试样本与非测试样本的差值
    sqDiffMat = diffMat ** 2 # 求方
    sqDistances = sqDiffMat.sum(axis=1)  # 行求和
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 将文本文件转换为NumPy矩阵
# Input: 文本文件路径
# Output: 包含训练样本数据的NumPy矩阵和类标签向量
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 得到文件行数
    returnMat = zeros((numberOfLines, 3))  # 创建Numpy矩阵并初始化0
    classLabelVector = []  # 初始化分类标签向量,存放文本行中最后一列分类标签
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()  # 去除回车符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 给矩阵填值
        classLabelVector.append(int(listFromLine[-1]))  # 取出最后一个字段作为标签值存入向量对象
        index += 1
    return returnMat, classLabelVector


# 归一化特征值,即将飞行公里数值转化为[0,1]区间值
# newValue = (oldValue-min)/(max-min)
# dataset: NumPy矩阵
# 返回值：归一化的numPy矩阵, 最大最小飞行公里数的差值行, 最小矩阵行
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals  # 取最大最小飞行公里数的差值
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))  # 矩阵每一行都与最小矩阵行做差值运算
    normDataSet = normDataSet / tile(ranges, (m, 1))  # element wise divide,上一步计算出的矩阵每一行去除最大最小插值矩阵
    return normDataSet, ranges, minVals




# 读取NumPy矩阵格式的特征值，显示为散列图
def test1():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    print(datingDataMat[0:20])
    print(datingLabels[0:20])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))  # 玩视频游戏所占百分比,每周消耗的冰淇淋公升数
    # plt.show()
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))  # 玩视频游戏所占百分比,每周消耗的冰淇淋公升数
    plt.show()

def test2():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(normMat)
    print(minVals)
    print(ranges)
    print(normMat.shape)


# print("========================================================")
# test1()
# print("========================================================")
# test2()

dating_class_test()










