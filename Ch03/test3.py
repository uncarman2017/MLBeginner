# 预测隐形眼镜类型的例子
# 步骤
# (1) 收集数据：提供的文本文件。
# (2) 准备数据：解析tab键分隔的数据行。
# (3) 分析数据：快速检查数据，确保正确地解析数据内容，使用createPlot()函数绘制最终的树形图。
# (4) 训练算法：使用3.1节的createTree()函数。
# (5) 测试算法：编写测试函数验证决策树可以正确分类给定的数据实例。
# (6) 使用算法：存储树的数据结构，以便下次使用时无需重新构造树。

import unittest

from Ch03 import trees
from Ch03 import treePlotter


class MyTestCase(unittest.TestCase):
    def test_lense1(self):
        fr = open('lenses.txt', 'r')
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = trees.createTree(lenses, lensesLabels)
        print(lensesTree)
        treePlotter.createPlot(lensesTree)

