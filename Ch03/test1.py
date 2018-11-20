import unittest

from Ch03 import trees


class MyTestCase(unittest.TestCase):
    def test_trees1(self):
        print('============计算香农熵的测试=============')
        dataset, labels = trees.createDataSet()
        print(dataset)
        shannonEnt = trees.calcShannonEnt(dataset)
        print("香农熵=" + str(shannonEnt))
        # 改变数据集，熵有变化
        dataset[0][-1] = "maybe"
        shannonEnt = trees.calcShannonEnt(dataset)
        print("香农熵=" + str(shannonEnt))
        self.assertTrue(shannonEnt > 0, "shannonEnt不能小于0")

    def test_trees2(self):
        print('============拆分数据集的测试=============')
        dataset, labels = trees.createDataSet()
        ds = trees.splitDataSet(dataset, 0, 1)
        print(ds)
        ds = trees.splitDataSet(dataset, 0, 0)
        print(ds)
        self.assertIsNotNone(ds, "ds is null")

    def test_tree3(self):
        print('============计算最优特征值的测试=============')
        dataset, labels = trees.createDataSet()
        bestFeature = trees.chooseBestFeatureToSplit(dataset)
        print("bestFeature=" + str(bestFeature))
        self.assertIsNotNone(bestFeature, "bestFeature is null")

    def test_tree4(self):
        print('============创建决策树的测试=============')
        dataset, labels = trees.createDataSet()
        mytree = trees.createTree(dataset, labels)
        print(mytree)
        self.assertIsNotNone(mytree, "mytree is null")

    def test_tree5(self):
        print('============保存决策树分类器的测试=============')
        dataset, labels = trees.createDataSet()
        mytree = trees.createTree(dataset, labels)
        trees.storeTree(mytree, 'classifierStorage2.txt')

    def test_tree6(self):
        print('============读取决策树分类器的测试=============')
        mytree = trees.grabTree("classifierStorage2.txt")
        print(mytree)


if __name__ == '__main__':
    unittest.main()
