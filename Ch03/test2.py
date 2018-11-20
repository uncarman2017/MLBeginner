import unittest

from Ch03 import treePlotter
from Ch03 import trees


class MyTestCase(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)

    def test_treePlotter1(self):
        print('============test1=============')
        mytree = treePlotter.retrieveTree(0)
        print(mytree)
        num = treePlotter.getNumLeafs(mytree)
        print("num=" + str(num))
        depth = treePlotter.getTreeDepth(mytree)
        print("depth=" + str(depth))
        self.assertTrue(num > 0, "num should be more than zero")
        self.assertTrue(depth > 0, "depth should be more than zero")

    def test_treePlotter2(self):
        print('============test2=============')
        mytree = treePlotter.retrieveTree(0)
        treePlotter.createPlot(mytree)

    def test_treePlotter3(self):
        print('============test3=============')
        mytree = treePlotter.retrieveTree(0)
        mytree['no surfacing'][3] = 'maybe'
        treePlotter.createPlot(mytree)

    def test_treePlotter4(self):
        print('============test4=============')
        fr = open("E:\github\ML\MLBeginner\Ch03\lenses.txt")
        lenses = [inst.strip().split('\t') for inst in fr.readline()]
        print(lenses)
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        print(lensesLabels)
        lenseTree = trees.createTree(lenses, lensesLabels)
        print(lenseTree)
        fr.close()

    def test_treePlotter5(self):
        print("============使用决策树的分类函数==============")
        myDat, labels = trees.createDataSet()
        print(labels)

        mytree = treePlotter.retrieveTree(0)
        print(mytree)
        result = trees.classify(mytree, labels, [1, 0])
        print(result)
        result = trees.classify(mytree, labels, [1, 1])
        print(result)


if __name__ == '__main__':
    unittest.main()
