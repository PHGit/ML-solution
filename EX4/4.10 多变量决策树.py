# coding: utf-8
from numpy import *
import pandas as pd
import plotDecisionTree
from math import log

class Node(object):
    """
    决策树节点类
    """
    def __init__(self):
        self.attr = None     # 决策节点属性
        self.value = None    # 节点与父节点的连接属性
        self.isLeaf = False  # 是否为叶节点 T or F
        self.decision = 0    # 叶节点分类值 int
        self.parent = None   # 父节点
        self.children = []   # 子列表

    @staticmethod
    def set_leaf(self, decision):
        """
        :param self:
        :param decision: 叶子节点分类值 int
        :return:
        """
        self.isLeaf = True
        self.decision = decision


class DataSet(object):
    """
    数据集类
    """
    def __init__(self, data_set, labels):
        """
        :param labels 标签集合 list[]
        :param data_set 数据集 list[list[]]:
        """
        self.num = len(data_set)  # 获得数据中的向量个数
        # print data_set
        self.data_set = data_set
        self.feature_set = [example[:-1] for example in data_set if self.data_set[0]]  # 特征向量集合
        self.class_set = [int(example[-1]) for example in data_set if self.data_set[0]]  # 类别向量集合
        self.labels = labels  # 特征向量对应的标签

    def class_most(self):
        """
        返回class_set中出现最多的类别 int类型
        :return:int
        """
        class_dict = {}
        for label in self.class_set:
            if label not in class_dict.keys():
                class_dict[label] = 0
                class_dict[label] += 1
        return max(class_dict)

    @staticmethod
    def sigmoid(inX):
        y = 1.0 / (1 + exp(-inX))
        return 1.0 / (1 + exp(-inX))

    def logistic_test(self):
        test_mat = []
        for i in range(self.num):
            new_vec = self.feature_set[i]+[1]
            test_mat.append(new_vec)
        n = len(test_mat[0])
        test_mat = array(test_mat)
        alpha = 0.6
        max_cycles = 3
        weights = array(ones((n, 1)))
        for k in range(400):
            a = dot(test_mat, weights)
            h = DataSet.sigmoid(a)
            error = (array(self.class_set).reshape(self.num,1) - h)
            weights = weights + alpha * dot(test_mat.transpose(), error)

        print weights
        return weights


    def pure_class(self):
        """
        判断当前的data_set是否属于同一类别，如果是返回类别，如果不是，返回-1
        :return: class  or -1
        """
        if sum(self.class_set) == self.num*self.class_set[0]:
            return self.class_set[0]
        else:
            return -1




    def feature_is_same(self):
        """
        判断数据集的所有特征是否相同
        :return: T or F
        """
        example = self.feature_set[0]
        for index in range(self.num):
            if self.feature_set[index] != example:
                return False
        return True



    def split_data_set(self, weights):
        """
        划分连续型数据
        :param axis:
        :param value:
        :return:feature_List[axis] <= value的数据集 和 feature_list[axis] > value的数据集
        """
        left_feature_list = []
        right_feature_list = []
        for i in range(self.num):
            new_vec = self.feature_set[i] + [1]
            result = dot(new_vec, weights)
            cla = DataSet.sigmoid(result)
            # print result
            if result <= 0:
                left_feature_list.append(self.data_set[i])
            else:
                right_feature_list.append(self.data_set[i])

        return left_feature_list, right_feature_list

    def choose_best_weights(self):
        """
        根据现有数据集划分出信息增益的集合，返回划分属性的索引
        :return: 最佳划分属性索引，int
        """
        best_label = ""
        new_weights = self.logistic_test()
        for i in range(len(self.labels)):
                best_label += '+' +str(new_weights[i]) + '*'+self.labels[i]
        best_label += '<='+str(-new_weights[-1])
        return new_weights, best_label


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self, data_set):
        self.node = self.tree_generate(data_set)

    def tree_generate(self, data_set):
        """
        :param self
        :param data_set: 数据集
        :return:
        """
        node = Node()
        if len(data_set.class_set) == 0 or data_set.feature_is_same():  # 如果数据集为空或者数据集的特征向量相同
            node.isLeaf = True   # 标记为叶子节点
            node.decision = data_set.class_most()   # 标记叶子节点的类别为数据集中样本数最多的类
            return node
        if data_set.pure_class() != -1:  # 如果数据集中样本属于同一类别C
            node.isLeaf = True   # 标记为叶子节点
            node.decision = data_set.pure_class()   # 标记叶子节点的类别为C类
            return node

        best_weights, best_label = data_set.choose_best_weights()  # 最佳划分数据集的标签索引,最佳划分数据集的标签
        node.attr = best_label   # 设置非叶节点的属性为最佳划分数据集的标签
        left_data_set, right_data_set = data_set.split_data_set(best_weights)
        left_node = Node()
        right_node = Node()

        if len(left_data_set) == 0:  # 如果子数据集为空
            left_node.isLeaf = True  # 标记新节点为叶子节点
            left_node.decision = data_set.class_most()  # 类别为父数据集中样本数最多的类
        else:
            left_node = self.tree_generate(DataSet(left_data_set,data_set.labels))
        if len(right_data_set) == 0:  # 如果子数据集为空
            right_node.isLeaf = True  # 标记新节点为叶子节点)
            right_node.decision = data_set.class_most()  # 类别为父数据集中样本数最多的类
        else:
            # print right_data_set.class_set
            right_node = self.tree_generate(DataSet(right_data_set,data_set.labels))
        left_node.value = 'Yes'
        right_node.value = 'No'
        node.children.append(left_node)
        node.children.append(right_node)
        return node






df = pd.read_csv('3a.csv')
data = df.values[:10, :].tolist()
labels = df.columns.values[0:-1].tolist()
data_set = DataSet(data, labels)
tree = DecisionTree(data_set)
plotDecisionTree.createPlot(tree)











