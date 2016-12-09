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
        return 1.0 / (1 + exp(-inX))

    def logistic_error_test(self, label_index,features_full):
        test_mat = []
        label = self.labels[label_index]
        feature_list = features_full[label]
        for i in range(self.num):
            vec = self.feature_set[i][label_index]
            if type(vec).__name__ != "float":
                vec = feature_list.index(vec)
            new_vec = [vec, 1]
            test_mat.append(new_vec)
        test_mat = array(test_mat)
        n = 2
        alpha = 0.1
        max_cycles = 10
        weights = array(ones((n, 1)))
        re_error = 0
        for k in range(max_cycles):
            a = dot(test_mat, weights)
            h = DataSet.sigmoid(a)
            error = (array(self.class_set).reshape(self.num,1) - h)
            weights = weights + alpha * dot(test_mat.transpose(), error)

        for i in range(self.num):
            # print weights
            result = dot(test_mat[i], weights)
            cla = DataSet.sigmoid(result)
            cla = 1 if cla > 0.5 else 0
            if cla != data_set.class_set[i]:
                re_error += 1
        print re_error
        return re_error


    def pure_class(self):
        """
        判断当前的data_set是否属于同一类别，如果是返回类别，如果不是，返回-1
        :return: class  or -1
        """
        if sum(self.class_set) == self.num*self.class_set[0]:
            return self.class_set[0]
        else:
            return -1

    def filter_value(self, axis, value):
        """
        过滤 feature_set[axis] 等于 value的数据集
        :param axis:
        :param value:
        :return:DataSet()
        """
        filtered_data_set = [example[:axis]+example[axis+1:] for example in self.data_set if example[axis] == value]
        sub_labels = self.labels[:]  # list 传参可能改变原参数值，故需要复制出新的labels避免影响原参数
        del(sub_labels[axis])       #删除划分属性
        return DataSet(filtered_data_set, sub_labels)


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

    # def cal_ent(self):
    #     """
    #     计算信息熵
    #     :return:float 信息熵
    #     """
    #     class_dict = {}
    #     for label in self.class_set:
    #         if label not in class_dict.keys():
    #             class_dict[label] = 0
    #         class_dict[label] += 1
    #     # 计算信息熵
    #     ent = 0
    #     for key in class_dict.keys():
    #         prob = float(class_dict[key]) / self.num
    #         ent -= prob*log(prob, 2)
    #     return ent

    def split_continuous_feature(self, axis, value):
        """
        划分连续型数据
        :param axis:
        :param value:
        :return:feature_List[axis] <= value的数据集 和 feature_list[axis] > value的数据集
        """
        left_feature_list = []
        right_feature_list = []
        for example in self.data_set:
            if example[axis] <= value:
                left_feature_list.append(example)  # 此处不用del第axis个标签，连续变量的属性可以作为后代节点的划分属性
            else:
                right_feature_list.append(example)
        return DataSet(left_feature_list, self.labels), DataSet(right_feature_list, self.labels)

    def choose_best_feature(self, features_full):
        """
        根据现有数据集划分出信息增益的集合，返回划分属性的索引
        :return: 最佳划分属性索引，int
        """
        num_features = len(self.feature_set[0])
        smallest_error = 1000
        best_feature_index = -1
        best_label = ""
        for index in range(num_features):
            new_error = self.logistic_error_test(index,features_full)
            if new_error < smallest_error:
                smallest_error = new_error
                best_feature_index = index
                best_label = self.labels[best_feature_index]
        return best_feature_index, best_label


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self, data_set,feature_dict):
        self.node = self.tree_generate(data_set,feature_dict)

    def tree_generate(self, data_set,feature_dict):
        """
        :param self
        :param data_set: 数据集
        :return:
        """
        node = Node()
        if data_set.pure_class() != -1:  # 如果数据集中样本属于同一类别C
            node.isLeaf = True   # 标记为叶子节点
            node.decision = data_set.pure_class()   # 标记叶子节点的类别为C类
            return node
        if len(data_set.class_set) == 0 or data_set.feature_is_same():  # 如果数据集为空或者数据集的特征向量相同
            node.isLeaf = True   # 标记为叶子节点
            node.decision = data_set.class_most()   # 标记叶子节点的类别为数据集中样本数最多的类
            return node
        best_feature_index, best_label = data_set.choose_best_feature(feature_dict)  # 最佳划分数据集的标签索引,最佳划分数据集的标签
        # best_label = data_set.labels[best_feature_index]
        node.attr = best_label   # 设置非叶节点的属性为最佳划分数据集的标签
        if u'<=' in node.attr:  #
            print 'test'
            mid_value = float(node.attr.split('<=')[1])  # 获得比较值
            left_data_set, right_data_set = data_set.split_continuous_feature(best_feature_index, mid_value)
            left_node = Node()
            print left_data_set.feature_set
            print right_data_set.feature_set
            right_node = Node()
            if left_data_set.num == 0:  # 如果子数据集为空
                left_node.isLeaf = True  # 标记新节点为叶子节点
                left_node.decision = data_set.class_most()  # 类别为父数据集中样本数最多的类
            else:
                left_node = self.tree_generate(left_data_set,feature_dict)
            if right_data_set.num == 0:  # 如果子数据集为空
                right_node.isLeaf = True  # 标记新节点为叶子节点
                right_node.decision = data_set.class_most()  # 类别为父数据集中样本数最多的类

            else:
                print right_data_set.class_set
                right_node = self.tree_generate(right_data_set,feature_dict)
            left_node.value = 'Yes'
            right_node.value = 'No'
            node.children.append(left_node)
            node.children.append(right_node)
        else:
            feat_set = set([example[best_feature_index] for example in data_set.feature_set])  # 最佳划分标签的可取值集合
            for feat in feat_set:
                new_node = Node()
                sub_data_set = data_set.filter_value(best_feature_index, feat)   # 划分数据集并返回子数据集
                if sub_data_set.num == 0:   # 如果子数据集为空
                    new_node.isLeaf = True  # 标记新节点为叶子节点
                    new_node.decision = data_set.class_most()   # 类别为父数据集中样本数最多的类
                else:
                    new_node = self.tree_generate(sub_data_set,feature_dict)
                new_node.value = feat  # 设置节点与父节点间的连接属性
                new_node.parent = node  # 设置父节点
                node.children.append(new_node)
        return node


def generate_full_features(data_set):
    features_list = {}
    labels = data_set.labels
    for i in range(len(data_set.feature_set[0])):
        new_feature = []
        for feature in data_set.feature_set:
            new_feature.append(feature[i])
        features_list[labels[i]] = list(set(new_feature))
    return features_list



df = pd.read_csv('watermelon_4_2.csv')
data = df.values[:10, :].tolist()
labels = df.columns.values[0:-1].tolist()
data_set = DataSet(data, labels)
feature_dict = generate_full_features(data_set)
tree = DecisionTree(data_set,feature_dict)
plotDecisionTree.createPlot(tree)











