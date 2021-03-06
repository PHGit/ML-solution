# coding: utf-8
from numpy import *
import pandas as pd
import plotDecisionTree

class Stack_Element(object):
    def __init__(self, data_set,current_depth,node):
        self.data_set = data_set
        self.current_depth = current_depth
        self.node = node


class Node(object):
    """
    决策树节点类
    """
    def __init__(self):
        self.attr = None     # 决策节点属性
        self.value = None    # 节点与父节点的连接属性
        self.isLeaf = False  # 是否为叶节点 T or F
        self.decision = 0    # 叶节点分类值 int
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

    def cal_gini(self):
        """
        计算基尼指数
        :return:float 基尼指数
        """
        class_dict = {}
        for label in self.class_set:
            if label not in class_dict.keys():
                class_dict[label] = 0
            class_dict[label] += 1
        # 计算基尼指数
        Gini = 1.0
        for key in class_dict.keys():
            prob = float(class_dict[key]) / self.num
            Gini -= prob * prob
        return Gini

    def choose_best_feature(self):
        """
        根据现有数据集划分出基尼指数最小的集合，返回划分属性的索引
        :return: 最佳划分属性索引，int
        """
        num_features = len(self.feature_set[0])
        best_gini = 1000
        best_feature_index = -1
        for index in range(num_features):
            unique_feat = set([example[index] for example in self.feature_set])
            new_gini = 0
            for feature in unique_feat:
                sub_data_set = self.filter_value(index,feature)
                prob = sub_data_set.num/float(self.num)
                new_gini += prob*sub_data_set.cal_gini()
            if new_gini < best_gini:
                best_gini = new_gini
                best_feature_index = index
        return best_feature_index


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self, data_set, feature_dict):
        self.node = self.tree_generate(data_set, feature_dict)
        self.data_set = data_set
        self.data_test = data_test






    def tree_generate(self, data_set, feature_dict):
        """
        :param self
        :param data_set: 数据集
        :return:
        """
        max_deepth = 10
        root_node = Stack_Element(data_set,1,Node())
        stack = [root_node]
        while len(stack):
            stack_element = stack[-1]

            element_data_set = stack_element.data_set
            node = stack_element.node
            if (element_data_set.pure_class() != -1) or (len(element_data_set.class_set) == 0) or (element_data_set.feature_is_same()) or (stack_element.current_depth >= max_deepth):
                # print len(queue)
                # print element_data_set.feature_set
                node.isLeaf = True  # 标记为叶子节点
                node.decision = element_data_set.class_most()  # 标记叶子节点的类别为C类
                stack.pop()
            else:
                # print element_data_set.feature_set
                best_feature_index = element_data_set.choose_best_feature()  # 最佳划分数据集的标签索引
                best_label = element_data_set.labels[best_feature_index]   # 最佳划分数据集的标签
                node.attr = best_label   # 设置非叶节点的属性为最佳划分数据集的标签
                feat_set_full = feature_dict[best_label]
                feat_set = set([example[best_feature_index] for example in element_data_set.feature_set])  # 最佳划分标签的可取值集合
                stack.pop()
                for feat in feat_set:
                    new_node = Node()
                    print element_data_set.feature_set
                    sub_data_set = element_data_set.filter_value(best_feature_index, feat)   # 划分数据集并返回子数据集
                    print sub_data_set.feature_set
                    print ''
                    if sub_data_set.num == 0:  # 如果子数据集为空
                        new_node.isLeaf = True  # 标记新节点为叶子节点
                        new_node.decision = element_data_set.class_most()  # 类别为父数据集中样本数最多的类
                    else:
                        new_stack_element = Stack_Element(sub_data_set, stack_element.current_depth + 1, new_node)
                        stack.append(new_stack_element)
                    new_node.value = feat  # 设置节点与父节点间的连接属性
                    node.children.append(new_node)

                for feat in feat_set_full-feat_set:
                    new_node = Node()
                    new_node.isLeaf = True  # 标记新节点为叶子节点
                    new_node.decision = element_data_set.class_most()  # 类别为父数据集中样本数最多的类
                    new_node.value = feat  # 设置节点与父节点间的连接属性
                    node.children.append(new_node)

        return root_node.node


def generate_full_features(data_set):
    features_list = {}
    labels = data_set.labels
    for i in range(len(data_set.feature_set[0])):
        new_feature = []
        for feature in data_set.feature_set:
            new_feature.append(feature[i])
        features_list[labels[i]] = set(new_feature)
    return features_list


df = pd.read_csv('watermelon_4_2.csv')
data = df.values[:11, :].tolist()
data_test = df.values[11:, :].tolist()
labels = df.columns.values[0:-1].tolist()
data_set = DataSet(data, labels)

feature_dict = generate_full_features(data_set)
data_test = DataSet(data_test, labels)
print data_test.feature_set
tree = DecisionTree(data_set,feature_dict)
plotDecisionTree.createPlot(tree)











