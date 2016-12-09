# coding: utf-8
from numpy import *
import pandas as pd
import plotDecisionTree
from math import log
class Queue_Element(object):
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
    def __init__(self, data_set, labels, weights):
        """
        :param labels 标签集合 list[]
        :param data_set 数据集 list[list[]]:
        """
        self.num = len(data_set)  # 获得数据中的向量个数
        self.data_set = data_set
        self.feature_set = [example[:-1] for example in data_set if self.data_set[0]]  # 特征向量集合
        self.weights = weights
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
        sub_weights = [self.weights[i] for i in range(self.num) if self.feature_set[i][axis] == value]
        del(sub_labels[axis])       #删除划分属性
        return DataSet(filtered_data_set, sub_labels,sub_weights)

    def filter_no_missing(self, axis):
        filtered_data_set = [example for example in self.data_set if example[axis] != '-']
        sub_weights = [self.weights[i] for i in range(self.num) if self.feature_set[i][axis] != '-']
        sub_labels = self.labels[:]
        return DataSet(filtered_data_set, sub_labels, sub_weights)

    def filter_missing(self, axis):
        filtered_data_set = [example[:axis]+example[axis+1:] for example in self.data_set if example[axis] == '-']
        sub_weights = [self.weights[i] for i in range(self.num) if self.feature_set[i][axis] == '-']
        sub_labels = self.labels[:]
        del (sub_labels[axis])
        return DataSet(filtered_data_set, sub_labels, sub_weights)


    def feature_is_same(self):
        """
        判断数据集的所有特征是否相同
        :return: T or F
        """
        example = self.feature_set[0]
        for index in range(self.num):
            for y in range(len(example)):
                if self.feature_set[index][y] != example[y]  and self.feature_set[index][y] != '-' and example[y] != '-':
                    return False
        return True

    def cal_ent(self):
        """
        计算信息熵
        :return:float 信息熵
        """
        class_dict = {}
        for label in self.class_set:
            if label not in class_dict.keys():
                class_dict[label] = 0
            class_dict[label] += 1
        # 计算信息熵
        ent = 0
        for key in class_dict.keys():
            prob = float(class_dict[key]) / self.num
            ent -= prob*log(prob, 2)
        return ent

    def choose_best_feature(self):
        """
        根据现有数据集划分出信息增益的集合，返回划分属性的索引
        :return: 最佳划分属性索引，int
        """
        num_features = len(self.feature_set[0])
        biggest_gain = 0
        best_feature_index = -1
        for index in range(num_features):
            new_ent = 0
            no_missing_data_set = self.filter_no_missing(index)
            no_missing_ent = no_missing_data_set.cal_ent()
            p_no_missing = no_missing_data_set.num/float(self.num)
            feature_list = [example[index] for example in no_missing_data_set.feature_set]
            unique_feat = set(feature_list)
            for feature in unique_feat:
                sub_data_set = no_missing_data_set.filter_value(index, feature)
                prob = sub_data_set.num / float(no_missing_data_set.num)
                new_ent += prob * sub_data_set.cal_ent()
            gain = p_no_missing*(no_missing_ent - new_ent)
            if gain > biggest_gain:
                biggest_gain = gain
                best_feature_index = index
        print self.data_set,best_feature_index
        return best_feature_index


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self,data_set, feature_dict):
        self.node = self.tree_generate(data_set, feature_dict)
        self.data_set = data_set


    def tree_generate(self, data_set, feature_dict):
        """
        :param self
        :param data_set: 数据集
        :return:
        """
        max_deepth = 10
        root_node = Queue_Element(data_set,1,Node())
        queue = [root_node]
        while len(queue):
            queue_element = queue[-1]
            element_data_set = queue_element.data_set
            node = queue_element.node
            if (element_data_set.pure_class() != -1) or (len(element_data_set.class_set) == 0) or (element_data_set.feature_is_same()) or (queue_element.current_depth >= max_deepth):
                node.isLeaf = True  # 标记为叶子节点
                node.decision = element_data_set.class_most()  # 标记叶子节点的类别为C类
            else:
                best_feature_index = element_data_set.choose_best_feature()  # 最佳划分数据集的标签索引
                best_label = element_data_set.labels[best_feature_index]   # 最佳划分数据集的标签
                node.attr = best_label   # 设置非叶节点的属性为最佳划分数据集的标签
                print node.attr
                feat_set_full = feature_dict[best_label]
                feat_set = set([example[best_feature_index] for example in element_data_set.feature_set if example[best_feature_index] != '-' ])  # 最佳划分标签的可取值集
                missing_data_set = element_data_set.filter_missing(best_feature_index)
                no_missing_data_set = element_data_set.filter_no_missing(best_feature_index)
                for feat in feat_set:
                    print feat
                    new_node = Node()
                    sub_data_set = element_data_set.filter_value(best_feature_index, feat)   # 划分数据集并返回子数据集
                    p_missing = sub_data_set.num/float(element_data_set.num)
                    missing_weights = [p_missing*example for example in missing_data_set.weights]
                    new_data_set = DataSet(sub_data_set.data_set+missing_data_set.data_set,sub_data_set.labels, sub_data_set.weights+missing_weights)
                    new_node.value = feat  # 设置节点与父节点间的连接属性
                    new_queue_element = Queue_Element(new_data_set, queue_element.current_depth + 1, new_node)
                    queue.insert(0, new_queue_element)
                    node.children.append(new_node)


                print '------------'

                for feat in feat_set_full-feat_set:
                    new_node = Node()
                    new_node.isLeaf = True  # 标记新节点为叶子节点
                    new_node.decision = element_data_set.class_most()  # 类别为父数据集中样本数最多的类
                    new_node.value = feat  # 设置节点与父节点间的连接属性
                    node.children.append(new_node)
            queue.pop()
        return root_node.node


def generate_full_features(data_set):
    features_list = {}
    labels = data_set.labels
    for i in range(len(data_set.feature_set[0])):
        new_feature = []
        for feature in data_set.feature_set:
            if feature[i] != '-':
                new_feature.append(feature[i])
                features_list[labels[i]] = set(new_feature)
    return features_list


df = pd.read_csv('watermelon_4_2a.csv')
data = df.values[:, :].tolist()
# data_test = df.values[11:, :].tolist()
labels = df.columns.values[0:-1].tolist()
data_set = DataSet(data, labels,[1]*len(data))
feature_dict = generate_full_features(data_set)
tree = DecisionTree(data_set,feature_dict)
plotDecisionTree.createPlot(tree)
# print tree.node.children[0].children[0].children[1].children











