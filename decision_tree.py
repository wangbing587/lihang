
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger


class TreeNode():
    """树结点"""
    def __init__(self, feature_idx=None, feature_val=None, node_val=None, child=None):
        """
        feature_idx:
            该结点对应的划分特征索引
        feature_val:
            划分特征对应的值
        node_val:
            该结点存储的值，**只有叶结点才存储类别**
        child:
            子树
        """
        self._feature_idx = feature_idx
        self._feature_val = feature_val
        # 叶结点存储类别
        self._node_val = node_val
        # 非叶结点存储划分信息
        self._child = child


class DecisionTreeScratch():
    """决策树算法Scratch实现"""
    def __init__(self, etype="gain", epsilon=0.01):
        """
        etype:
            可取值有
            gain: 使用信息增益
            ratio: 使用信息增益比
        epsilon:
            当信息增益或信息增益比小于该阈值时直接把对应结点作为叶结点
        """
        self._root = None
        self._etype = etype
        self._epsilon = epsilon

    def fit(self, X, y):
        """
        模型训练
        X:
            训练集，每一行表示一个样本，每一列表示一个特征或属性
        y:
            训练集标签
        """
        self._root = self._build_tree(X, y)

    def predict(self, x):
        """给定输入样本，预测其类别"""
        return self._predict(x, self._root)

    def _build_tree(self, X, y):
        """
        构建树
        X:
            用于构建子树的数据集
        y:
            X对应的标签
        """
        # 最后只剩下一个特征时将该子树置为叶结点
        if X.shape[1] == 1:
            node_val = self._vote_label(y)
            return TreeNode(node_val=node_val)

        # 子树只剩下一个类别时直接置为叶结点
        if np.unique(y).shape[0] == 1:
            return TreeNode(node_val=y[0])

        n_feature = X.shape[1]
        # 最大信息增益或信息增益比
        max_gain = -np.inf
        # 信息增益最大或信息增益比最大的特征索引
        max_fea_idx = 0
        # 遍历每一个特征
        for i in range(n_feature):
            if self._etype == "gain":
                gain = self._calc_gain(X[:, i], y)
            else:
                gain = self._calc_gain_ration(X[:, i], y)
            if gain > max_gain:
                max_gain = gain
                max_fea_idx = i

        # 如果以该特征进行划分信息增益或增益比太小则不进行划分
        if max_gain < self._epsilon:
            node_val = self._vote_label(y)
            return TreeNode(node_val=node_val)

        child_tree = dict()
        # 遍历所选特征每一个可能的值，对每一个值构建子树
        feature_val = np.unique(X[:, max_fea_idx])
        for fea_val in feature_val:
            # 如果要对连续型数据进行划分，可将此处的==换成<=，类似CART
            child_X = X[X[:, max_fea_idx] == fea_val]
            child_y = y[X[:, max_fea_idx] == fea_val]
            child_X = np.delete(child_X, max_fea_idx, 1)
            # 构建子树
            child_tree[fea_val] = self._build_tree(child_X, child_y)
        return TreeNode(max_fea_idx, child=child_tree)

    def _predict(self, x, tree=None):
        """
        给定输入样本，将其划分到所属叶结点
        """
        # 根结点
        if tree is None:
            tree = self._root

        # 叶结点直接返回类别
        if tree._node_val is not None:
            return tree._node_val

        fea_idx = tree._feature_idx
        for fea_val, child_node in tree._child.items():
            if x[fea_idx] == fea_val:
                # 如果是叶结点直接返回类别
                if child_node._node_val is not None:
                    return child_node._node_val
                else:
                    # 否则继续去子树中找
                    return self._predict(x, child_node)

    def _vote_label(self, y):
        """统计y中次数出现最多的类别"""
        # 计算每个标签及其出现次数
        label, num_label = np.unique(y, return_counts=True)
        # 返回出现次数最多的类别作为叶结点类别
        return label[np.argmax(num_label)]

    def _calc_entropy(self, y):
        """
        计算熵
        y:
            数据集标签
        """
        entropy = 0
        # 计算每个类别的数量
        _, num_ck = np.unique(y, return_counts=True)
        for n in num_ck:
            p = n / y.shape[0]
            entropy -= p * np.log2(p)
        return entropy

    def _calc_condition_entropy(self, x, y):
        """
        计算条件熵
        x:
            数据集的某个特征对应的列向量，训练集中一行表示一个样本，列表示特征
        y:
            数据集标签
        """
        cond_entropy = 0
        # 计算特征x可能的取值及对应出现的次数
        xval, num_x = np.unique(x, return_counts=True)
        # 遍历该特征的每个值
        for v, n in zip(xval, num_x):
            # 该值所对应的划分
            y_sub = y[x == v]
            sub_entropy = self._calc_entropy(y_sub)
            p = n / y.shape[0]
            cond_entropy += p * sub_entropy
        return cond_entropy

    def _calc_gain(self, x, y):
        """
        计算信息增益
        x:
            数据集的某个特征对应的列向量，训练集中一行表示一个样本，列表示特征
        y:
            数据集标签
        """
        return self._calc_entropy(y) - self._calc_condition_entropy(x, y)

    def _calc_gain_ration(self, x, y):
        """
        计算信息增益比
        x:
            数据集的某个特征对应的列向量，训练集中一行表示一个样本，列表示特征
        y:
            数据集标签
        """
        return self._calc_gain(x, y) / self._calc_entropy(x)


def main():
    parser = argparse.ArgumentParser(description="决策树算法Scratch代码命令行参数")
    parser.add_argument("--epsilon", type=float, default=0.1, help="当信息增益或信息增益比小于该阈值时直接把对应结点作为叶结点")
    parser.add_argument("--etype", type=str, default="gain", help="使用信息增益还是信息增益比进行特征选择，可选值有gain, ratio")
    args = parser.parse_args()

    dataset = load_iris()
    xtrain, _, ytrain, _ = train_test_split(dataset.data, dataset.target, train_size=0.8, shuffle=True)

    model = DecisionTreeScratch(args.etype, args.epsilon)
    model.fit(xtrain, ytrain)

    # !!! 手上没有合适的数据集，暂时拿训练集来测试 !!!
    n_train = xtrain.shape[0]
    n_right = 0
    for i in range(n_train):
        y_pred = model.predict(xtrain[i])
        if y_pred == ytrain[i]:
            n_right += 1
        else:
            logger.info("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(ytrain[i], y_pred))
    logger.info("Scratch模型在训练集上的准确率为：{}%".format(n_right * 100 / n_train))

if __name__ == "__main__":
    main()

