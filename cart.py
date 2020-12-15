
import argparse

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from loguru import logger


class TreeNode():
    """树结点"""
    def __init__(self, feature_idx=None, feature_val=None, node_val=None,
                left_child=None, right_child=None):
        """
        feature_idx:
            该结点对应的划分特征索引
        feature_val:
            划分特征对应的值
        node_val:
            该结点存储的值，只有叶结点才存储类别信息，回归树存储结点的平均值，分类树存储类别出现次数最多的类别
        left_child:
            左子树
        right_child:
            右子树
        """
        self._feature_idx = feature_idx
        self._feature_val = feature_val
        self._node_val = node_val
        self._left_child = left_child
        self._right_child = right_child


class CARTScratch(object):
    """CART算法Scratch实现"""
    def __init__(self, min_sample=2, min_gain=1e-6, max_depth=np.inf):
        """
        min_sample:
            当数据集样本数少于min_sample时不再划分
        min_gain:
            如果划分后收益不能超过该值则不进行划分
            对分类树来说基尼指数需要有足够的下降
            对回归树来说平方误差要有足够的下降
        max_depth:
            树的最大高度
        """
        self._root = None
        self._min_sample = min_sample
        self._min_gain = min_gain
        self._max_depth = max_depth

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
        """给定输入样本，预测其类别或者输出"""
        return self._predict(x, self._root)

    def _build_tree(self, X, y, cur_depth=0):
        """
        构建树
        X:
            用于构建子树的数据集
        y:
            X对应的标签
        cur_depth:
            当前树的高度
        """
        # 如果子树只剩下一个类别时则置为叶结点
        if np.unique(y).shape[0] == 1:
            node_val = self._calc_node_val(y)
            return TreeNode(node_val=node_val)

        # 划分前基尼指数或平方差
        before_divide = self._calc_evaluation(y)
        # 基尼指数或平方差最小的划分
        min_evaluation = np.inf
        # 最佳划分特征索引
        best_feature_idx = None
        # 最佳划分特征的值
        best_feature_val = None

        n_sample, n_feature = X.shape
        # 样本数至少大于给定阈值，且当前树的高度不能超过阈值才继续划分
        if n_sample >= self._min_sample and cur_depth <= self._max_depth:
            # 遍历每一个特征
            for i in range(n_feature):
                feature_value = np.unique(X[:, i])
                # 遍历该特征的每一个值
                for fea_val in feature_value:
                    # 左边划分
                    X_left = X[X[:,i] <= fea_val]
                    y_left = y[X[:,i] <= fea_val]

                    # 右边划分
                    X_right = X[X[:,i] > fea_val]
                    y_right = y[X[:,i] > fea_val]

                    if X_left.shape[0] > 0 and y_left.shape[0] > 0 and X_right.shape[0] > 0 and y_right.shape[0] > 0:
                        # 划分后的基尼指数或平方差
                        after_divide = self._calc_division(y_left, y_right)
                        if after_divide < min_evaluation:
                            min_evaluation = after_divide
                            best_feature_idx = i
                            best_feature_val = fea_val

        # 如果划分前后基尼指数或平方差有足够的下降才继续划分
        if before_divide - min_evaluation > self._min_gain:
            # 左边划分
            X_left = X[X[:,best_feature_idx] <= best_feature_val]
            y_left = y[X[:,best_feature_idx] <= best_feature_val]

            # 右边划分
            X_right = X[X[:,best_feature_idx] > best_feature_val]
            y_right = y[X[:,best_feature_idx] > best_feature_val]

            # 构建左子树
            left_child = self._build_tree(X_left, y_left, cur_depth+1)
            # 构建右子树
            right_child = self._build_tree(X_right, y_right, cur_depth+1)

            return TreeNode(feature_idx=best_feature_idx, feature_val=best_feature_val,
                            left_child=left_child, right_child=right_child)

        # 样本数少于给定阈值，或者树的高度超过阈值，或者未找到合适的划分则置为叶结点
        node_val = self._calc_node_val(y)
        return TreeNode(node_val=node_val)

    def _predict(self, x, tree=None):
        """给定输入预测输出"""
        # 根结点
        if tree is None:
            tree = self._root

        # 叶结点直接返回预测值
        if tree._node_val is not None:
            return tree._node_val

        # 用该结点对应的划分索引获取输入样本在对应特征上的取值
        feature_val = x[tree._feature_idx]
        if feature_val <= tree._feature_val:
            return self._predict(x, tree._left_child)
        return self._predict(x, tree._right_child)

    def _calc_division(self, y_left, y_right):
        """计算划分后的基尼指数或平方差"""
        return NotImplementedError()

    def _calc_evaluation(self, y):
        """计算数据集基尼指数或平方差"""
        return NotImplementedError()

    def _calc_node_val(self, y):
        """计算叶结点的值，分类树和回归树分别实现"""
        return NotImplementedError()


class CARTClassificationScratch(CARTScratch):
    def _calc_division(self, y_left, y_right):
        """计算划分后的基尼指数"""
        # 计算左右子树基尼指数
        left_evaluation = self._calc_evaluation(y_left)
        right_evaluation = self._calc_evaluation(y_right)
        p_left = y_left.shape[0] / (y_left.shape[0] + y_right.shape[0])
        p_right = y_right.shape[0] / (y_left.shape[0] + y_right.shape[0])
        # 划分后的基尼指数
        after_divide = p_left * left_evaluation + p_right * right_evaluation
        return after_divide

    def _calc_evaluation(self, y):
        """计算标签为y的数据集的基尼指数"""
        # 计算每个类别样本的个数
        _, num_ck = np.unique(y, return_counts=True)
        gini = 1
        for n in num_ck:
            gini -= (n / y.shape[0]) ** 2
        return gini

    def _calc_node_val(self, y):
        """分类树从标签中进行投票获取出现次数最多的值作为叶结点的类别"""
        # 计算每个标签及其出现次数
        label, num_label = np.unique(y, return_counts=True)
        # 返回出现次数最多的类别作为叶结点类别
        return label[np.argmax(num_label)]


class CARTRegressionScratch(CARTScratch):
    def _calc_division(self, y_left, y_right):
        """计算划分后的平方差"""
         # 计算左右子树平方差
        left_evaluation = self._calc_evaluation(y_left)
        right_evaluation = self._calc_evaluation(y_right)
        # 划分后的平方差
        after_divide = left_evaluation + right_evaluation
        return after_divide

    def _calc_evaluation(self, y):
        """计算平方差"""
        return np.sum(np.power(y - np.mean(y), 2))

    def _calc_node_val(self, y):
        """回归树返回标签的平均值作为叶结点的预测值"""
        return np.mean(y)


def main():
    parser = argparse.ArgumentParser(description="CART算法Scratch代码命令行参数")
    parser.add_argument("--min_sample", type=int, default=2, help="子树中样本数少于该值时不再进行划分")
    parser.add_argument("--max_depth", type=int, default=20, help="树的最大高度")
    parser.add_argument("--min_gain", type=float, default=1e-6, help="划分后收益不超过该值则不进行划分")
    args = parser.parse_args()

    iris_data, iris_y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(iris_data, iris_y, train_size=0.8, shuffle=True)

    model = CARTClassificationScratch(args.min_sample, args.min_gain, args.max_depth)
    model.fit(xtrain, ytrain)

    n_test = xtest.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtest[i])
        if y_pred == ytest[i]:
            n_right += 1
        else:
            logger.info("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(ytest[i], y_pred))
    logger.info("Scratch分类模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))

    skmodel = DecisionTreeClassifier()
    skmodel.fit(xtrain, ytrain)
    logger.info("sklearn分类模型在测试集上准确率为：{}%".format(100 * skmodel.score(xtest, ytest)))

    logger.info("\n------------------- 以下为CART回归算法结果 -------------------\n")

    boston_data, boston_y = load_boston(return_X_y=True)
    boston_xtrain, boston_xtest, boston_ytrain, boston_ytest = train_test_split(boston_data, boston_y, train_size=0.8, shuffle=True)

    regressor_model = CARTRegressionScratch(args.min_sample, args.min_gain, args.max_depth)
    regressor_model.fit(boston_xtrain, boston_ytrain)

    n_boston_test = boston_xtest.shape[0]
    total_diff = 0
    for i in range(n_boston_test):
        y_pred = regressor_model.predict(boston_xtest[i])
        total_diff += abs(y_pred - boston_ytest[i])
        logger.info("Scratch模型预测房价为：{:.1f}，\t 真实房价为：{}".format(y_pred, boston_ytest[i]))

    boston_skmodel = DecisionTreeRegressor()
    boston_skmodel.fit(boston_xtrain, boston_ytrain)

    y_skpred = boston_skmodel.predict(boston_xtest)
    sk_total_diff = 0
    for i in range(n_boston_test):
        sk_total_diff += abs(y_skpred[i] - boston_ytest[i])
        logger.info("skmodel模型预测房价为：{:.1f}，\t 真实房价为：{}".format(y_skpred[i], boston_ytest[i]))

    logger.info("Scratch回归模型平均预测误差为：{:.1f}".format(total_diff / n_boston_test))
    logger.info("skmodel回归模型平均预测误差为：{:.1f}".format(sk_total_diff / n_boston_test))

if __name__ == "__main__":
    main()

