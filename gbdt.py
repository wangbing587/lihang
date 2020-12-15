
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from loguru import logger
from cart import CARTRegressionScratch
from util import SquareLoss, CrossEntropyLoss


class GBDTScratch(object):
    """梯度提升树Scratch实现"""
    def __init__(self, n_estimator=10, learning_rate=0.01):
        """
        n_estimator:
            残差树个数
        learning_rate:
            学习率
        """
        self._n_estimator = n_estimator
        self._lr = learning_rate
        # 存储残差树
        self._trees = []

    def fit(self, X, y):
        """模型训练"""
        pass

    def predict(self, x):
        """给定输入样本，预测输出"""
        pass


class GBDTClassificationScratch(GBDTScratch):
    def __init__(self, n_estimator=10, learning_rate=0.01, min_sample=2, min_gain=0.1, max_depth=10):
        super(GBDTClassificationScratch, self).__init__(n_estimator, learning_rate)
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
        self._min_sample = min_sample
        self._min_gain = min_gain
        self._max_depth = max_depth
        # 分类树损失函数维交叉熵损失
        self._loss = CrossEntropyLoss()

    def fit(self, X, y):
        """模型训练"""
        # 先对输入标签做one hot编码
        y = self._to_one_hot(y)
        n_sample, self._n_class = y.shape
        # 初始残差为每个类别的平均值
        residual_pred = np.full_like(y, np.mean(y, axis=0))
        for _ in range(self._n_estimator):
            label_trees = []
            residual_update = np.zeros_like(residual_pred)
            # 每个类别分别学习树
            for j in range(self._n_class):
                residual_gradient = self._loss.calc_gradient(y[:, j], residual_pred[:, j])
                tree = CARTRegressionScratch(self._min_sample, self._min_gain, self._max_depth)
                # 每棵树以残差为目标进行训练
                tree.fit(X, residual_gradient)
                label_trees.append(tree)
                for i in range(n_sample):
                    residual_update[i, j] = tree.predict(X[i])
            self._trees.append(label_trees)
            residual_pred -= self._lr * residual_update

    def predict(self, x):
        """给定输入样本，预测输出"""
        y_pred = np.zeros(self._n_class)
        for label_trees in self._trees:
            for i in range(len(label_trees)):
                residual_update = label_trees[i].predict(x)
                y_pred[i] -= self._lr * residual_update
        # 返回概率值最大的类别，省略了指数计算
        return np.argmax(y_pred)

    def _to_one_hot(self, y):
        """将离散标签进行one hot编码"""
        n_col = np.amax(y) + 1
        one_hot = np.zeros((y.shape[0], n_col))
        # 将类别所在列置为1
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot


class GBDTRegressionScratch(GBDTScratch):
    def __init__(self, n_estimator=10, learning_rate=0.01, min_sample=2, min_gain=0.1, max_depth=10):
        super(GBDTRegressionScratch, self).__init__(n_estimator, learning_rate)
        # 回归树损失函数维平方损失
        self._loss = SquareLoss()
        for _ in range(self._n_estimator):
            tree = CARTRegressionScratch(min_sample, min_gain, max_depth)
            self._trees.append(tree)

    def fit(self, X, y):
        """模型训练"""
        n_sample = y.shape[0]
        residual_pred = np.zeros(n_sample)
        for i in range(self._n_estimator):
            residual_gradient = self._loss.calc_gradient(y, residual_pred)
            # 每棵树以残差为目标进行训练
            self._trees[i].fit(X, residual_gradient)
            residual_update = np.zeros(n_sample)
            for j in range(n_sample):
                residual_update[j] = self._trees[i].predict(X[j])
            residual_pred -= self._lr * residual_update

    def predict(self, x):
        """给定输入样本，预测输出"""
        y_pred = 0
        for tree in self._trees:
            residual_update = tree.predict(x)
            y_pred -= self._lr * residual_update
        return y_pred


def main():
    parser = argparse.ArgumentParser(description="CART算法Scratch代码命令行参数")
    parser.add_argument("--n_estimator", type=int, default=10, help="残差树个数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--min_sample", type=int, default=2, help="子树中样本数少于该值时不再进行划分")
    parser.add_argument("--max_depth", type=int, default=20, help="树的最大高度")
    parser.add_argument("--min_gain", type=float, default=1e-6, help="划分后收益不超过该值则不进行划分")
    args = parser.parse_args()

    iris_data, iris_y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(iris_data, iris_y, train_size=0.8, shuffle=True)

    model = GBDTClassificationScratch(args.n_estimator, args.lr, args.min_sample, args.min_gain, args.max_depth)
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

    boston_data, boston_y = load_boston(return_X_y=True)
    boston_xtrain, boston_xtest, boston_ytrain, boston_ytest = train_test_split(boston_data, boston_y, train_size=0.8, shuffle=True)

    regressor_model = GBDTRegressionScratch(100, 0.1, args.min_sample, args.min_gain, args.max_depth)
    # regressor_model = GBDTRegressionScratch(args.n_estimator, args.lr, args.min_sample, args.min_gain, args.max_depth)
    regressor_model.fit(boston_xtrain, boston_ytrain)

    n_boston_test = boston_xtest.shape[0]
    total_diff = 0
    for i in range(n_boston_test):
        y_pred = regressor_model.predict(boston_xtest[i])
        total_diff += abs(y_pred - boston_ytest[i])
        logger.info("Scratch模型预测房价为：{:.1f}，\t 真实房价为：{}".format(y_pred, boston_ytest[i]))

    logger.info("Scratch回归模型平均预测误差为：{:.1f}".format(total_diff / n_boston_test))

if __name__ == "__main__":
    main()

