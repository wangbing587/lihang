
import argparse
from collections import defaultdict

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from loguru import logger


class KNNClassifierScratch():
    """kNN分类算法Scratch实现"""
    def __init__(self, k):
        """
        k:
            在进行分类预测时通过最近的多少个近邻进行投票分类
        """
        self._k = k

    def fit(self, X, y):
        """
        原始kNN算法没有训练过程，这里加fit接口单纯只是想加，没有别的意义
        如果不想要该接口，可以把训练集和标签初始化放到构造函数或者predict函数中
        """
        self._xtrain = X
        self._y = y

    def predict(self, x):
        idx = np.argsort([np.linalg.norm(x - x_sample) for x_sample in self._xtrain])[:self._k]
        knn_y = self._y[idx]
        vote = defaultdict(int)
        for y_pred in knn_y:
            vote[y_pred] += 1
        vote_list = list(vote.items())
        vote_list.sort(key=lambda x: x[1], reverse=True)
        # 还可以顺便计算分到该类别的概率
        # p = vote_list[0][1] / self._k
        return vote_list[0][0]


class KNNRegressorScratch():
    """kNN回归算法Scratch实现"""
    def __init__(self, k=5, enable_weight=False):
        """
        k:
            在进行预测时通过最近的多少个近邻样本值的平均作为预测值
        enable_weight:
            在计算k个近邻的平均值时是否使用加权平均
            如果为true则以距离的倒数作为权重
            如果为false则直接求平均值
        """
        self._k = k
        self._enable_weight = enable_weight

    def fit(self, X, y):
        """
        原始kNN算法没有训练过程，这里加fit接口单纯只是想加，没有别的意义
        如果不想要该接口，可以把训练集和标签初始化放到构造函数或者predict函数中
        """
        self._xtrain = X
        self._y = y

    def predict(self, x):
        dist = np.array([np.linalg.norm(x - x_sample) for x_sample in self._xtrain])
        idx = np.argsort(dist)[:self._k]
        knn_y = self._y[idx]
        if not self._enable_weight:
            return np.mean(knn_y)

        # 以距离倒数作为权重
        dist = np.reciprocal(dist[idx])
        total_dist = np.sum(dist)
        y_pred = 0
        for i, v in enumerate(knn_y):
            y_pred += v * dist[i] / total_dist
        return y_pred


def main():
    parser = argparse.ArgumentParser(description="kNN算法Scratch代码命令行参数")
    parser.add_argument("--k", type=int, default=5, help="从最近的多少个近邻中投票计算类别")
    parser.add_argument("--enable_weight", action="store_true", help="在计算k个近邻的平均值时是否使用加权平均")
    args = parser.parse_args()

    iris_data, iris_y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(iris_data, iris_y, train_size=0.8, shuffle=True)

    model = KNNClassifierScratch(args.k)
    model.fit(xtrain, ytrain)

    n_test = xtest.shape[0]
    n_right = 0
    for i in range(n_test):
        y_pred = model.predict(xtest[i])
        if y_pred == ytest[i]:
            n_right += 1
        else:
            logger.info("该样本真实标签为：{}，但是Scratch模型预测标签为：{}".format(ytest[i], y_pred))
    logger.info("Scratch模型在测试集上的准确率为：{}%".format(n_right * 100 / n_test))

    skmodel = KNeighborsClassifier(n_neighbors=args.k)
    skmodel.fit(xtrain, ytrain)
    logger.info("sklearn模型在测试集上准确率为：{}%".format(100 * skmodel.score(xtest, ytest)))

    logger.info("\n------------------- 以下为kNN回归算法结果 -------------------\n")

    boston_data, boston_y = load_boston(return_X_y=True)
    boston_xtrain, boston_xtest, boston_ytrain, boston_ytest = train_test_split(boston_data, boston_y, train_size=0.8, shuffle=True)

    regressor_model = KNNRegressorScratch(args.k, args.enable_weight)
    regressor_model.fit(boston_xtrain, boston_ytrain)

    n_boston_test = boston_xtest.shape[0]
    total_diff = 0
    for i in range(n_boston_test):
        y_pred = regressor_model.predict(boston_xtest[i])
        total_diff += abs(y_pred - boston_ytest[i])
        logger.info("Scratch模型预测房价为：{:.1f}，\t 真实房价为：{}".format(y_pred, boston_ytest[i]))

    weight_str = "distance" if args.enable_weight else "uniform"
    boston_skmodel = KNeighborsRegressor(n_neighbors=args.k, weights=weight_str)
    boston_skmodel.fit(boston_xtrain, boston_ytrain)

    y_skpred = boston_skmodel.predict(boston_xtest)
    sk_total_diff = 0
    for i in range(n_boston_test):
        sk_total_diff += abs(y_skpred[i] - boston_ytest[i])
        logger.info("skmodel模型预测房价为：{:.1f}，\t 真实房价为：{}".format(y_skpred[i], boston_ytest[i]))

    logger.info("Scratch模型平均预测误差为：{:.1f}".format(total_diff / n_boston_test))
    logger.info("skmodel模型平均预测误差为：{:.1f}".format(sk_total_diff / n_boston_test))

if __name__ == "__main__":
    main()

