
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger
from util import plot_scatter


class KMeansScratch():
    """kmeans Scratch实现"""
    def __init__(self, k=3, max_iter=1000):
        """
        k:
            聚类中心个数
        max_iter:
            最大迭代次数
        """
        self._k = k
        self._max_iter = max_iter

    def fit(self, X):
        """模型训练"""
        n_sample, n_feature = X.shape
        # 每个簇的中心
        self._centroid = np.zeros((self._k, n_feature))
        # 每个簇所包含的样本索引
        self._cluster = [[] for _ in range(self._k)]

        rng = np.random.default_rng()
        # 随机获取k个索引
        rnd_idx = rng.permutation(n_sample)[:self._k]
        # 初始随机k个中心
        self._centroid = X[rnd_idx]

        for _ in range(self._max_iter):
            for i in range(n_sample):
                # 样本i距离其它样本的最小距离
                min_dist = np.inf
                # 最小距离所对应的中心
                min_centroid = 0
                for j in range(self._k):
                    # 第i个样本与第j个中心的距离
                    dist = np.linalg.norm(X[i] - self._centroid[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_centroid = j
                # 将第i个样本划分到最近的簇
                self._cluster[min_centroid].append(i)

            prev_centroid = self._centroid
            # 更新聚类中心
            for j in range(self._k):
                # 该簇所有样本索引
                idx = self._cluster[j]
                self._centroid[j] = np.mean(X[idx], axis=0)

            centroid_diff = self._centroid - prev_centroid
            # 如果聚类中心不再移动则收敛
            if np.all(centroid_diff == 0):
                break

    def predict(self, X):
        """为每个样本赋值一个聚类中心"""
        y_pred = np.zeros(X.shape[0])
        for i in range(self._k):
            # 遍历该簇样本
            for j in self._cluster[i]:
                y_pred[j] = i
        return y_pred

        # n_sample = X.shape[0]
        # y_pred = np.zeros(n_sample)
        # for i in range(n_sample):
        #     min_dist = np.inf
        #     min_idx = 0
        #     for j in range(self._k):
        #         dist = np.linalg.norm(X[i] - self._centroid[j])
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_idx = j
        #     # 距离最近的中心的类别作为预测类别
        #     y_pred[i] = min_idx
        # return y_pred


def main():
    parser = argparse.ArgumentParser(description="kmeans算法Scratch代码命令行参数")
    parser.add_argument("--k", type=int, default=3, help="聚类中心")
    parser.add_argument("--max_iter", type=int, default=1000, help="最大迭代次数")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    xtrain, _, ytrain, _ = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = KMeansScratch(args.k, args.max_iter)
    model.fit(xtrain)
    y_pred = model.predict(xtrain)

    # 用真实标签绘制散点图
    plot_scatter(xtrain, ytrain)

    # 用聚类标签绘制散点图
    plot_scatter(xtrain, y_pred)

if __name__ == "__main__":
    main()

