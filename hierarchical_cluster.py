
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger
from util import plot_scatter


class HierarchicalClusterScratch():
    """层次聚类算法Scratch实现"""
    def __init__(self, k=3):
        """
        k:
            类别个数
        """
        self._k = k

    def fit(self, X):
        """自底向上聚合"""
        n_sample = X.shape[0]
        self._cluster = [X[i] for i in range(n_sample)]
        while len(self._cluster) > self._k:
            n_sample = len(self._cluster)
            cluster_copy = self._cluster.copy()
            self._cluster.clear()
            # 两两计算距离距离最近的两个样本
            min_dist = np.inf
            min_i = 0
            min_j = 0
            for i in range(n_sample):
                for j in range(n_sample):
                    if i == j:
                        continue
                    dist = np.linalg.norm(cluster_copy[i] - cluster_copy[j])
                    if dist < min_dist:
                        min_dist = dist
                        min_i = i
                        min_j = j
            # 合并两个距离最近的中心为新的中心
            new_cluster = (cluster_copy[min_i] + cluster_copy[min_j]) / 2
            self._cluster.append(new_cluster)
            for k in range(n_sample):
                if k == min_i or k == min_j:
                    continue
                self._cluster.append(cluster_copy[k])

    def predict(self, X):
        """为每一个样本指定一个聚类中心"""
        n_sample = X.shape[0]
        y_pred = np.zeros(n_sample)
        for i in range(n_sample):
            min_dist = np.inf
            min_idx = 0
            for j in range(self._k):
                dist = np.linalg.norm(X[i] - self._cluster[j])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            # 距离最近的中心的类别作为预测类别
            y_pred[i] = min_idx
        return y_pred


def main():
    parser = argparse.ArgumentParser(description="层次聚类算法Scratch代码命令行参数")
    parser.add_argument("--k", type=int, default=3, help="聚类中心")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    xtrain, _, ytrain, _ = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = HierarchicalClusterScratch(args.k)
    model.fit(xtrain)
    y_pred = model.predict(xtrain)

    # 用真实标签绘制散点图
    plot_scatter(xtrain, ytrain)

    # 用聚类标签绘制散点图
    plot_scatter(xtrain, y_pred)

if __name__ == "__main__":
    main()

