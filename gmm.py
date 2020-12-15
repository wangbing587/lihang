
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger
from util import plot_scatter


class GMMScratch():
    """高斯混合模型Scratch实现"""
    def __init__(self, k=3, max_iter=1000, tolerance=1e-6):
        """
        k:
            高斯模型个数或者说聚类中心个数
        max_iter:
            最大迭代次数
        tolerance:
            收敛阈值，当前与上一次迭代之差小于该阈值时认为收敛
        """
        self._k = k
        self._max_iter = max_iter
        self._tolerance = tolerance
        # 所有高斯模型的参数
        self._gmm_params = []
        # 保存响应度
        self._resp = None
        # 保存每个样本所属聚类中心
        self._sample_cluster = None
        # 本轮迭代最大响应度
        self._cur_max_resp = None
        # 上一轮迭代最大响应度
        self._prev_max_resp = None

    def fit(self, X):
        """训练高斯混合模型"""
        # 初始化高斯混合模型参数
        self._init_params(X)

        for i in range(self._max_iter):
            self._e_step(X)
            self._m_step(X)

            if i > 2:
                resp_diff = np.linalg.norm(self._cur_max_resp - self._prev_max_resp)
                # 当前与上一轮最大响应度之差小于给定阈值时认为收敛
                if resp_diff < self._tolerance:
                    break
            self._prev_max_resp = self._cur_max_resp

    def predict(self, X):
        """为每个样本赋值一个聚类中心"""
        # 使用更新后的参数重新给样本指定中心
        self._e_step(X)
        return self._sample_cluster

    def _e_step(self, X):
        """期望最大化的E步"""
        n_sample = X.shape[0]
        # 每列为一个高斯模型的概率密度
        likelihoods = np.zeros((n_sample, self._k))
        for i in range(self._k):
            likelihoods[:, i] = self._alpha[i] * self._gaussian_pdf(X, self._gmm_params[i])
        # 混合高斯模型，每行求和
        sum_likelihoods = np.sum(likelihoods, axis=1)
        # 计算响应度
        self._resp = likelihoods / np.expand_dims(sum_likelihoods, axis=1)
        # 为每一个样本赋值一个聚类中心，使用响应度最大的中心
        self._sample_cluster = self._resp.argmax(axis=1)
        # 保存每次迭代的最大响应度，用于收敛判断
        self._cur_max_resp = np.amax(self._resp, axis=1)

    def _m_step(self, X):
        """期望最大化的M步"""
        n_sample = X.shape[0]
        for i in range(self._k):
            # 第i个高斯模型对应的响应度
            resp = np.expand_dims(self._resp[:, i], axis=1)
            # 更新均值
            mean = np.sum(resp * X, axis=0) / np.sum(resp)
            # 更新协方差
            covar = (X - mean).T.dot((X - mean) * resp) / np.sum(resp)
            self._gmm_params[i]["mean"] = mean
            self._gmm_params[i]["cov"] = covar
        # 更新高斯模型的权重
        self._alpha = np.sum(self._resp, axis=0) / n_sample

    def _init_params(self, X):
        """初始化高斯模型参数"""
        n_sample = X.shape[0]
        # 初始每个高斯模型权重均等
        self._alpha = np.ones(self._k) / self._k
        for _ in range(self._k):
            params = dict()
            # 初始化高斯模型均值和协方差
            rng = np.random.default_rng()
            params["mean"] = X[rng.choice(np.arange(n_sample))]
            params["cov"] = np.cov(X.T)
            self._gmm_params.append(params)

    def _gaussian_pdf(self, X, params):
        """
        给定数据集和高斯模型参数，计算高斯模型概率，概率密度函数如下
        p(x) = \frac{1}{(2 \pi)^{\frac{n}{2}} |\Sigma|^{\frac{1}{2}}} \exp[-\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu)]
        """
        n_sample, n_feature = X.shape
        mean = params["mean"]
        covar = params["cov"]
        # 协方差矩阵的行列式
        determinant = np.linalg.det(covar)
        # 概率密度
        likelihood = np.zeros(n_sample)
        # 概率密度前面的系数
        coeff = 1 / np.sqrt(np.power(2*np.pi, n_feature) * determinant)
        for i, x in enumerate(X):
            # 概率密度指数部分
            exponent = np.exp(-0.5 * (x - mean).T.dot(np.linalg.pinv(covar)).dot(x - mean))
            likelihood[i] = coeff * exponent
        return likelihood


def main():
    parser = argparse.ArgumentParser(description="高斯混合模型算法Scratch代码命令行参数")
    parser.add_argument("--k", type=int, default=3, help="聚类中心或高斯模型个数")
    parser.add_argument("--max_iter", type=int, default=1000, help="最大迭代次数")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="模型收敛阈值")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    xtrain, _, ytrain, _ = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = GMMScratch(args.k, args.max_iter, args.tolerance)
    model.fit(xtrain)
    y_pred = model.predict(xtrain)

    # 用真实标签绘制散点图
    plot_scatter(xtrain, ytrain)

    # 用聚类标签绘制散点图
    plot_scatter(xtrain, y_pred)

if __name__ == "__main__":
    main()

