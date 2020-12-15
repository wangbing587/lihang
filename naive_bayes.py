
from collections import defaultdict

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger


class NaiveBayesScratch():
    """朴素贝叶斯算法Scratch实现"""
    def __init__(self):
        # 存储先验概率 P(Y=ck)
        self._prior_prob = defaultdict(float)
        # 存储似然概率 P(X|Y=ck)
        self._likelihood = defaultdict(defaultdict)
        # 存储每个类别的样本在训练集中出现次数
        self._ck_counter = defaultdict(float)
        # 存储每一个特征可能取值的个数
        self._Sj = defaultdict(float)

    def fit(self, X, y):
        """
        模型训练，参数估计使用贝叶斯估计
        X:
            训练集，每一行表示一个样本，每一列表示一个特征或属性
        y:
            训练集标签
        """
        n_sample, n_feature = X.shape
        # 计算每个类别可能的取值以及每个类别样本个数
        ck, num_ck = np.unique(y, return_counts=True)
        self._ck_counter = dict(zip(ck, num_ck))
        for label, num_label in self._ck_counter.items():
            # 计算先验概率，做了拉普拉斯平滑处理
            self._prior_prob[label] = (num_label + 1) / (n_sample + ck.shape[0])

        # 记录每个类别样本对应的索引
        ck_idx = []
        for label in ck:
            label_idx = np.squeeze(np.argwhere(y == label))
            ck_idx.append(label_idx)

        # 遍历每个类别
        for label, idx in zip(ck, ck_idx):
            xdata = X[idx]
            # 记录该类别所有特征对应的概率
            label_likelihood = defaultdict(defaultdict)
            # 遍历每个特征
            for i in range(n_feature):
                # 记录该特征每个取值对应的概率
                feature_val_prob = defaultdict(float)
                # 获取该列特征可能的取值和每个取值出现的次数
                feature_val, feature_cnt = np.unique(xdata[:, i], return_counts=True)
                self._Sj[i] = feature_val.shape[0]
                feature_counter = dict(zip(feature_val, feature_cnt))
                for fea_val, cnt in feature_counter.items():
                    # 计算该列特征每个取值的概率，做了拉普拉斯平滑
                    feature_val_prob[fea_val] = (cnt + 1) / (self._ck_counter[label] + self._Sj[i])
                label_likelihood[i] = feature_val_prob
            self._likelihood[label] = label_likelihood

    def predict(self, x):
        """
        输入样本，输出其类别，本质上是计算后验概率
        **注意计算后验概率的时候对概率取对数**，概率连乘可能导致浮点数下溢，取对数将连乘转化为求和
        """
        # 保存分类到每个类别的后验概率
        post_prob = defaultdict(float)
        # 遍历每个类别计算后验概率
        for label, label_likelihood in self._likelihood.items():
            prob = np.log(self._prior_prob[label])
            # 遍历样本每一维特征
            for i, fea_val in enumerate(x):
                feature_val_prob = label_likelihood[i]
                # 如果该特征值出现在训练集中则直接获取概率
                if fea_val in feature_val_prob:
                    prob += np.log(feature_val_prob[fea_val])
                else:
                    # 如果该特征没有出现在训练集中则采用拉普拉斯平滑计算概率
                    laplace_prob = 1 / (self._ck_counter[label] + self._Sj[i])
                    prob += np.log(laplace_prob)
            post_prob[label] = prob
        prob_list = list(post_prob.items())
        prob_list.sort(key=lambda v: v[1], reverse=True)
        # 返回后验概率最大的类别作为预测类别
        return prob_list[0][0]


def main():
    X, y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = NaiveBayesScratch()
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

if __name__ == "__main__":
    main()

