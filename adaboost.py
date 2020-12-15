
import argparse

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger


class WeakClassifier():
    """adaboost弱分类器信息"""
    def __init__(self):
        # 划分特征索引
        self.feature_idx = None
        # 划分值
        self.fea_val = None
        # 取阈值方式，大于或小于划分值
        self.threshold_type = None
        # 弱分类器权重
        self.alpha = None


class AdaboostScratch():
    """adaboost算法Scratch实现"""
    def __init__(self, ncls):
        """
        ncls:
            训练多少个弱分类器
        """
        self._ncls = ncls
        # 保存所有弱分类器
        self._classifier = []

    def fit(self, X, y):
        """模型训练"""
        n_sample, n_feature = X.shape
        # 初始化每个样本权重均等
        w = np.ones(n_sample) / n_sample

        for _ in range(self._ncls):
            # 本次迭代对应的弱分类器
            ws = WeakClassifier()
            # 最小误差
            min_error = np.inf

            # 遍历每一维特征
            for i in range(n_feature):
                feature_value = np.unique(X[:, i])
                # 遍历特征的每一个取值作为划分值，也可以设计步长从特征最小值搜索到最大值
                for fea_val in feature_value:
                    # 需要考虑大于或小于划分值两种情况下，哪种情况能使分类误差更小
                    for threshold_type in ["less", "great"]:
                        # 预测每个样本类别
                        y_pred = self._stump_predict(X, i, fea_val, threshold_type)
                        # 预测错误为1，预测正确为0
                        error_sample = np.ones(n_sample)
                        error_sample[y_pred == y] = 0
                        # 以当前特征的当前划分值划分时，分类误差
                        err = np.dot(w, error_sample)
                        if err < min_error:
                            # 记录误差最小的划分
                            min_error = err
                            ws.feature_idx = i
                            ws.fea_val = fea_val
                            ws.threshold_type = threshold_type
            # 最佳划分情况下对样本类别的预测
            y_pred = self._stump_predict(X, ws.feature_idx, ws.fea_val, ws.threshold_type)

            # 计算弱分类器权重，最小误差可能为0
            ws.alpha = 0.5 * np.log((1-min_error)/(min_error+1e-15))
            # 更新每一个样本的权重
            w *= np.exp(-ws.alpha * y * y_pred)
            w /= np.sum(w)

            self._classifier.append(ws)

    def predict(self, x):
        """预测样本类别"""
        y_pred = 0
        for cls in self._classifier:
            pred = 1
            if cls.threshold_type == "less":
                if x[cls.feature_idx] <= cls.fea_val:
                    pred = -1
            else:
                if x[cls.feature_idx] > cls.fea_val:
                    pred = -1
            y_pred += cls.alpha * pred
        return np.sign(y_pred)

    def _stump_predict(self, X, feature_idx, fea_val, threshold_type):
        """
        给定划分特征，划分值，划分类型预测数据集类别
        X:
            数据集
        feature_idx:
            划分索引
        fea_val:
            划分值
        threshold_type:
            划分类型，大于或小于
        """
        y_pred = np.ones(X.shape[0])
        if threshold_type == "less":
            y_pred[X[:, feature_idx] <= fea_val] = -1
        else:
            y_pred[X[:, feature_idx] > fea_val] = -1
        return y_pred


def main():
    parser = argparse.ArgumentParser(description="adaboost算法Scratch代码命令行参数")
    parser.add_argument("--ncls", type=int, default=30, help="训练多少个弱分类器")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    y[:50] = -1
    xtrain, xtest, ytrain, ytest = train_test_split(X[:100], y[:100], train_size=0.8, shuffle=True)

    model = AdaboostScratch(args.ncls)
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

    skmodel = AdaBoostClassifier()
    skmodel.fit(xtrain, ytrain)
    logger.info("sklearn模型在测试集上准确率为：{}%".format(100 * skmodel.score(xtest, ytest)))

if __name__ == "__main__":
    main()

