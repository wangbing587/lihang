
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from loguru import logger


class LogisticRegressionScratch():
    """Logistic算法Scratch实现"""
    def __init__(self, n_epoch=500, learning_rate=0.01, loss_tolerance=0.0001):
        """
        n_epoch:
            训练时迭代多少个epoch之后终止训练
        learning_rate:
            学习率
        loss_tolerance:
            当前损失与上一个epoch损失之差的绝对值小于loss_tolerance时终止训练
        """
        self._n_epoch = n_epoch
        self._lr = learning_rate
        self._loss_tolerance = loss_tolerance

    def fit(self, X, y):
        """
        模型训练
        X:
            训练集，每一行表示一个样本，每一列表示一个特征或属性
        y:
            训练集标签
        """
        n_sample, n_feature = X.shape
        rnd_val = 1 / np.sqrt(n_feature)
        rng = np.random.default_rng()
        # 均匀随机初始化权重参数
        self._w = rng.uniform(-rnd_val, rnd_val, size=n_feature)
        # 偏置初始化为0
        self._b = 0

        num_epoch = 0
        prev_loss = 0
        while True:
            cur_loss = 0
            for i in range(n_sample):
                y_pred = self._sigmoid(np.dot(self._w, X[i]) + self._b)
                y_diff = y[i] - y_pred
                self._w += self._lr * y_diff * X[i]
                self._b += self._lr * y_diff
                cur_loss += abs(y_diff)
            num_epoch += 1
            loss_diff = cur_loss - prev_loss
            prev_loss = cur_loss

            # 训练终止条件：
            # 1. 训练epoch数达到指定的epoch数时停止训练
            # 2. 本epoch损失与上一个epoch损失差异小于指定的阈值时停止训练
            if num_epoch >= self._n_epoch or abs(loss_diff) < self._loss_tolerance:
                break

    def predict(self, x):
        """给定输入样本，预测其类别"""
        logit = np.dot(self._w, x) + self._b
        return 1 if logit > 0 else 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser(description="Logistic回归算法Scratch代码命令行参数")
    parser.add_argument("--nepoch", type=int, default=500, help="训练多少个epoch后终止训练")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--loss_tolerance", type=float, default=0.0001, help="当前损失与上一个epoch损失之差的绝对值小于该值时终止训练")
    args = parser.parse_args()

    X, y = load_iris(return_X_y=True)
    xtrain, xtest, ytrain, ytest = train_test_split(X[:100], y[:100], train_size=0.8, shuffle=True)

    model = LogisticRegressionScratch(args.nepoch, args.lr, args.loss_tolerance)
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

    skmodel = LogisticRegression(max_iter=args.nepoch)
    skmodel.fit(xtrain, ytrain)
    logger.info("sklearn模型在测试集上准确率为：{}%".format(100 * skmodel.score(xtest, ytest)))

if __name__ == "__main__":
    main()

