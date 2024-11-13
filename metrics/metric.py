from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 父类：指标
class Metric(ABC):
    """
    指标的父类，定义了计算指标的接口
    """

    @abstractmethod
    def compute(self, params: dict):
        """
        计算具体的指标

        :param params: 包含多个参数的字典，至少包含 'y_true' 和 'y_pred'
        :return: 返回指标的计算结果
        """
        pass


# 子类：准确率指标
class Accuracy(Metric):
    def compute(self, params: dict):
        """
        计算准确率

        :param params: 字典，包含 'y_true' 和 'y_pred'
        :return: 返回准确率
        """
        y_true = params.get('y_true')
        y_pred = params.get('y_pred')
        return accuracy_score(y_true, y_pred)


# 子类：精确度指标
class Precision(Metric):
    def __init__(self, average='binary'):
        self.average = average

    def compute(self, params: dict):
        """
        计算精确度

        :param params: 字典，包含 'y_true' 和 'y_pred'
        :return: 返回精确度
        """
        y_true = params.get('y_true')
        y_pred = params.get('y_pred')
        return precision_score(y_true, y_pred, average=self.average)


# 子类：召回率指标
class Recall(Metric):
    def __init__(self, average='binary'):
        self.average = average

    def compute(self, params: dict):
        """
        计算召回率

        :param params: 字典，包含 'y_true' 和 'y_pred'
        :return: 返回召回率
        """
        y_true = params.get('y_true')
        y_pred = params.get('y_pred')
        return recall_score(y_true, y_pred, average=self.average)


# 子类：F1分数指标
class F1(Metric):
    def __init__(self, average='binary'):
        self.average = average

    def compute(self, params: dict):
        """
        计算F1分数

        :param params: 字典，包含 'y_true' 和 'y_pred'
        :return: 返回F1分数
        """
        y_true = params.get('y_true')
        y_pred = params.get('y_pred')
        return f1_score(y_true, y_pred, average=self.average)
