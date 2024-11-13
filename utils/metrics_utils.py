import numpy as np
from .import_utils import lazy_import_module


def build_metrics(metric_dict: dict):
    """
    根据配置字典动态构建并返回指标类实例列表。

    该方法遍历传入的字典 `metric_dict`，从指定模块中动态导入每个指标类，并根据配置的参数实例化相应的指标对象。 
    所有实例化的指标对象将被添加到 `metrics` 列表中并返回。

    :param metric_dict: dict, 配置字典，其中包含指标名称及其对应的参数。
                        字典的键是指标函数的名称（例如 'accuracy_score'），值是该函数需要的参数（字典形式）。
                        示例：{'accuracy_score': {}, 'precision_score': {'average': 'micro'}}。
    :return: list, 指标类实例的列表，包含所有根据配置文件实例化的指标对象。
    """
    metrics = []
    # 遍历指标字典，动态导入相应的指标函数并计算
    for metric_name, params in metric_dict.items():
        # 动态导入sklearn.metrics中的指标函数
        metric_cls = lazy_import_module("metrics", metric_name)
        
        # 计算指标值
        metric = metric_cls( **params) if params else metric_cls()
        
        # 将结果存入列表
        metrics.append(metric)
    return metrics

def calculate_metrics(metric_list: list, params: dict):
    """
    根据列表动态计算指标，并返回计算结果的字典。

    :param metric_list: dist, 指定要计算的指标类。
    param params: 字典，包含 'y_true' 和 'y_pred'

    :return: dict, 包含每个指标名称和对应计算结果的字典。
    """
    metrics_result = {}

    for metric in metric_list:
        # 获取指标实例的类名称作为字典的键
        metric_name = metric.__class__.__name__
        
        # 调用每个指标实例的 compute 方法计算指标结果
        result = metric.compute(params)
        
        # 将计算结果添加到结果字典
        metrics_result[metric_name] = result

    return metrics_result