# tyee.metrics

`tyee.metrics` 模块提供了一系列用于评估模型性能的指标。

## 分类指标

| 类名                   | `self.name`            | 指标描述                              |
| ---------------------- | ---------------------- | ------------------------------------- |
| `Accuracy`             | `accuracy`             | 计算准确率。                          |
| `BalancedAccuracy`     | `balanced_accuracy`    | 计算平衡准确率。                      |
| `CohenKappa`           | `cohen_kappa`          | 计算科恩Kappa系数。                   |
| `TruePositive`         | `tp`                   | 计算真阳性。                          |
| `TrueNegative`         | `tn`                   | 计算真阴性。                          |
| `FalsePositive`        | `fp`                   | 计算假阳性。                          |
| `FalseNegative`        | `fn`                   | 计算假阴性。                          |
| `PR_AUC`               | `pr_auc`               | 计算PR曲线下面积。                    |
| `ROC_AUC`              | `roc_auc`              | 计算ROC曲线下面积。                   |
| `Precision`            | `precision`            | 计算精确率。                          |
| `Recall`               | `recall`               | 计算召回率。                          |
| `F1`                   | `f1`                   | 计算F1分数。                          |
| `Jaccard`              | `jaccard`              | 计算杰卡德相似系数。                  |
| `ROC_AUC_Macro_OVO`    | `roc_auc_macro_ovo`    | 计算ROC AUC分数 (宏平均, OVO策略)。   |
| `ROC_AUC_Macro_OVR`    | `roc_auc_macro_ovr`    | 计算ROC AUC分数 (宏平均, OVR策略)。   |
| `ROC_AUC_Weighted_OVO` | `roc_auc_weighted_ovo` | 计算ROC AUC分数 (加权平均, OVO策略)。 |
| `ROC_AUC_Weighted_OVR` | `roc_auc_weighted_ovr` | 计算ROC AUC分数 (加权平均, OVR策略)。 |
| `PR_AUC_Macro`         | `pr_auc_macro`         | 计算PR曲线下面积 (宏平均)。           |
| `PR_AUC_Weighted`      | `pr_auc_weighted`      | 计算PR曲线下面积 (加权平均)。         |
| `F1_Macro`             | `f1_macro`             | 计算F1分数 (宏平均)。                 |
| `F1_Micro`             | `f1_micro`             | 计算F1分数 (微平均)。                 |
| `F1_Weighted`          | `f1_weighted`          | 计算F1分数 (加权平均)。               |
| `Precision_Macro`      | `precision_macro`      | 计算精确率 (宏平均)。                 |
| `Precision_Micro`      | `precision_micro`      | 计算精确率 (微平均)。                 |
| `Precision_Weighted`   | `precision_weighted`   | 计算精确率 (加权平均)。               |
| `Recall_Macro`         | `recall_macro`         | 计算召回率 (宏平均)。                 |
| `Recall_Micro`         | `recall_micro`         | 计算召回率 (微平均)。                 |
| `Recall_Weighted`      | `recall_weighted`      | 计算召回率 (加权平均)。               |
| `jaccard_Macro`        | `jaccard_macro`        | 计算杰卡德相似系数 (宏平均)。         |
| `jaccard_Micro`        | `jaccard_micro`        | 计算杰卡德相似系数 (微平均)。         |
| `jaccard_Weighted`     | `jaccard_weighted`     | 计算杰卡德相似系数 (加权平均)。       |

## 回归指标

| 类名                 | `self.name`           | 指标描述                                                     |
| -------------------- | --------------------- | ------------------------------------------------------------ |
| `MeanSquaredError`   | `mse`                 | 计算均方误差。                                               |
| `R2Score`            | `r2`                  | 计算R²分数 (决定系数)。                                      |
| `MeanAbsoluteError`  | `mae`                 | 计算平均绝对误差。                                           |
| `RMSE`               | `rmse`                | 计算均方根误差。                                             |
| `CosineSimilarity`   | `cosine_similarity`   | 计算目标与输出之间的平均余弦相似度。                         |
| `PearsonCorrelation` | `pearson_correlation` | 计算目标与输出之间的皮尔逊相关系数。                         |
| `MeanCC`             | `mean_cc`             | 计算各通道/特征上皮尔逊相关系数的平均值 (输出经过高斯滤波处理)。 |

## `tyee.utils.MetricEvaluator` 

`MetricEvaluator` 类是一个工具类，旨在简化模型评估过程中多个指标的计算。它负责动态创建指标对象、聚合每批次的计算结果，并通过一步操作计算所有指标。

**初始化**

初始化评估器时，您需要提供一个**指标名称的列表** (字符串)，这些名称对应于各个指标类中的 `self.name` 属性。

**评估过程**

1. **在循环中更新结果**: 在您的评估循环中（遍历数据批次时），为每个批次调用 `update_metrics` 方法。传入一个至少包含 `'label'` (真实值) 和 `'output'` (模型预测值) 键的字典。`MetricEvaluator` 会自动处理 `torch.Tensor` 到 `numpy` 数组的转换。
2. **计算最终指标**: 循环结束后，调用 `calculate_metrics` 方法。该方法将使用聚合后的结果计算所有指定的指标，并返回一个包含指标名称和计算结果（保留四位小数）的字典。内部结果列表随后会被清空，为下一次评估做准备。

**使用样例**

~~~python
# 1. 初始化 MetricEvaluator，传入需要计算的指标名称列表
evaluator = MetricEvaluator(metric_list=['mse', 'mae'])

# 假设这是您的评估循环
for batch in validation_dataloader:
    # 从数据加载器获取数据
    inputs, labels = batch
    
    # 模型进行预测
    outputs = model(inputs)
    
    # 准备结果字典
    result_dict = {'label': labels, 'output': outputs}
    
    # 2. 更新当前批次的结果
    evaluator.update_metrics(result_dict)

# 3. 循环结束后，计算所有指标
final_metrics = evaluator.calculate_metrics()

# 打印最终结果
# 输出示例: {'mse': 0.0233, 'mae': 0.1167}
print("Metrics Result:", final_metrics)
~~~

