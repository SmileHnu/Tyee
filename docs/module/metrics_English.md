# tyee.metrics

The `tyee.metrics` module provides a collection of metrics for evaluating the performance of models.

## classification 

| Class Name             | `self.name`            | Metric Description                                           |
| ---------------------- | ---------------------- | ------------------------------------------------------------ |
| `Accuracy`             | `accuracy`             | Computes accuracy classification score.                      |
| `BalancedAccuracy`     | `balanced_accuracy`    | Computes the balanced accuracy, which avoids inflated performance on imbalanced datasets. |
| `CohenKappa`           | `cohen_kappa`          | Computes Cohen's kappa, a statistic that measures inter-annotator agreement. |
| `TruePositive`         | `tp`                   | Computes True Positives (TP).                                |
| `TrueNegative`         | `tn`                   | Computes True Negatives (TN).                                |
| `FalsePositive`        | `fp`                   | Computes False Positives (FP).                               |
| `FalseNegative`        | `fn`                   | Computes False Negatives (FN).                               |
| `PR_AUC`               | `pr_auc`               | Computes the area under the Precision-Recall curve.          |
| `ROC_AUC`              | `roc_auc`              | Computes the area under the Receiver Operating Characteristic curve. |
| `Precision`            | `precision`            | Computes the precision (tp / (tp + fp)).                     |
| `Recall`               | `recall`               | Computes the recall (tp / (tp + fn)).                        |
| `F1`                   | `f1`                   | Computes the F1 score, the harmonic mean of precision and recall. |
| `Jaccard`              | `jaccard`              | Computes the Jaccard similarity coefficient score.           |
| `ROC_AUC_Macro_OVO`    | `roc_auc_macro_ovo`    | Computes ROC AUC score with macro averaging and One-vs-One multi-class strategy. |
| `ROC_AUC_Macro_OVR`    | `roc_auc_macro_ovr`    | Computes ROC AUC score with macro averaging and One-vs-Rest multi-class strategy. |
| `ROC_AUC_Weighted_OVO` | `roc_auc_weighted_ovo` | Computes ROC AUC score with weighted averaging and One-vs-One multi-class strategy. |
| `ROC_AUC_Weighted_OVR` | `roc_auc_weighted_ovr` | Computes ROC AUC score with weighted averaging and One-vs-Rest multi-class strategy. |
| `PR_AUC_Macro`         | `pr_auc_macro`         | Computes PR AUC score with macro averaging.                  |
| `PR_AUC_Weighted`      | `pr_auc_weighted`      | Computes PR AUC score with weighted averaging.               |
| `F1_Macro`             | `f1_macro`             | Computes F1 score with macro averaging.                      |
| `F1_Micro`             | `f1_micro`             | Computes F1 score with micro averaging.                      |
| `F1_Weighted`          | `f1_weighted`          | Computes F1 score with weighted averaging.                   |
| `Precision_Macro`      | `precision_macro`      | Computes precision with macro averaging.                     |
| `Precision_Micro`      | `precision_micro`      | Computes precision with micro averaging.                     |
| `Precision_Weighted`   | `precision_weighted`   | Computes precision with weighted averaging.                  |
| `Recall_Macro`         | `recall_macro`         | Computes recall with macro averaging.                        |
| `Recall_Micro`         | `recall_micro`         | Computes recall with micro averaging.                        |
| `Recall_Weighted`      | `recall_weighted`      | Computes recall with weighted averaging.                     |
| `jaccard_Macro`        | `jaccard_macro`        | Computes Jaccard score with macro averaging.                 |
| `jaccard_Micro`        | `jaccard_micro`        | Computes Jaccard score with micro averaging.                 |
| `jaccard_Weighted`     | `jaccard_weighted`     | Computes Jaccard score with weighted averaging.              |

## regression

| Class Name           | `self.name`           | Metric Description                                           |
| -------------------- | --------------------- | ------------------------------------------------------------ |
| `MeanSquaredError`   | `mse`                 | Computes Mean Squared Error.                                 |
| `R2Score`            | `r2`                  | Computes R-squared (coefficient of determination).           |
| `MeanAbsoluteError`  | `mae`                 | Computes Mean Absolute Error.                                |
| `RMSE`               | `rmse`                | Computes Root Mean Squared Error.                            |
| `CosineSimilarity`   | `cosine_similarity`   | Computes the mean cosine similarity between targets and outputs. |
| `PearsonCorrelation` | `pearson_correlation` | Computes the Pearson correlation coefficient between targets and outputs. |
| `MeanCC`             | `mean_cc`             | Computes the mean of Pearson correlation coefficients across channels/features (outputs are Gaussian filtered). |

## `MetricEvaluator`

The `MetricEvaluator` class is a utility designed to simplify the process of calculating multiple metrics during model evaluation. It handles the dynamic creation of metric objects, aggregates batch-by-batch results, and computes all metrics in a single step.

**Initialization**

To initialize the evaluator, you need to provide a **list of metric names** (strings) that correspond to the `self.name` attribute of the metric classes.

**evaluation process**

1. **Update Results in a Loop**: Inside your evaluation loop (iterating over batches of data), call the `update_metrics` method for each batch. Pass a dictionary containing at least the `'label'` (ground truth) and `'output'` (model prediction) keys. `MetricEvaluator` will automatically handle the conversion from `torch.Tensor` to `numpy` arrays.
2. **Calculate Final Metrics**: After the loop finishes, call the `calculate_metrics` method. This will compute all the specified metrics using the aggregated results and return a dictionary containing the metric names and their computed values (rounded to 4 decimal places). The internal results list is then cleared for the next evaluation run.

**example**

~~~python
# 1. Initialize MetricEvaluator with a list of desired metric names
evaluator = MetricEvaluator(metric_list=['mse', 'mae'])

# Assume this is your evaluation loop
for batch in validation_dataloader:
    # Get data from the dataloader
    inputs, labels = batch
    
    # Get model predictions
    outputs = model(inputs)
    
    # Prepare the result dictionary
    result_dict = {'label': labels, 'output': outputs}
    
    # 2. Update the results for the current batch
    evaluator.update_metrics(result_dict)

# 3. After the loop, calculate all metrics
final_metrics = evaluator.calculate_metrics()

# Print the final results
# Example output: {'mse': 0.0233, 'mae': 0.1167}
print("Metrics Result:", final_metrics)
~~~

