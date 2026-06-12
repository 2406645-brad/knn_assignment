import numpy as np

def accuracy_score(y_true, y_pred):
    """Compute the accuracy score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) == 0:
        return 0.0
    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Returns a 2D numpy array.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.array(labels)
        
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    cm = np.zeros((n_labels, n_labels), dtype=np.int64)
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1
            
    return cm


def precision_score(y_true, y_pred, pos_label=1):
    """Compute the precision (TP / (TP + FP)) for binary classification."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(y_true, y_pred, pos_label=1):
    """Compute the recall (TP / (TP + FN)) for binary classification."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred, pos_label=1):
    """Compute the F1 score (2 * (precision * recall) / (precision + recall))."""
    prec = precision_score(y_true, y_pred, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    
    if (prec + rec) == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def classification_report(y_true, y_pred, labels=None):
    """
    Build a text report showing the main classification metrics.
    Works for binary classification and multiclass labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    else:
        labels = np.array(labels)
        
    report = f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
    report += "-" * 55 + "\n"
    
    total_samples = len(y_true)
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        support = np.sum(y_true == label)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)
        
        report += f"{str(label):<12} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {support:<10}\n"
        
    report += "-" * 55 + "\n"
    
    # Calculate macro average
    macro_prec = np.mean(precisions)
    macro_rec = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    report += f"{'macro avg':<12} {macro_prec:<10.4f} {macro_rec:<10.4f} {macro_f1:<10.4f} {total_samples:<10}\n"
    
    # Calculate weighted average
    weights = np.array(supports) / total_samples if total_samples > 0 else np.zeros_like(supports)
    weighted_prec = np.sum(np.array(precisions) * weights)
    weighted_rec = np.sum(np.array(recalls) * weights)
    weighted_f1 = np.sum(np.array(f1s) * weights)
    report += f"{'weighted avg':<12} {weighted_prec:<10.4f} {weighted_rec:<10.4f} {weighted_f1:<10.4f} {total_samples:<10}\n"
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    report += f"{'accuracy':<12} {'':<10} {'':<10} {acc:<10.4f} {total_samples:<10}\n"
    
    return report
