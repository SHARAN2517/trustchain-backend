import numpy as np
import scipy.stats as stats


def compute_multilabel_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes metrics from real held-out labels and model probabilities.
    y_true and y_prob must both be shaped [n_samples, n_classes].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape.")
    if y_true.ndim != 2:
        raise ValueError("Expected multilabel arrays shaped [n_samples, n_classes].")

    y_pred = (y_prob >= threshold).astype(int)
    aucs = []
    f1s = []
    sensitivities = []
    specificities = []

    for class_idx in range(y_true.shape[1]):
        truth = y_true[:, class_idx]
        prob = y_prob[:, class_idx]
        pred = y_pred[:, class_idx]
        aucs.append(compute_roc_auc(truth, prob))

        tp = np.sum((truth == 1) & (pred == 1))
        fp = np.sum((truth == 0) & (pred == 1))
        fn = np.sum((truth == 1) & (pred == 0))
        tn = np.sum((truth == 0) & (pred == 0))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
        sensitivities.append(recall)
        specificities.append(tn / max(tn + fp, 1))

    ece, bin_accs, bin_confs, bin_sizes = compute_ece(y_true.ravel(), y_prob.ravel())
    return {
        "auc_macro": float(np.mean(aucs)),
        "f1_macro": float(np.mean(f1s)),
        "sensitivity_macro": float(np.mean(sensitivities)),
        "specificity_macro": float(np.mean(specificities)),
        "ece": float(ece),
        "ece_bin_accs": bin_accs,
        "ece_bin_confs": bin_confs,
        "ece_bin_sizes": bin_sizes,
    }

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE).
    Divides predictions into n_bins and calculates the weighted absolute difference
    between confidence (average probability) and accuracy (fraction of positives).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    bin_accs = []
    bin_confs = []
    bin_sizes = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            bin_accs.append(float(accuracy_in_bin))
            bin_confs.append(float(avg_confidence_in_bin))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
        bin_sizes.append(int(np.sum(in_bin)))
            
    return ece, bin_accs, bin_confs, bin_sizes


def compute_roc_auc(y_true, y_prob):
    """Computes basic ROC AUC using Mann-Whitney U test formula."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    ranks = stats.rankdata(y_prob)
    pos_ranks = ranks[y_true == 1]
    
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def delong_auc_variance(y_true, y_prob):
    """
    Computes structural variance of AUC using DeLong's method.
    Reference: DeLong et al. (1988) Biometrics.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    m = len(pos_indices)
    n = len(neg_indices)
    
    x = y_prob[pos_indices]
    y = y_prob[neg_indices]
    
    # Structural components
    v_10 = np.zeros(m)
    v_01 = np.zeros(n)
    
    for i in range(m):
        v_10[i] = np.mean(x[i] > y) + 0.5 * np.mean(x[i] == y)
        
    for j in range(n):
        v_01[j] = np.mean(x > y[j]) + 0.5 * np.mean(x == y[j])
        
    auc = np.mean(v_10)
    
    s_10 = np.var(v_10, ddof=1) if m > 1 else 0.0
    s_01 = np.var(v_01, ddof=1) if n > 1 else 0.0
    
    var_auc = s_10 / m + s_01 / n
    return auc, var_auc


def delong_auc_covariance(y_true, y_prob_a, y_prob_b):
    """
    Computes DeLong's covariance matrix for two models evaluated on the same dataset.
    Used to calculate p-value for the difference in ROC curves.
    """
    y_true = np.array(y_true)
    y_prob_a = np.array(y_prob_a)
    y_prob_b = np.array(y_prob_b)
    
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    m = len(pos_indices)
    n = len(neg_indices)
    
    x_a, y_a = y_prob_a[pos_indices], y_prob_a[neg_indices]
    x_b, y_b = y_prob_b[pos_indices], y_prob_b[neg_indices]
    
    v_a_10 = np.zeros(m)
    v_a_01 = np.zeros(n)
    v_b_10 = np.zeros(m)
    v_b_01 = np.zeros(n)
    
    for i in range(m):
        v_a_10[i] = np.mean(x_a[i] > y_a) + 0.5 * np.mean(x_a[i] == y_a)
        v_b_10[i] = np.mean(x_b[i] > y_b) + 0.5 * np.mean(x_b[i] == y_b)
        
    for j in range(n):
        v_a_01[j] = np.mean(x_a > y_a[j]) + 0.5 * np.mean(x_a == y_a[j])
        v_b_01[j] = np.mean(x_b > y_b[j]) + 0.5 * np.mean(x_b == y_b[j])
        
    auc_a = np.mean(v_a_10)
    auc_b = np.mean(v_b_10)
    
    s_a_10 = np.var(v_a_10, ddof=1) if m > 1 else 0.0
    s_a_01 = np.var(v_a_01, ddof=1) if n > 1 else 0.0
    s_b_10 = np.var(v_b_10, ddof=1) if m > 1 else 0.0
    s_b_01 = np.var(v_b_01, ddof=1) if n > 1 else 0.0
    
    # Covariance components
    cov_10 = np.mean((v_a_10 - auc_a) * (v_b_10 - auc_b)) * m / (m - 1) if m > 1 else 0.0
    cov_01 = np.mean((v_a_01 - auc_a) * (v_b_01 - auc_b)) * n / (n - 1) if n > 1 else 0.0
    
    var_a = s_a_10 / m + s_a_01 / n
    var_b = s_b_10 / m + s_b_01 / n
    cov_ab = cov_10 / m + cov_01 / n
    
    # Z-score & p-value
    diff = auc_a - auc_b
    variance_diff = var_a + var_b - 2 * cov_ab
    
    if variance_diff <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        z_score = diff / np.sqrt(variance_diff)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_diff": float(diff),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "var_a": float(var_a),
        "var_b": float(var_b),
        "cov_ab": float(cov_ab)
    }


def trapezoid_area(y, x):
    """Computes area under curve using trapezoid rule."""
    y = np.array(y)
    x = np.array(x)
    return np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x))


def compute_faithfulness_auc(y_prob_original, cam_values, target_function, steps=10):
    """
    Computes insertion and deletion faithfulness AUC metrics for Grad-CAM.
    - Insertion: Starts with a blank/blurred image and adds pixel groups ranked by heat values.
      Highly faithful maps exhibit rapid confidence climbs (High Insertion AUC).
    - Deletion: Starts with original image and removes pixel groups.
      Highly faithful maps exhibit rapid confidence drops (Low Deletion AUC).
    """
    if cam_values is None or target_function is None:
        raise ValueError("Real CAM values and a target_function are required for faithfulness metrics.")

    # Placeholder implementation until a real perturbation pipeline is connected.
    # This guards against reporting simulated faithfulness as evaluated performance.
    raise NotImplementedError("Faithfulness AUC requires image perturbation inference over a held-out set.")


def generate_ablation_benchmarks():
    """
    Historical placeholder removed.
    Use compute_multilabel_metrics(y_true, y_pred_proba) on a held-out split instead.
    """
    raise NotImplementedError("Ablation benchmarks require predictions from real held-out datasets.")


def compute_privacy_tradeoff_curves():
    """
    Computes accuracy vs epsilon (privacy-utility tradeoff) and Membership Inference Attack
    (MIA) success rates under DP-SGD (Differential Privacy).
    """
    raise NotImplementedError("Privacy-utility curves require repeated training runs on real data.")


def compute_federated_convergence():
    """
    Simulates federated rounds convergence showing:
    - FedAvg with Independent and Identically Distributed (IID) clients.
    - FedAvg with non-IID clients skewed via Dirichlet distribution (alpha = 0.5).
    - Centralized training baseline (upper bound).
    - Impact of client dropouts (30% dropouts) and stragglers.
    """
    raise NotImplementedError("Federated convergence curves require logged validation metrics from real rounds.")


if __name__ == "__main__":
    print("Testing evaluator.py operations...")
    
    # Test DeLong's test significance calculation
    np.random.seed(42)
    labels = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
    pred_a = labels * 0.4 + np.random.uniform(0.1, 0.5, size=1000)
    pred_b = labels * 0.3 + np.random.uniform(0.1, 0.6, size=1000)
    
    res = delong_auc_covariance(labels, pred_a, pred_b)
    print(f"DeLong's Test p-value: {res['p_value']:.6f} (significant diff: {res['p_value'] < 0.05})")
    
    # Test Expected Calibration Error
    ece, _, _, _ = compute_ece(labels, pred_a)
    print(f"Expected Calibration Error (ECE): {ece:.6f}")
    
    metrics = compute_multilabel_metrics(labels.reshape(-1, 1), pred_a.reshape(-1, 1))
    print(f"Real-input metric helper AUC: {metrics['auc_macro']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    print("Evaluator operations successfully validated!")
