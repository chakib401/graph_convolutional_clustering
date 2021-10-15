from sklearn import metrics
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, **kwargs):
    def cmat_to_psuedo_y_true_and_y_pred(cmat):
        y_true = []
        y_pred = []
        for true_class, row in enumerate(cmat):
            for pred_class, elm in enumerate(row):
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
        return y_true, y_pred

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)


def output_metrics(X, y_true, y_pred):
    return [
        clustering_accuracy(y_true, y_pred),
        nmi(y_true, y_pred),
        ari(y_true, y_pred),
        clustering_f1_score(y_true, y_pred, average='macro'),
        davies_bouldin_score(X, y_pred),
        silhouette_score(X, y_pred)
    ]


def print_metrics(metrics_means, metrics_stds, time_mean=None, time_std=None):
    if time_mean is not None: print(f'time_mean:{time_mean} ', end='')
    print(f'loss_mean:{metrics_means[6]} '
          f'acc_mean:{metrics_means[0]} '
          f'ari_mean:{metrics_means[2]} '
          f'nmi_mean:{metrics_means[1]} '
          f'db_mean:{metrics_means[4]} '
          f'sil_mean:{metrics_means[5]} '
          f'f1_mean:{metrics_means[3]} ', end=' ')

    if time_std is not None: print(f'time_std:{time_std} ', end='')
    print(f'loss_std:{metrics_stds[6]} '
          f'acc_std:{metrics_stds[0]} '
          f'ari_std:{metrics_stds[2]} '
          f'nmi_std:{metrics_stds[1]} '
          f'f1_std:{metrics_stds[3]} '
          f'db_std:{metrics_stds[4]} '
          f'sil_std:{metrics_stds[5]} ')
