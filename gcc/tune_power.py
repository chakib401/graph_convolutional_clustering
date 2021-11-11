from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


import tensorflow as tf
import numpy as np
from gcc.metrics import output_metrics, print_metrics
from gcc.optimizer import optimize
from gcc.utils import read_dataset, is_close, preprocess_dataset

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Parameters
flags.DEFINE_string('dataset', 'cora', 'Name of the graph dataset (cora, citeseer, pubmed or wiki).')
flags.DEFINE_integer('min_power', 1, 'Smallest propagation order to test.')
flags.DEFINE_integer('max_power', 150, 'Largest propagation order to test.')
flags.DEFINE_integer('runs', 20, 'Number of runs per power.')
flags.DEFINE_integer('n_clusters', 0, 'Number of clusters (0 for ground truth).')
flags.DEFINE_integer('max_iter', 30, 'Number of iterations of the algorithm.')
flags.DEFINE_float('tol', 10e-7, 'Tolerance threshold of convergence.')

dataset = flags.FLAGS.dataset
min_power = flags.FLAGS.min_power
max_power = flags.FLAGS.max_power
runs = flags.FLAGS.runs
n_clusters = flags.FLAGS.n_clusters
max_iter = flags.FLAGS.max_iter
tolerance = flags.FLAGS.tol

# Read the dataset
adj, features, labels, n_classes = read_dataset(dataset)
if n_clusters == 0: n_clusters = n_classes
# Process the dataset
tf_idf = (dataset == 'cora' or dataset == 'citeseer') # normalize binary word datasets
norm_adj, features = preprocess_dataset(adj, features, tf_idf=tf_idf)

# compute min_power matrix
for power in range(1, min_power):
    features = norm_adj @ features

# apply the algorithm from min_power to max_power matrices
global_metrics_means = []
global_metrics_stds = []
value_1 = np.nan

for power in range(min_power, max_power + 1):
    print(f"== power {power} ==")

    features = norm_adj @ features
    run_metrics = []

    for run in range(runs):
        G, F, W, losses = optimize(features, n_clusters, n_clusters,
                                   max_iter=max_iter, tolerance=tolerance)
        metrics = output_metrics(features @ W, labels, G)
        run_metrics.append(metrics + [losses[-1] if max_iter > 0 else 0])

    global_metrics_means.append(np.mean(run_metrics, 0))
    global_metrics_stds.append(np.std(run_metrics, 0))

    print_metrics(global_metrics_means[-1], global_metrics_stds[-1])

    value = global_metrics_means[-1][-1]
    if is_close(value, value_1, features.shape[1] / features.shape[0]):
        print(f'best power: {power-1}')
        break
    value_1 = value
