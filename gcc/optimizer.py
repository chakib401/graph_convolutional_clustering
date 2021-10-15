import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def update_rule_F(XW, G, k):
    F = tf.math.unsorted_segment_mean(XW, G, k)
    return F


def update_rule_W(X, F, G):
    _, U, V = tf.linalg.svd(tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
    W = U @ tf.transpose(V)
    return W


def update_rule_G(XW, F):
    centroids_expanded = F[:, None, ...]
    distances = tf.reduce_mean(tf.math.squared_difference(XW, centroids_expanded), 2)
    G = tf.math.argmin(distances, 0, output_type=tf.dtypes.int32)
    return G


def init_G_F(XW, k):
    km = KMeans(k).fit(XW)
    G = km.labels_
    F = km.cluster_centers_
    return G, F


def init_W(X, f):
    pca = PCA(f, svd_solver='randomized').fit(X)
    W = pca.components_.T
    return W


@tf.function
def train_loop(X, F, G, W, k, max_iter, tolerance):
    losses = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    prev_loss = tf.float64.max

    for i in tf.range(max_iter):

        W = update_rule_W(X, F, G)
        XW = X @ W
        G = update_rule_G(XW, F)
        F = update_rule_F(XW, G, k)

        loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
        if prev_loss - loss < tolerance:
            break

        losses = losses.write(i, loss)
        prev_loss = loss

    return G, F, W, losses.stack()


def optimize(X, k, f, max_iter=30, tolerance=10e-7):
    # init G and F
    W = init_W(X, f)
    G, F = init_G_F(X @ W, k)
    G, F, W, loss_history = train_loop(X, F, G, W, k, max_iter, tolerance)

    return G, F, W, loss_history
