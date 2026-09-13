import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

__all__ = [
    'GNNModel',
]


def l2_regularizer(scale):
    """L2 regularizer compatible with TF1-style ref variables.

    tf.keras.regularizers.l2 chokes on the float32_ref dtype of TF1 variables,
    so implement the penalty directly on the graph.
    """
    def regularizer(x):
        return scale * tf.nn.l2_loss(x)

    return regularizer


def bias_add(inputs, name):
    """Add a learnable bias to the inputs (replacement for slim.layers.bias_add).

    Creates a zero-initialized bias variable scoped under `name` and adds it to
    every row of `inputs`.
    """
    output_dim = int(inputs.get_shape()[1])
    bias = tf.get_variable(f"{name}-bias", [output_dim], dtype=tf.float32,
                           initializer=tf.zeros_initializer())
    return tf.nn.bias_add(inputs, bias)


class GNNModel(object):
    """Base class for all Graph Neural Network (GNN) models."""

    def __init__(self, features, graph_adj, targets):
        """Create a model.

        Parameters
        ----------
        graph_adj : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        features : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr]
            Attribute matrix in CSR or numpy format.
        targets : np.ndarray, shape [num_nodes, num_classes]
            One-hot matrix of node labels.
        """
        self.targets = targets
        self.graph_adj = self._preprocess_adj(graph_adj)
        self.features = self._preprocess_features(features)

    def _inference(self):
        """
        Builds the inference graph of the model.

        Returns
        -------
        logits : tf.Tensor, shape [num_nodes, num_classes]
            The logits produced by the model (before feeding into softmax).
        """
        raise NotImplementedError

    def _predict(self):
        """
        Computes predictions of the model on the targets given in the constructor.

        Returns
        -------
        predictions : tf.Tensor, shape [num_nodes, num_classes]
            Softmax probabilities for each node and each class.
        """
        with tf.name_scope('predict'):
            return tf.nn.softmax(self.inference)

    def _loss(self):
        """
        Computes the cross-entropy plus regularization loss of the model on the targets given in the constructor.

        Returns
        -------
        loss : tf.Tensor, shape [], dtype tf.float32
            A Tensor that, if evaluated, yields the model's loss on the targets given in the constructor.
        """
        with tf.name_scope('loss'):
            output = self.inference
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=self.targets)
            )
            regularization_losses = tf.losses.get_regularization_losses()
            if not regularization_losses:
                return loss
            return loss + tf.add_n(regularization_losses)

    def _build_model_graphs(self):
        """
        Builds the graph portions for inference, prediction and loss computation and adds them as fields to the class.
        Call this method as the last statement of your __init__.
        """
        self.inference = self._inference()
        self.predict = self._predict()
        self.loss = self._loss()

    def _preprocess_features(self, features):
        """
        Preprocessing function for the features. Called by the constructor. Even if no preprocessing is needed, the
        features might need to be converted to a tf.SparseTensor in this method using the function
        util.to_sparse_tensor.

        Returns
        -------
        features_tensor : tf.Tensor or tf.SparseTensor
            The features as a (sparse) tensor.
        """
        raise NotImplementedError

    def _preprocess_adj(self, graph_adj):
        """
        Preprocessing function for the adjacency matrix. Called by the constructor. Even if no preprocessing is needed,
        the adjacency matrix might need to be converted to a tf.SparseTensor in this method using the function
        util.to_sparse_tensor.

        Returns
        -------
        graph_adj_tensor : tf.Tensor or tf.SparseTensor
            The adjacency matrix as a (sparse) tensor.
        """
        raise NotImplementedError

    def optimize(self, learning_rate, global_step):
        """
        Defines the optimizing operation for the model.

        Parameters
        ----------
        learning_rate : tf.Tensor, shape [], dtype tf.float32 or scalar
            The initial learning rate for the optimizer.
        global_step : tf.Variable, shape [], dtype tf.int32
            The global step of the training process. Will be incremented by the optimizer.

        Returns
        -------
        train_step : tf.Tensor
            The optimiziation operation for one train step.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        return optimizer.minimize(self.loss, global_step=global_step)
