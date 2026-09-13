import json
import sqlite3

import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
import yaml

tf.disable_v2_behavior()


def to_sparse_tensor(M, value=False):
    """Convert a scipy sparse matrix to a tf SparseTensor or SparseTensorValue.

    Parameters
    ----------
    M : scipy.sparse.sparse
        Matrix in Scipy sparse format.
    value : bool, default False
        Convert to tf.SparseTensorValue if True, else to tf.SparseTensor.

    Returns
    -------
    S : tf.SparseTensor or tf.SparseTensorValue
        Matrix as a sparse tensor.

    Author: Oleksandr Shchur
    """
    M = sp.coo_matrix(M)
    if value:
        return tf.SparseTensorValue(np.vstack((M.row, M.col)).T, M.data, M.shape)
    else:
        return tf.SparseTensor(np.vstack((M.row, M.col)).T, M.data, M.shape)


def dropout_supporting_sparse_tensors(X, keep_prob):
    """Add dropout layer on top of X.

    Parameters
    ----------
    X : tf.Tensor or tf.SparseTensor
        Tensor over which dropout is applied.
    keep_prob : float, tf.placeholder
        Probability of keeping a value (= 1 - probability of dropout).

    Returns
    -------
    X : tf.Tensor or tf.SparseTensor
        Tensor with elementwise dropout applied.

    Author: Oleksandr Shchur & Johannes Klicpera
    """
    if isinstance(X, tf.SparseTensor):
        values_after_dropout = tf.nn.dropout(X.values, rate=1 - keep_prob)
        return tf.SparseTensor(X.indices, values_after_dropout, X.dense_shape)
    else:
        return tf.nn.dropout(X, rate=1 - keep_prob)


def scatter_add_tensor(tensor, indices, out_shape, name=None):
    """
    Code taken from https://github.com/tensorflow/tensorflow/issues/2358 and adapted.

    Adds up elements in tensor that have the same value in indices.

    Must have shape(tensor)[0] == shape(indices)[0].
    :param tensor: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16,
        int16, int8, complex64, complex128, qint8, quint8, qint32, half.
    :param indices: 1-D tensor of indices.
    :param out_shape: The shape of the output tensor. Must have out_shape[1] == shape(tensor)[1].
    :param name: A name for the operation (optional).
    :return: Tensor with same datatype as tensor and shape out_shape.
    """
    with tf.name_scope(name, 'scatter_add_tensor') as scope:
        indices = tf.expand_dims(indices, -1)
        # the scatter_nd function adds up values for duplicate indices what is exactly what we want
        return tf.scatter_nd(indices, tensor, out_shape, name=scope)

def uniform_float(random_state, lower, upper, number, log_scale=False):
    """Author: Oleksandr Shchur"""
    if log_scale:
        lower = np.log(lower)
        upper = np.log(upper)
        logit = random_state.uniform(lower, upper, number)
        return np.exp(logit)
    else:
        return random_state.uniform(lower, upper, number)


def uniform_int(random_state, lower, upper, number, log_scale=False):
    """Author: Oleksandr Shchur"""
    if not isinstance(lower, int):
        raise ValueError("lower must be of type 'int', got {0} instead"
                         .format(type(lower)))
    if not isinstance(upper, int):
        raise ValueError("upper must be of type 'int', got {0} instead"
                         .format(type(upper)))
    if log_scale:
        lower = np.log(lower)
        upper = np.log(upper)
        logit = random_state.uniform(lower, upper, number)
        return np.exp(logit).astype(np.int32)
    else:
        return random_state.randint(int(lower), int(upper), number)


def generate_random_parameter_settings(search_spaces_dict, num_experiments, seed):
    if seed is not None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState()

    settings = {}
    for param in search_spaces_dict:
        if search_spaces_dict[param]["format"] == "values":
            settings[param] = random_state.choice(search_spaces_dict[param]["values"], size=num_experiments)

        elif search_spaces_dict[param]["format"] == "range":
            if search_spaces_dict[param]["dtype"] == "int":
                gen_func = uniform_int
            else:
                gen_func = uniform_float
            settings[param] = gen_func(random_state,
                                       lower=search_spaces_dict[param]["min"],
                                       upper=search_spaces_dict[param]["max"],
                                       number=num_experiments,
                                       log_scale=search_spaces_dict[param]["log_scale"])

        else:
            raise ValueError(f"Unknown format {search_spaces_dict[param]['format']}.")

    settings = {key: settings[key].tolist() for key in settings}  # convert to python datatypes since MongoDB cannot
    # serialize numpy datatypes
    return settings


def get_mongo_config(config_path):
    with open(config_path, 'r') as conf:
        config = yaml.safe_load(conf)
    return config['db_path']


def get_experiment_config(config_path):
    with open(config_path, 'r') as conf:
        return yaml.safe_load(conf)


def get_pending_connection(db_path):
    """Open a connection to the SQLite database holding the pending-jobs queue.

    The `pending` table is created automatically if it does not exist yet.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g. from the experiment config's `db_path` entry).

    Returns
    -------
    conn : sqlite3.Connection
        Connection to the database. The caller is responsible for closing it.
    """
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            running INTEGER NOT NULL DEFAULT 0,
            config TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def fetch_pending_job(conn):
    """Atomically claim the next pending job.

    Uses BEGIN IMMEDIATE so that only one worker can claim a job at a time,
    mirroring the atomic find_one_and_update of the previous MongoDB backend.

    Parameters
    ----------
    conn : sqlite3.Connection
        Connection returned by get_pending_connection.

    Returns
    -------
    job : dict or None
        The claimed job record {"id": ..., "config": {...}}, or None if the queue is empty.
    """
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute("SELECT id, config FROM pending WHERE running = 0 ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        cur.execute("UPDATE pending SET running = 1 WHERE id = ?", (row["id"],))
        conn.commit()
        return {"id": row["id"], "config": json.loads(row["config"])}
    except Exception:
        conn.rollback()
        raise


def is_binary_bag_of_words(features):
    features_coo = features.tocoo()
    return all(single_entry == 1.0 for _, _, single_entry in zip(features_coo.row, features_coo.col, features_coo.data))


def get_num_trainable_weights():
    variables = tf.trainable_variables()
    return sum(np.prod(variable.get_shape()) for variable in variables)
