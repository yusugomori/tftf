import tensorflow as tf


def categorical_crossentropy(y, t):
    loss = \
        tf.reduce_mean(-tf.reduce_sum(
                       t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                       axis=list(range(1, len(y.get_shape())))))

    return loss
