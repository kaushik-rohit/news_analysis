import tensorflow as tf


def doc2vec_network():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(100)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
        tf.keras.layers.Dense(21, activation='softmax')
    ])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0,
                                                      name='categorical_crossentropy')

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model


def lda_network():
    pass
