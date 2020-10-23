import tensorflow as tf


def doc2vec_network(enc, input_dim=100):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_dim)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
        tf.keras.layers.Dense(len(enc.categories_[0]), activation='softmax')
    ])
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0,
                                                      name='categorical_crossentropy')

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model
