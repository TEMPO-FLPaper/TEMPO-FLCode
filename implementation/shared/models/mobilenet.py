"""Reference-derived models/mobilenet. See README.md."""

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

class Operations:
    # Reference cells (zero-based): {'tempo': 19, 'ditto': 17, 'hfl': 17, 'flat': 17}
    def build_tempo_model(self, num_classes):
        inputs = keras.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3), name="image")
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(self.IMG_SIZE, self.IMG_SIZE, 3),
            alpha=self.CONFIG["mobilenet_alpha"],
            include_top=False,
            weights="imagenet",
        )
        backbone.trainable = True
        for layer in backbone.layers[: -self.CONFIG["trainable_tail_layers"]]:
            layer.trainable = False
        x = backbone(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.Dropout(self.CONFIG["dropout"], name="dropout")(x)
        logits = layers.Dense(num_classes, activation=None, name="logits")(x)
        return keras.Model(inputs, logits, name="TEMPO_MobileNetV2_V12_2")

    # Reference cells (zero-based): {'tempo': 19, 'ditto': 17, 'hfl': 17, 'flat': 17}
    def compile_for_fit(self, model, learning_rate):
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate, clipnorm=self.CONFIG["gradient_clip_norm"]
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
