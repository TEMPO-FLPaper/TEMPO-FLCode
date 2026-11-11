"""Reference-derived data/pipeline. See README.md."""

import numpy as np

import tensorflow as tf

class Operations:
    # Reference cells (zero-based): {'tempo': 18, 'ditto': 16, 'hfl': 16, 'flat': 16}
    def decode_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE])
        image = tf.cast(image, tf.float32)
        return (image, label)

    # Reference cells (zero-based): {'tempo': 18, 'ditto': 16, 'hfl': 16, 'flat': 16}
    def preprocess_mobilenet(self, image, label):
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return (image, label)

    # Reference cells (zero-based): {'tempo': 18, 'ditto': 16, 'hfl': 16, 'flat': 16}
    def make_tf_dataset(self, df, batch_size=32, training=False, shuffle=True):
        paths = df["path"].astype(str).values
        labels = df["label_id"].astype(np.int32).values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(
                max(1, len(df)), seed=self.CONFIG["seed"], reshuffle_each_iteration=True
            )
        ds = ds.map(self.decode_image, num_parallel_calls=self.DATA_PARALLEL_CALLS)
        if training:
            ds = ds.map(
                lambda x, y: (self.augmenter(x, training=True), y),
                num_parallel_calls=self.DATA_PARALLEL_CALLS,
            )
        ds = ds.map(self.preprocess_mobilenet, num_parallel_calls=self.DATA_PARALLEL_CALLS)
        ds = ds.batch(int(batch_size))
        ds = ds.prefetch(self.DATA_PREFETCH)
        options = tf.data.Options()
        options.threading.private_threadpool_size = int(
            self.CONFIG["tf_data_private_threadpool_size"]
        )
        options.threading.max_intra_op_parallelism = 1
        options.experimental_deterministic = True
        ds = ds.with_options(options)
        return ds

    # Reference cells (zero-based): {'tempo': 18, 'ditto': 16, 'hfl': 16, 'flat': 16}
    def load_batch_tensor(self, df, training=False):
        images = []
        labels = []
        for _, row in df.iterrows():
            image = tf.io.read_file(str(row["path"]))
            image = tf.io.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, [self.IMG_SIZE, self.IMG_SIZE])
            image = tf.cast(image, tf.float32)
            if training:
                image = self.augmenter(image, training=True)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            images.append(image)
            labels.append(int(row["label_id"]))
        if not images:
            return (
                tf.zeros((0, self.IMG_SIZE, self.IMG_SIZE, 3), dtype=tf.float32),
                tf.zeros((0,), dtype=tf.int32),
            )
        return (tf.stack(images), tf.convert_to_tensor(labels, dtype=tf.int32))
