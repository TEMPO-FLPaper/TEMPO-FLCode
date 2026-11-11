"""Keras 3 / TensorFlow 2.16 serialization bridge; quantization stays in the reference exporter."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid
import tensorflow as tf
from tensorflow import keras


@contextmanager
def converter_factory(model, directory):
    """Yield a converter constructor for the same inference graph and weights."""
    if tf.__version__.startswith("2.16.") and keras.__version__.startswith("3."):
        # This runtime's direct converter requests the removed Keras _get_save_spec API.
        staging = Path(directory).resolve() / (".convert-" + uuid.uuid4().hex)
        # Inherit the selected output directory's Windows ACLs instead of
        # tempfile's restrictive mode, which some managed hosts cannot traverse.
        staging.mkdir()
        try:
            model.export(str(staging))
            yield lambda: tf.lite.TFLiteConverter.from_saved_model(str(staging))
        finally:
            if staging.resolve().parent != Path(directory).resolve():
                raise ValueError("Invalid converter staging directory.")
            shutil.rmtree(staging)
    else:
        yield lambda: tf.lite.TFLiteConverter.from_keras_model(model)
