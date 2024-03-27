import os
import tempfile
import pytest
import tensorflow as tf
from quantizer.Keras2TFLiteQuantizer import Keras2TFLiteQuantizer
from quantizer.KerasModelLoader import load_keras_model

# Create a temporary directory for saving test TFLite models
TEST_SAVE_DIR = tempfile.mkdtemp()
os.environ["TEST_SAVE_DIR"] = TEST_SAVE_DIR
print(f"Temporary test directory created at {TEST_SAVE_DIR}")

@pytest.fixture(scope="module")
def quantizer():
    """Fixture to initialize a Keras2TFLiteQuantizer instance for testing."""
    model, preproc_fn = load_keras_model("mobilenetv2")
    quantizer_instance = Keras2TFLiteQuantizer(
        model=model,
        model_name="test_mobilenetv2",
        preproc_fn=preproc_fn,
        saved_dir=TEST_SAVE_DIR
    )
    return quantizer_instance
