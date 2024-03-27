import pytest
from quantizer.KerasModelLoader import load_keras_model
from quantizer.Keras2TFLiteQuantizer import Keras2TFLiteQuantizer, InvalidQuantizationError

@pytest.fixture
def quantizer():
    # Load model and preprocessing function
    model, preproc_fn = load_keras_model("mobilenetv2")
    return Keras2TFLiteQuantizer(model=model, model_name="mobilenetv2", preproc_fn=preproc_fn, saved_dir="./test_dir")

def test_no_quant(quantizer):
    quantized_model = quantizer.tfl_no_quant()
    assert quantized_model is not None
    assert isinstance(quantized_model[0], bytes)

def test_dynamic_int8_quant(quantizer):
    quantized_model = quantizer.tfl_dynamic_range_quant(weight_type="int8")
    assert quantized_model is not None
    assert isinstance(quantized_model[0], bytes)

def test_full_integer_quant_invalid_inference_type(quantizer):
    with pytest.raises(InvalidQuantizationError):
        quantizer.tfl_full_integer_quant(if_input_type="invalid_type", if_output_type="float32")

def test_full_integer_quant(quantizer):
    quantized_model = quantizer.tfl_full_integer_quant(signal_type="int8", weight_type="int8", if_input_type="float32", if_output_type="float32")
    assert quantized_model is not None
    assert isinstance(quantized_model[0], bytes)

def test_model_loader():
    model, preproc_fn = load_keras_model("mobilenetv2")
    assert model is not None
    assert callable(preproc_fn)
