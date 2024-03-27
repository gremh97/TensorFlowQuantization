# Keras2TFLiteQuantizer
`Keras2TFLiteQuantizer` is a utility to convert Keras models to the TFLite format, with support for various quantization options. This class helps convert models like MobileNetV2 into lightweight versions using formats like `fp32` (floating point 32-bit), `int8`, `float16`, and more.

## Features
- Converts Keras models to TFLite format
- Includes a utility function `load_keras_model` to load pre-configured Keras models with preprocessing functions
- Supports multiple quantization options:
    - Dynamic Range Quantization (`int8`, `float16`)
    - Full Integer Quantization (`int8`, `int16`)
    - No Quantization (FP32)
- Offers customizable settings for file saving and data type conversions, with exception handling for invalid configurations

## Directory Structure
```bash
TensorFlowQuantization/
├── quantizer/
│   ├── __init__.py
│   ├── Keras2TFLiteQuantizer.py        # Keras2TFLiteQuantizer class definition
│   └── KerasModelLoader.py             # KerasModel and it's preprocess function
├── tests/
│   ├── __init__.py
│   └── test_quantizer.py               # Test file
└── requirements.txt                    # Dependency file
└── README.md                           # Project description file
```

## Installation
1. Install required libraries: Install tensorflow and other dependencies.

```bash
pip install -r requirements.txt
```
2. Organize directory structure: Place the file containing the Keras2TFLiteQuantizer class in `src/`, and the test files in `tests/.

## Usage Example
```python
import tensorflow as tf
from src.your_module_name import Keras2TFLiteQuantizer

# Load a Keras model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3))
preproc_fn = tf.keras.applications.mobilenet_v2.preprocess_input

# Instantiate the Keras2TFLiteQuantizer
quantizer = Keras2TFLiteQuantizer(
    model=model, 
    model_name="mobilenetv2", 
    preproc_fn=preproc_fn, 
    saved_dir="./tflite_models"
)

# Generate TFLite models with various quantization options
tflite_fp32 = quantizer.tfl_no_quant()
tflite_dynamic_int8 = quantizer.tfl_dynamic_range_quant(weight_type="int8")
tflite_full_int8 = quantizer.tfl_full_integer_quant(signal_type="int8", weight_type="int8")
```
### Key Methods
- `tfl_no_quant`: Converts a Keras model to TFLite format without quantization
- `tfl_dynamic_range_quant(weight_type="int8")`: Applies dynamic range quantization to `int8` or `float16`
- `tfl_full_integer_quant`: Applies full integer quantization
    - Parameters:
        - `signal_type`: Type of quantization for activations (int8 or int16)
        - `weight_type`: Type of quantization for weights (int8)
        - `if_input_type`: Input tensor type (float32, int8, uint8)
        - `if_output_type`: Output tensor type (float32, int8, uint8)

## Exception Handling
- `InvalidQuantizationError`: Raised for invalid quantization configurations or unsupported data types
- `NotImplementedError`: Raised when unsupported weight_type or data types are specified


## Testing
Run tests using `pytest` from the project root directory:

```bash
pytest
```
This command executes all tests in the `tests/` directory to verify that the class methods work as expected.