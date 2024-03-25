import pathlib
import tensorflow as tf

class InvalidQuantizationError(Exception):
    def __init__(self, message="Invalid quantization configuration"):
        self.message = message
        super().__init__(self.message)


class Keras2TFLiteQuantizer:
    def __init__(self, model, model_name=None, preproc_fn=None, saved_dir=None, data_dir="/data/image/imagenet/val"):
        self.model          = model
        self.model_name     = model_name
        self.preproc_fn     = preproc_fn
        self.saved_dir      = saved_dir
        self.data_dir       = data_dir
        self.tflite_model   = None


    def tfl_no_quant(self):
        """     Convert Keras model to TFLite model w/o quantization        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        self.tflite_model = converter.convert()
        # Save quantized TFLite model
        if self.saved_dir: 
            tflite_model_file = self.save_tflite_model("fp32")
            return (self.tflite_model, tflite_model_file)
        return (self.tflite_model)


    def tfl_dynamic_range_quant(self, weight_type="int8"):
        """Convert Keras model to TFLite model w/ quantizing the weight to float16 or int8. 

        Args:
        weight_type (str): Type of quantization for model weights. Supported values are "float16" and "int8". Default is "int8".

        Returns:
        tuple: A tuple containing the quantized TFLite model content and the file path if `saved_dir` is provided. Otherwise, returns the TFLite model content.
        """
        # Supported `weight_type = ("int8", "float16")
        supported_weight_type = ("int8", "float16", "fp16")
        if weight_type.lower() not in supported_weight_type:
            raise NotImplementedError(f"Unknown weight_type='{weight_type}' is provided\n Choose one of supported weight_type={supported_weight_type}")
        # Convert Keras model to TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if weight_type in supported_weight_type[1:]: 
            converter.target_spec.supported_types = [tf.float16]
        self.tflite_model = converter.convert()
        # Save quantized TFLite model
        if self.saved_dir: 
            tflite_model_file = self.save_tflite_model(f"dynamic_{weight_type}")
            return (self.tflite_model, tflite_model_file)
        return (self.tflite_model)

    
    def tfl_dynamic_int8_quant(self):
        """         Convert Keras model to TFLite model w/ quantizing the weight to int8        """       
        # redundant to `tfl_dynamic_range_quant(weight_type="int8")`
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.tflite_model = converter.convert()
        
        # Save quantized TFLite model
        if self.saved_dir: 
            tflite_model_file = self.save_tflite_model("dynamic_int8")
            return (self.tflite_model, tflite_model_file)
        return (self.tflite_model)

    
    def tfl_dynamic_float16_quant(self):
        """          Convert Keras model to TFLite model w/ quantizing the weight to float16      """ 
        # redundant to `tfl_dynamic_range_quant(weight_type="float16")`
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        self.tflite_model = converter.convert()
        
        # Save quantized TFLite model
        if self.saved_dir: 
            tflite_model_file = self.save_tflite_model("dynamic_fp16")
            return (self.tflite_model, tflite_model_file)
        return (self.tflite_model)



    def tfl_full_integer_quant(self, signal_type="int8", weight_type="int8", if_input_type="float32", if_output_type="float32"):
        """Convert the Keras model to a fully quantized TFLite model with integer quantization.

        Args:
        signal_type (str): Type of quantization for model signals(activations). Supported values are "int8" and "int16". Default is "int8".
        weight_type (str): Type of quantization for model weights. Supported value is "int8". Default is "int8".
        if_input_type (str): Type of input tensor in inference. Supported values are "float32", "int8", and "uint8". Default is "float32".
        if_output_type (str): Type of output tensor in inference. Supported values are "float32", "int8", and "uint8". Default is "float32".

        Returns:
        tuple: A tuple containing the quantized TFLite model content and the file path if `saved_dir` is provided. Otherwise, returns the TFLite model content.
        """
        # Check if if_input_type and if_output_type are valid
        supported_inference_type = ("float32", "fp32", "int8", "uint8")
        if  if_input_type not in supported_inference_type or if_output_type not in supported_inference_type:
            raise InvalidQuantizationError(f"Invalid inference types: {if_input_type}, {if_output_type} \n Supported inference in/output type: (float32, int8, uint8)")
        
        # Check if signal_type and weight_type are valid
        valid_quant_types = [("int8", "int8"), ("int16", "int8")]
        if (signal_type, weight_type) not in valid_quant_types:
            raise InvalidQuantizationError(f"Invalid quantization types: {signal_type}, {weight_type} \n Supported combination of (signal_type, weight_type): {{valid_quant_types}}")
        
        # Convert Keras model to TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if (signal_type, weight_type) == ("int16", "int8"):
            # Integer only: 16-bit activations with 8-bit weights (experimental)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
        else:
            # Integer only: 8-bit activations with 8-bit weights
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = self._representative_data_gen
        converter.inference_input_type = self._get_tf_data_type(if_input_type)
        converter.inference_output_type = self._get_tf_data_type(if_output_type)
        self.tflite_model = converter.convert()

        # Save quantized TFLite model
        if self.saved_dir: 
            tflite_f_name = f"full_int_{signal_type}x{weight_type}"
            if if_input_type in supported_inference_type[2:] or if_output_type in supported_inference_type[2:]:
                tflite_f_name += "if8"
            
            tflite_model_file = self.save_tflite_model(tflite_f_name)
            return (self.tflite_model, tflite_model_file)
        return (self.tflite_model)


    def _representative_data_gen(self):
        """Generate representative data for quantization.

        This method generates representative data for quantizing the TFLite model.
        It yields preprocessed images from the specified data directory.

        Returns:
            generator: A generator that yields preprocessed images.
        """
        dataset = tf.keras.utils.image_dataset_from_directory(
                self.data_dir,
                label_mode=None,
                batch_size=1,
                image_size=(224, 224),
                shuffle=True,
                crop_to_aspect_ratio=True
            )
        for images in dataset.take(100):
            yield[self.preproc_fn(images)]


    def _get_tf_data_type(self, type):
        if type.lower() in ("float32" , "fp32"):
            return tf.float32
        elif type.lower() in ("float16" , "fp16"):
            return tf.float16
        elif type.lower() == "int16":
            return tf.int16
        elif type.lower() == "int8":
            return tf.int8
        elif type.lower() == "uint8":
            return tf.uint8
        else:
            raise NotImplementedError(f"Unknown data_type='{type}' is provided\n Choose one of supported type:(float32, float16, int16, int8, uint8)")


    def save_tflite_model(self, model_info):
        tflite_models_dir = pathlib.Path(self.saved_dir)
        tflite_models_dir.mkdir(exist_ok=True, parents=True)

        if self.model_name:
            tflite_model_file = f"{tflite_models_dir}/{self.model_name}_{model_info}.tflite"
        else:
            tflite_model_file = f"{tflite_models_dir}/{model_info}.tflite"

        with open(tflite_model_file, 'wb') as f:
            f.write(self.tflite_model)

        return tflite_model_file

  


if __name__ =="__main__":
    import KerasModelLoader

    saved_dir           = "./tf_quantization/tflite_models"

    # Load keras model
    model_name          = "mobilenetv2"
    model, preproc_fn   = KerasModelLoader.load_keras_model(model_name)

    # Make TFLite model
    tfl_quantizer           = Keras2TFLiteQuantizer(model=model, model_name=model_name, preproc_fn=preproc_fn, saved_dir=saved_dir)
    tfl_model               = tfl_quantizer.tfl_no_quant()
    tfl_dynamic_int8        = tfl_quantizer.tfl_dynamic_range_quant(weight_type="int8")
    tfl_dynamic_fp16        = tfl_quantizer.tfl_dynamic_range_quant(weight_type="fp16")
    tfl_full_int_8x8        = tfl_quantizer.tfl_full_integer_quant(signal_type="int8", weight_type="int8", if_input_type="float32", if_output_type="float32")
    tfl_full_int_16x8       = tfl_quantizer.tfl_full_integer_quant(signal_type="int16", weight_type="int8", if_input_type="float32", if_output_type="float32")
    tfl_full_int_8x8_if8    = tfl_quantizer.tfl_full_integer_quant(signal_type="int8", weight_type="int8", if_input_type="int8", if_output_type="int8")

    # Verify TFlite model
    interpreter         = tf.lite.Interpreter(model_content=tfl_full_int_8x8_if8[0])
    print(f"quantization: {interpreter.get_input_details()[0]['quantization']}")
    input_details       = interpreter.get_input_details()[0]["dtype"]
    print('input: ', input_details)
    output_details      = interpreter.get_output_details()[0]["dtype"]
    print('output: ', output_details)
