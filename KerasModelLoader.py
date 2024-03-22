########################################################
#                     gremh97                          #
########################################################
import importlib

def keras_model_loader(model_name):
    """
    Return the Keras model instance and its corresponding preproc_fn for the given model name.

    Args:
    model_name (str): The name of the Keras model.

    Returns:
    the Keras model instance and its corresponding preproc_fn. If the mode name is not found in the dictionary, return None.
    """

    KerasModels = {
        """
        {model_name in lowercase:[keras.applications Module, model Fuction, modelparams(dict), preprocessing function], ...}
        """
        "efficientnetb0": ["efficient", "EfficientNetB0", {}, lambda x: x],
        "efficientnetb1": ["efficient", "EfficientNetB1", {}, lambda x: x],
        "efficientnetb2": ["efficient", "EfficientNetB2", {}, lambda x: x],
        "efficientnetb3": ["efficient", "EfficientNetB3", {}, lambda x: x],
        "efficientnetb4": ["efficient", "EfficientNetB4", {}, lambda x: x],
        "efficientnetb5": ["efficient", "EfficientNetB5", {}, lambda x: x],
        "efficientnetb6": ["efficient", "EfficientNetB6", {}, lambda x: x],
        "efficientnetb7": ["efficient", "EfficientNetB7", {}, lambda x: x],
        "regnetx002": ["regnet", "RegNetX002", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx004": ["regnet", "RegNetX004", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx006": ["regnet", "RegNetX006", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx008": ["regnet", "RegNetX008", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx016": ["regnet", "RegNetX016", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx032": ["regnet", "RegNetX032", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx040": ["regnet", "RegNetX040", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx064": ["regnet", "RegNetX064", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx080": ["regnet", "RegNetX080", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx120": ["regnet", "RegNetX120", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx160": ["regnet", "RegNetX160", {"include_preprocessing": False}, lambda x: x/255],
        "regnetx320": ["regnet", "RegNetX320", {"include_preprocessing": False}, lambda x: x/255],
        "regnety002": ["regnet", "RegNetY002", {"include_preprocessing": False}, lambda x: x/255],
        "regnety004": ["regnet", "RegNetY004", {"include_preprocessing": False}, lambda x: x/255],
        "regnety006": ["regnet", "RegNetY006", {"include_preprocessing": False}, lambda x: x/255],
        "regnety008": ["regnet", "RegNetY008", {"include_preprocessing": False}, lambda x: x/255],
        "regnety016": ["regnet", "RegNetY016", {"include_preprocessing": False}, lambda x: x/255],
        "regnety032": ["regnet", "RegNetY032", {"include_preprocessing": False}, lambda x: x/255],
        "regnety040": ["regnet", "RegNetY040", {"include_preprocessing": False}, lambda x: x/255],
        "regnety064": ["regnet", "RegNetY064", {"include_preprocessing": False}, lambda x: x/255],
        "regnety080": ["regnet", "RegNetY080", {"include_preprocessing": False}, lambda x: x/255],
        "regnety120": ["regnet", "RegNetY120", {"include_preprocessing": False}, lambda x: x/255],
        "regnety160": ["regnet", "RegNetY160", {"include_preprocessing": False}, lambda x: x/255],
        "regnety320": ["regnet", "RegNetY320", {"include_preprocessing": False}, lambda x: x/255],
        "inceptionresnetv2": ["inception_resnet_v2", "InceptionResNetV2", {}, lambda x: x/127.5-1],
        "inceptionv3": ["inception_v3", "InceptionV3", {}, lambda x: x/127.5-1],
        "resnet50v2":   ["resnet_v2", "ResNet50V2",  {},  lambda x: x/127.5-1],
        "resnet101v2":  ["resnet_v2", "ResNet101V2", {},  lambda x: x/127.5-1],
        "resnet152v2":  ["resnet_v2", "ResNet152V2", {},  lambda x: x/127.5-1],
        "nasnetmobile": ["nasnet", "NASNetMobile", {}, lambda x: x/127.5-1],
        "nasnetlarge":  ["nasnet", "NASNetLarge" , {}, lambda x: x/127.5-1],
        "xception": ["xception", "Xception", {}, lambda x: x/127.5-1],
        "mobilenet": ["mobilenet", "MobileNet", {}, lambda x: x/127.5-1],
        "mobilenetv2": ["mobilenet_v2", "MobileNetV2", {}, lambda x: x/127.5-1],
        "mobilenetv3small": ["", "MobileNetV3Small", {"input_shape": (224,224,3), "include_preprocessing": False}, lambda x: x/127.5-1],
        "mobilenetv3large": ["", "MobileNetV3Large", {"input_shape": (224,224,3), "include_preprocessing": False}, lambda x: x/127.5-1],
        "efficientnetv2s": ["efficientnet_v2", "EfficientNetV2S", {"include_preprocessing": False}, lambda x: x/127.5-1],
        "efficientnetv2m": ["efficientnet_v2", "EfficientNetV2M", {"include_preprocessing": False}, lambda x: x/127.5-1],
        "efficientnetv2l": ["efficientnet_v2", "EfficientNetV2L", {"include_preprocessing": False}, lambda x: x/127.5-1],
        "densenet121": ["densenet", "DenseNet121", {}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "densenet169": ["densenet", "DenseNet169", {}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "densenet201": ["densenet", "DenseNet201", {}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "efficientnetv2b0": ["efficientnet_v2", "EfficientNetV2B0", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "efficientnetv2b1": ["efficientnet_v2", "EfficientNetV2B1", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "efficientnetv2b2": ["efficientnet_v2", "EfficientNetV2B2", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "efficientnetv2b3": ["efficientnet_v2", "EfficientNetV2B3", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs50":  ["resnet_rs", "ResNetRS50", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs101": ["resnet_rs", "ResNetRS101", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs152": ["resnet_rs", "ResNetRS152", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs200": ["resnet_rs", "ResNetRS200", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs270": ["resnet_rs", "ResNetRS270", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs350": ["resnet_rs", "ResNetRS350", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "resnetrs420": ["resnet_rs", "ResNetRS420", {"include_preprocessing": False}, lambda x: (x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]],
        "convnexttiny":     ["convnext", "ConvNeXtTiny",   {"include_preprocessing": False}, lambda x: ((x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]) * 255],
        "convnextsmall":    ["convnext", "ConvNeXtSmall",  {"include_preprocessing": False}, lambda x: ((x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]) * 255],
        "convnextbase":     ["convnext", "ConvNeXtBase",   {"include_preprocessing": False}, lambda x: ((x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]) * 255],
        "convnextlarge":    ["convnext", "ConvNeXtLarge",  {"include_preprocessing": False}, lambda x: ((x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]) * 255],
        "convnextxlarge":   ["convnext", "ConvNeXtXLarge", {"include_preprocessing": False}, lambda x: ((x/255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]) * 255],
        "vgg16": ["vgg16", "VGG16", {}, lambda x: x[...,::-1] - [103.939, 116.779, 123.68]],
        "vgg19": ["vgg19", "VGG19", {}, lambda x: x[...,::-1] - [103.939, 116.779, 123.68]],
        "resnet50":  ["resnet", "ResNet50", {}, lambda x: x[...,::-1] - [103.939, 116.779, 123.68]],
        "resnet101": ["resnet", "ResNet101", {}, lambda x: x[...,::-1] - [103.939, 116.779, 123.68]],
        "resnet152": ["resnet", "ResNet152", {}, lambda x: x[...,::-1] - [103.939, 116.779, 123.68]]
    }        

    try:
        keras_model_info        = KerasModels[model_name.lower()]
        frompkg                 = f"tensorflow.keras.applications.{keras_model_info[0]}"
        kerasApplicationsModule = importlib.import_module(frompkg)
        modelFuction            = getattr(kerasApplicationsModule, keras_model_info[1])
        model                   = modelFuction(** keras_model_info[2])
        return model, keras_model_info[3]        # return model and preproc_fn

    except KeyError:
        print(f"Model '{model_name}' not found in the keras.applications.")
        return None, None


if __name__ == '__main__':
    model_name = "mobilenetv2"

    # Load model
    model, preproc_fn   = keras_model_loader(model_name)
