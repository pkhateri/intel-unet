#
#

"""
This module creates a TensorFlow/Keras model
from model.py and then saves its visualization into a file.
"""
import os
import tensorflow as tf 
from dataloader_oct_png import DatasetGenerator, get_oct_filelist

#import numpy as np

from argparser_visualization import args

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# If hyperthreading is NOT enabled, then use
#os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

os.environ["KMP_BLOCKTIME"] = str(args.blocktime)

os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["INTRA_THREADS"] = str(args.num_threads)
os.environ["INTER_THREADS"] = str(args.num_inter_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime

def test_intel_tensorflow():
    """
    Check if Intel version of TensorFlow is installed
    """
    import tensorflow as tf

    print("We are using Tensorflow version {}".format(tf.__version__))

    major_version = int(tf.__version__.split(".")[0])
    # comment these lines out, as they are not compatible for newer versions of tensorflow.
    #if major_version >= 2:
    #    from tensorflow.python import _pywrap_util_port
    #    print("Intel-optimizations (DNNL) enabled:",
    #          _pywrap_util_port.IsMklEnabled())
    #else:
    #    print("Intel-optimizations (DNNL) enabled:",
    #          tf.pywrap_tensorflow.IsMklEnabled())

if __name__ == "__main__":

    print("Runtime arguments = {}".format(args))
    test_intel_tensorflow() # Print if we are using Intel-optimized TensorFlow

    """
    Load data, create a model, and visualize it.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the OCT png image directory to a TensorFlow data loader ...")
    print("-" * 30)

    trainFiles, validateFiles, testFiles = get_oct_filelist(data_path=args.data_path, seed=args.seed, split=args.split)

    ds_train = DatasetGenerator(trainFiles, batch_size=args.batch_size, crop_dim=[args.crop_dim,args.crop_dim], augment=True, seed=args.seed)
    ds_validation = DatasetGenerator(validateFiles, batch_size=args.batch_size, crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)
    ds_test = DatasetGenerator(testFiles, batch_size=args.batch_size, crop_dim=[args.crop_dim,args.crop_dim], augment=False, seed=args.seed)

    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)


    """
    Step 2: Define the model
    """
    if args.use_pconv:
        from model_pconv import unet
    else:
        from model import unet

    unet_model = unet(channels_first=args.channels_first,
                 fms=args.featuremaps,
                 output_path=args.output_path,
                 inference_filename=args.inference_filename,
                 model_name=args.model_name,
                 optimizer_name=args.optimizer_name,
                 learning_rate=args.learningrate,
                 weight_dice_loss=args.weight_dice_loss,
                 use_upsampling=args.use_upsampling,
                 use_dropout=args.use_dropout,
                 print_model=args.print_model)

    model = unet_model.create_model(
        ds_train.get_input_shape(), ds_train.get_output_shape(), model_name=args.model_name)

    model_filename, model_callbacks = unet_model.get_callbacks()

    """
    Step 3: visualize the model
    """
    import visualkeras
    from collections import defaultdict
    from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D, concatenate, SpatialDropout2D, Conv2DTranspose
    from PIL import ImageFont
    
    font = ImageFont.truetype(args.font_file, args.font_size) 
    color_map = defaultdict(dict)
    color_map[Input]['fill'] = args.input_color
    color_map[Dense]['fill'] = args.dense_color
    color_map[Conv2D]['fill'] = args.conv2d_color
    color_map[MaxPooling2D]['fill'] = args.maxpooling2d_color
    color_map[SpatialDropout2D]['fill'] = args.spatialdropout2d_color
    color_map[Conv2DTranspose]['fill'] = args.conv2dtranspose_color
    color_map[concatenate]['fill'] = args.concatenate_color
    color_map[ZeroPadding2D]['fill'] = args.zeropadding2d_color
    color_map[Dropout]['fill'] = args.dropout_color
    color_map[Flatten]['fill'] = args.flatten_color

    visual_filename = os.path.join(args.output_path,'model_visualization.png')
    visualkeras.layered_view(model, color_map=color_map, legend=True, font=font, draw_volume=True, scale_xy=0.8, scale_z=0.1, to_file=visual_filename)


