# run:
to avoid the error `The NVIDIA Driver was not detected` run with --nv flag:
```
singularity run --nv ./my_file.sif
```

# different containers:
1. build from ubuntu: `intel_unet_ubuntu.def`
    - it works but this error occurs: `failed with error CUPTI could not be loaded or symbol could not be found`
    - the /usr/local/cuda path is not available
    - speed: 10epoch/31min Total time elapsed for program = 0:31:21.534654 seconds
2. build from cuda: `intel_unet_cuda.def`
    - it works but this error occurs: `failed with error CUPTI could not be loaded or symbol could not be found`
    - the /usr/local/cuda path is available
    - speed: 10epoch/31 Total time elapsed for program = 0:31:17.326035 seconds
3. build from tesorflow 20.08: `intel_unet_tensorflow_20_08.def`
    - works but gives this error at the start of container:
    ```
        WARNING: Detected NVIDIA NVIDIA RTX A6000 GPU, which is not yet supported in this version of the container
        ERROR: No supported GPU(s) detected to run this container
    ```
    - speed: 11epoch/64min Total time elapsed for program = 1:04:55.346415 seconds
4. build from tensorflow 22.06: `intel_unet_tensorflow_22_06.def`
    - fails with the following errors:
```
E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape in2DUNet_Brats_Decathlon/spatial_dropout2d/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:425] Loaded runtime CuDNN library: 8.4.1 but source was compiled with: 8.6.0.  CuDNN library needs to have matching major version and equal or higher minor version. If using a binary install, upgrade your CuDNN library.  If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.
Traceback (most recent call last):
  File "train.py", line 150, in <module>
    model.fit(ds_train,
  File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/pkhateri/.local/lib/python3.8/site-packages/tensorflow/python/eager/execute.py", line 53, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.UnimplementedError: Graph execution error:

Detected at node '2DUNet_Brats_Decathlon/encodeAa/Relu' defined at (most recent call last):
    File "train.py", line 150, in <module>
      model.fit(ds_train,
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 1742, in fit
      tmp_logs = self.train_function(iterator)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 1338, in train_function
      return step_function(self, iterator)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 1322, in step_function
      outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 1303, in run_step
      outputs = model.train_step(data)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 1080, in train_step
      y_pred = self(x, training=True)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/training.py", line 569, in __call__
      return super().__call__(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/base_layer.py", line 1150, in __call__
      outputs = call_fn(inputs, *args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 96, in error_handler
      return fn(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/functional.py", line 512, in call
      return self._run_internal_graph(inputs, training=training, mask=mask)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/functional.py", line 669, in _run_internal_graph
      outputs = node.layer(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 65, in error_handler
      return fn(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/engine/base_layer.py", line 1150, in __call__
      outputs = call_fn(inputs, *args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/utils/traceback_utils.py", line 96, in error_handler
      return fn(*args, **kwargs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/layers/convolutional/base_conv.py", line 321, in call
      return self.activation(outputs)
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/activations.py", line 321, in relu
      return backend.relu(
    File "/home/pkhateri/.local/lib/python3.8/site-packages/keras/src/backend.py", line 5397, in relu
      x = tf.nn.relu(x)
Node: '2DUNet_Brats_Decathlon/encodeAa/Relu'
DNN library is not found.
	 [[{{node 2DUNet_Brats_Decathlon/encodeAa/Relu}}]] [Op:__inference_train_function_5508]
```

