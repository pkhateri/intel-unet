# github repo
https://github.com/IntelAI/unet

# data from
http://medicaldecathlon.com/ -> get data -> https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2 -> Task01_BrainTumour.tar -> un-tarr -> save in Documents/data/decathlon

# install
virtualenv --python=python3.7 venv3.7
go inside venv:
pip install tensorflow==2.2.0 tqdm psutil nibabel
pip install visualkeras # parisa added: for visualization
pip install protobuf==3.20.* # parisa added: to avoid error
pip install -U tensorboard_plugin_profile # for profiling

# install on vm-manager & with gpu
- vm-manager has python 3.8 only:
virtualenv --python=python3.8 venv3.8
- go inside venv:
- I had to change tensorflow version to a higher version so that it is compatible with the current cuda and cuDNN
pip install tensorflow-gpu tqdm psutil nibabel
pip install visualkeras # parisa added: for visualization
pip install protobuf # parisa added: to avoid error

# install on miac using singularity
## build the container
- go to the singularity recipe folder
- singularity build --fakeroot intel-unet-prerequisites-22.06-tf2-py3.sif intel-unet-prerequisites-22.06-tf2-py3.def
## run the container
- singularity run --nv --bind /projects /projects/parisa/git_software/intel-unet/singularity/intel-unet-prerequisites-20.08-tf2-py3.sif
## ready to go


# jupyter notebook in venv
go inside the venv: source venv3.7/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv3.7
jupyter notebook -> kernel -> choose kernel

# input data
- in the setting.py file choose the input image type (2D/3D).
- The 3D supports decathlon input format.
- The 2D supports either png input files (zoltan data) or json files (progstar). In any case, the png or json files should be divided in 3 folders: train/val/test, and each image is represented by one png/json file.
- in case of 2D png inputs, red channel represents the image and the green channel represents the label.
- in case of 2D json, the json file contains the path to the image and label files.

# run the training
`cd 2D`
`mkdir output/output_dir`
modify setting.py for input paramaters
`python train.py`

# run the inference (test)
`python plot_tf_inference_examples.py`

# run the tensorboard
tensorboard --logdir 2D/output/output_dir/keras_tensorboard_transposed/unet_block0_inter4_intra4/
- specify the port if run on remote:
tensorboard --logdir 2D/output/output_dir/keras_tensorboard_transposed/unet_block0_inter4_intra4/ --port 6006

# Genreate json files for progstar data:
```
python3 create_json_files_for_progstar_dataset.py -c train.csv -o /projec/parisa/data/progstar/intel_unet/train
```
