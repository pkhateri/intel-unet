# pull
pulled the container images from nvcr docker hub:
```
singularity pull tensorflow-19.12-tf2-py3.sif docker://nvcr.io/nvidia/tensorflow:19.12-tf2-py3
```
first pulled 22.04 but it was not compatible with the driver on miac workstation.

# run:
to avoid the error `The NVIDIA Driver was not detected` run with --nv flag:
```
singularity run --nv ./tensorflow-19.12-tf2-py3.sif
```
