# TransparentDepth
Bachelor Thesis: Monocular Depth Map Estimation of Transparent Structures

## Getting started

```shell
conda create -n MDE python=3.7
conda activate MDE
```

```shell
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
pip install termcolor oyaml attrdict h5py timm tqdm matplotlib tensorboardX Imath
pip install git+https://github.com/aleju/imgaug
pip install git+https://github.com/jamesbowman/openexrpython.git
```

For Adabin:

```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```


## Run project

training:

```shell
python train.py -c config/config.yaml
```

Evaluation:

```shell
python train.py -c config/config.yaml
```


## Reference and Credits

//TODO

ClearGrasp: 
https://github.com/Shreeyak/cleargrasp

DenseDepth:
https://github.com/ialhashim/DenseDepth
