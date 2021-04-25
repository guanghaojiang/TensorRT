# EfficientNet-B0

EfficientNet-B0 architecture from
     "Searching for EfficientNet" <https://arxiv.org/pdf/1905.11946.pdf>.

For the Pytorch implementation, you can refer to [EfficientNet.pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## Run

1. generate EfficientNet-B0.wts from pytorch implementation 
     For example,

```
python gen_wts.py 
```
a file 'EfficientNet-B0.wts' will be generated

2. build and run
```
cd /TensorRT-EfficientNet-B0
mkdir build
cd build
cmake ..
make
sudo ./EfficinetNetB0 -s  // serialize model to plan file i.e. 'EfficientNetB0.engine'
sudo ./EfficinetNetB0 -d  // deserialize plan file and run inference
```

3. see if the output is same as pytorch side
