# BoTNet

BoTNet architecture from
     "Searching for BoTNet" <https://arxiv.org/abs/2101.11605>.

For the Pytorch implementation, you can refer to [EfficientNet.pytorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## Run

1. generate BoTNet.wts from pytorch implementation
For example,

```
python gen_wts.py 

```
a file 'BoTNet.wts' will be generated

2. build and run
```
cd /TensorRT-BoTNet
mkdir build
cd build
cmake ..
make
sudo ./botnet -s  // serialize model to plan file i.e. 'botnet.engine'
sudo ./botnet -d  // deserialize plan file and run inference
```

3. see if the output is same as pytorch side
