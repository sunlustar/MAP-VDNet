# MAP-VDNet

This repository contains the Pytorch implementation for the paper "**Deep Maximum a Posterior Estimator for Video Denoising**" (IJCV 2021).

[[paper]](https://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/IJCV_VD_final.pdf)
[[project]](https://see.xidian.edu.cn/faculty/wsdong/Projects/MAP-VDNet.htm).


## Environment

Python 3.6.5

Pytorch 1.2.0

CUDA

Matlab

## Usage

### Installation

0. Clone this repository: ```git clone https://github.com/sunlustar/MAP-VDNet.git```

1. Download the training dataset: [Vimeo-90K](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).

2. Generate simulated noisy frames by running **./make_dataset/add_noise_main.m**

3. Download the [pre-trained denoising models](https://drive.google.com/drive/folders/1r-FZ1eZ1H8v5k8UsFFa4snd8wTDd0_YL), then unzip this file and place the models into **./mynetwork/**.

4. Download the test datasets [ASU and DTMC-HD](https://drive.google.com/drive/folders/1bOmulFTlGozqb_49ocBDKzblHA4LpSKN), then unzip these files and place the test data into **./testsets/**.

### Training

1. Pretrain the alignment network: ```python train_pretrain.py```

2. Train the video denoising network: ```python train.py```

### Testing

1. Test on ASU dataset: ```python evaluate_ASU.py```

2. Test on DTMC-HD dataset: ```python evaluate_DTMC-HD.py```

3. Test on Vimeo-90K dataset: ```python evaluate.py```

### Citation
If you find our work useful for your research, please consider citing the following papers

```
@article{sun2021deep,
  title={Deep maximum a posterior estimator for video denoising},
  author={Sun, Lu and Dong, Weisheng and Li, Xin and Wu, Jinjian and Li, Leida and Shi, Guangming},
  journal={International Journal of Computer Vision},
  volume={129},
  pages={2827--2845},
  year={2021},
  publisher={Springer}
}
```


### Contact
If you have any questions, please send an email to sunlustar@163.com.

