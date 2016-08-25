# Is Faster R-CNN Doing Well for Pedestrian Detection?

By Liliang Zhang, Liang Lin, Xiaodan Liang, Kaiming He

### Introduction

This code is relative to an [arXiv tech report](https://arxiv.org/abs/1607.07032), which is also accepted on ECCV 2016.

The RPN code in this repo is written based on the MATLAB re-implementation of Faster R-CNN. Details about Faster R-CNN are in: [ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn).

This BF code in this repo is written based on Piotr's Image & Video Matlab Toolbox. Details about Piotr's Toolbox are in: [pdollar/toolbox](https://github.com/pdollar/toolbox).

This code has been tested on Ubuntu 14.04 with MATLAB 2014b.

### Citing Faster R-CNN

If you find this repo useful in your research, please consider citing:

    @article{zhang2016faster,
      title={Is Faster R-CNN Doing Well for Pedestrian Detection?},
      author={Zhang, Liliang and Lin, Liang and Liang, Xiaodan and He, Kaiming},
      journal={arXiv preprint arXiv:1607.07032},
      year={2016}
    }

### Requirements

0. `Caffe` build for RPN+BF (see [here](https://github.com/zhangliliang/caffe/tree/RPN_BF))
    - If the mex in 'external/caffe/matlab/caffe_faster_rcnn' could not run under your system, please follow the [instructions](https://github.com/zhangliliang/caffe/tree/RPN_BF) on our Caffe branch to compile and replace the mex.

0. MATLAB

0. GPU: Titan X, K40c, etc.

### Testing Demo

0. Download `VGG16\_caltech\_final.zip` from [BaiduYun](https://pan.baidu.com/s/1miNdKZe), and unzip it in the repo folder.

0. Start MATLAB from the repo folder.

0. Run `faster_rcnn_build.m`

0. Run `experiments/script_rpn_bf_pedestrian_VGG16_caltech_demo.m` to see the detection results on some images collected in Internet.




