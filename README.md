# LigMSANet
## Introduction
Scale variation and real-time counting are challenging problems for crowd counting in highly congested scenes. To remedy these issues, we proposed a Lightweight Multi-Scale Adaptive Network (LigMSANet). There are two strong points in our method. First, the scale limitation is broken and the proportion of neurons with different receptive field sizes are adjusted spontaneously according to input images through a novel multi-scale adaptation module (MSAM). Second, the model performance is significantly improved at a little cost of parameter by replacing the standard convolution with the depthwise separable convolution and a tailored MobileNetV2 with 5 bottleneck blocks (here, the step size of the fourth bottleneck block is 1). To demonstrate the effectiveness of the proposed method, we conduct extensive experiments on three major crowd counting datasets (ShanghaiTech, UCF_CC_50 and UCSD) and our method achieves superior performance to state-of-the-art methods while with much less parameters and runtimes.
## Installation
1 Environment<br>
Env: Python 3.7; keras 2.3.1; CUDA 10.1; <br>
Install some packages

## Resoult on ShanghaiTech Part B

Method            | MAE  | MSE | Params (MB)  |  Runtime | fps | FLOPs 
 ----             | -----| ------|   ---- | ----- | ------  | ------ 
 LigMSANet(ours)  | 10.9 | 17.5 | 0.63| 44| 22.7| 1.3
