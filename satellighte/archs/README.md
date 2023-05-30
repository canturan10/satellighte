[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coatnet-marrying-convolution-and-attention/image-classification-on-gashissdb)](https://paperswithcode.com/sota/image-classification-on-gashissdb?p=coatnet-marrying-convolution-and-attention) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/coatnet-marrying-convolution-and-attention/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=coatnet-marrying-convolution-and-attention) 

# Architectures  <!-- omit in toc -->

- [CoAtNet](#coatnet)
- [EfficientNet](#efficientnet)
- [MobileNetV2](#mobilenetv2)
- [ResNet](#resnet)
- [Citation](#citation)

## CoAtNet

Transformers have attracted increasing interests in computer vision, but they still fall behind state-of-the-art convolutional networks. In this work, we show that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, we present **CoAtNets**(pronounced "coat" nets), a family of hybrid models built from two key insights: (1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency.

| Architecture | Configuration | Parameters | Model Size |
| :----------: | :-----------: | :--------: | :--------: |
| **CoAtNet**  |       0       |   17.8 M   |   71 MB    |
| **CoAtNet**  |       1       |   33.2 M   |   132 MB   |
| **CoAtNet**  |       2       |   55.8 M   |   223 MB   |
| **CoAtNet**  |       3       |  117.0 M   |   470 MB   |
| **CoAtNet**  |       4       |  203.0 M   |   815 MB   |

## EfficientNet

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2), in addition to squeeze-and-excitation blocks.

|   Architecture   | Configuration | Parameters | Model Size |
| :--------------: | :-----------: | :--------: | :--------: |
| **EfficientNet** |      b0       |   4.1 M    |   16 MB    |
| **EfficientNet** |      b1       |   6.6 M    |   26 MB    |
| **EfficientNet** |      b2       |   7.8 M    |   30 MB    |
| **EfficientNet** |      b3       |   10.8 M   |   42 MB    |
| **EfficientNet** |      b4       |   17.6 M   |   70 MB    |
| **EfficientNet** |      b5       |   28.4 M   |   113 MB   |
| **EfficientNet** |      b6       |   40.8 M   |   163 MB   |
| **EfficientNet** |      b7       |   63.8 M   |   225 MB   |
| **EfficientNet** |     v2-s      |   20.2 M   |   80 MB    |
| **EfficientNet** |     v2-m      |   52.9 M   |   211 MB   |
| **EfficientNet** |     v2-l      |   117 M    |   468 MB   |

## MobileNetV2

**MobileNetV2** is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an [inverted residual structure](https://paperswithcode.com/method/inverted-residual-block) where the residual connections are between the bottleneck layers.  The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

|  Architecture   | Configuration | Parameters | Model Size |
| :-------------: | :-----------: | :--------: | :--------: |
| **MobileNetV2** |    default    |   2.3 M    |    9 MB    |

## ResNet

**Residual Networks**, or **ResNets**, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping. They stack residual blocks ontop of each other to form network: e.g. a ResNet-50 has fifty layers using these blocks.

| Architecture | Configuration | Parameters | Model Size |
| :----------: | :-----------: | :--------: | :--------: |
|  **ResNet**  |      18       |   14.0 M   |   56 MB    |
|  **ResNet**  |      34       |   45.9 M   |   183 MB   |
|  **ResNet**  |      50       |   23.6 M   |   94 MB    |
|  **ResNet**  |      101      |   42.6 M   |   170 MB   |
|  **ResNet**  |      152      |   58.2 M   |   232 MB   |

## Citation

```BibTeX
@misc{dai2021coatnet,
      title={CoAtNet: Marrying Convolution and Attention for All Data Sizes},
      author={Zihang Dai and Hanxiao Liu and Quoc V. Le and Mingxing Tan},
      year={2021},
      eprint={2106.04803},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```BibTeX
@article{DBLP:journals/corr/abs-1801-04381,
  author    = {Mark Sandler and
               Andrew G. Howard and
               Menglong Zhu and
               Andrey Zhmoginov and
               Liang{-}Chieh Chen},
  title     = {Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification,
               Detection and Segmentation},
  journal   = {CoRR},
  volume    = {abs/1801.04381},
  year      = {2018},
  url       = {http://arxiv.org/abs/1801.04381},
  archivePrefix = {arXiv},
  eprint    = {1801.04381},
  timestamp = {Tue, 12 Jan 2021 15:30:06 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1801-04381.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```BibTeX
@article{DBLP:journals/corr/abs-1905-11946,
  author    = {Mingxing Tan and
               Quoc V. Le},
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  journal   = {CoRR},
  volume    = {abs/1905.11946},
  year      = {2019},
  url       = {http://arxiv.org/abs/1905.11946},
  eprinttype = {arXiv},
  eprint    = {1905.11946},
  timestamp = {Mon, 03 Jun 2019 13:42:33 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1905-11946.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```BibTeX
@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  eprinttype = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
