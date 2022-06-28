# Architectures  <!-- omit in toc -->

- [MobileNetV2](#mobilenetv2)
- [EfficientNet](#efficientnet)
- [Citation](#citation)

## MobileNetV2

**MobileNetV2** is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an [inverted residual structure](https://paperswithcode.com/method/inverted-residual-block) where the residual connections are between the bottleneck layers.  The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

|  Architecture   | Configuration | Parameters | Model Size |
| :-------------: | :-----------: | :--------: | :--------: |
| **MobileNetV2** |    default    |   2.2 M    |  8.947 MB  |

## EfficientNet

**EfficientNet** is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of [MobileNetV2](https://paperswithcode.com/method/mobilenetv2), in addition to squeeze-and-excitation blocks.

|  Architecture   | Configuration | Parameters | Model Size |
| :-------------: | :-----------: | :--------: | :--------: |
| **EfficientNet** |    B0    |   4.0 M    |  16.081 MB  |

## Citation

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
