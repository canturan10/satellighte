<p align="right">
    <a href="https://www.buymeacoffee.com/canturan10"><img src="https://img.buymeacoffee.com/button-api/?text=You can buy me a coffee&emoji=&slug=canturan10&button_colour=5F7FFF&font_colour=ffffff&font_family=Comic&outline_colour=000000&coffee_colour=FFDD00" width="200" /></a>
</p>

<!-- PROJECT SUMMARY -->
<p align="center">
    <img width="100px" src="https://raw.githubusercontent.com/canturan10/satellighte/master/src/satellighte.png" align="center" alt="Satellighte" />
<h2 align="center">Satellighte</h2>
<h4 align="center">Satellite Image Classification</h4>

<p align="center">
    <strong>
        <a href="https://canturan10.github.io/satellighte/">Website</a>
        •
        <a href="https://satellighte.readthedocs.io/">Docs</a>
        •
        <a href="https://share.streamlit.io/canturan10/satellighte-streamlit/app.py">Demo</a>
    </strong>
</p>

<!-- TABLE OF CONTENTS -->
<details>
    <summary>
        <strong>
            TABLE OF CONTENTS
        </strong>
    </summary>
    <ol>
        <li>
            <a href="#about-the-satellighte">About The Satellighte</a>
        </li>
        <li>
            <a href="##prerequisites">Prerequisites</a>
        </li>
        <li>
            <a href="#installation">Installation</a>
            <ul>
                <li><a href="#from-pypi">From Pypi</a></li>
                <li><a href="#from-source">From Source</a></li>
            </ul>
        </li>
        <li><a href="#usage-examples">Usage Examples</a></li>
        <li><a href="#architectures">Architectures</a></li>
        <li><a href="#datasets">Datasets</a></li>
        <li><a href="#deployments">Deployments</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#tests">Tests</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#contributors">Contributors</a></li>
        <li><a href="#contact">Contact</a></li>
        <li><a href="#license">License</a></li>
        <li><a href="#references">References</a></li>
        <li><a href="#citations">Citations</a></li>
    </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Satellighte

**Satellighte** is an image classification library  that consist state-of-the-art deep learning methods. It is a combination of the words **'Satellite'** and **'Light'**, and its purpose is to establish a light structure to classify satellite images, but to obtain robust results.

> **Satellite image classification** is the most significant technique used in remote sensing for the computerized study and pattern recognition of satellite information, which is based on diversity structures of the image that involve rigorous validation of the training samples depending on the used classification algorithm.
>
> _Source: [paperswithcode](https://paperswithcode.com/task/satellite-image-classification)_

<!-- PREREQUISITES -->
## Prerequisites

Before you begin, ensure you have met the following requirements:

| requirement       | version  |
| ----------------- | -------- |
| imageio           | ~=2.15.0 |
| numpy             | ~=1.21.0 |
| pytorch_lightning | ~=1.6.0  |
| scikit-learn      | ~=1.0.2  |
| torch             | ~=1.8.1  |

<!-- INSTALLATION -->
## Installation

To install Satellighte, follow these steps:

### From Pypi

```bash
pip install satellighte
```

### From Source

```bash
git clone https://github.com/canturan10/satellighte.git
cd satellighte
pip install .
```

#### From Source For Development

```bash
git clone https://github.com/canturan10/satellighte.git
cd satellighte
pip install -e ".[all]"
```
<!-- USAGE EXAMPLES -->
## Usage Examples

```python
import imageio
import satellighte as sat

img = imageio.imread("test.jpg")

model = sat.Classifier.from_pretrained("model_config_dataset")
model.eval()

results = model.predict(img)
# [{'cls1': 0.55, 'cls2': 0.45}]
```

<!-- _For more examples, please refer to the [Documentation](https://github.com/canturan10/readme-template)_ -->

<!-- ARCHITECTURES -->
## Architectures

- [x] [MobileNetV2](https://github.com/canturan10/satellighte/blob/master/satellighte/archs/README.md#mobilenetv2)
- [x] [EfficientNet](https://github.com/canturan10/satellighte/blob/master/satellighte/archs/README.md#efficientnet)
- [x] [ResNet](https://github.com/canturan10/satellighte/blob/master/satellighte/archs/README.md#resnet)
- [ ] [CoAtNet](https://github.com/canturan10/satellighte/blob/master/satellighte/archs/README.md)

_For more information, please refer to the [Architectures](https://github.com/canturan10/satellighte/blob/master/satellighte/archs)_

<!-- DATASETS -->
## Datasets

- [x] [EuroSAT](https://github.com/canturan10/satellighte/blob/master/satellighte/datasets/README.md#EuroSAT)
- [x] [RESISC45](https://github.com/canturan10/satellighte/blob/master/satellighte/datasets/README.md)

_For more information, please refer to the [Datasets](https://github.com/canturan10/satellighte/blob/master/satellighte/datasets)_

<!-- DEPLOYMENTS -->
## Deployments

- [x] [FastAPI](https://github.com/canturan10/satellighte/blob/master/deployment/README.md#fastapi)
- [x] [ONNX](https://github.com/canturan10/satellighte/blob/master/deployment/README.md#onnx)
- [x] [DeepSparse](https://github.com/canturan10/satellighte/blob/master/deployment/README.md#deepsparse)
- [x] [TensorFlow](https://github.com/canturan10/satellighte/blob/master/deployment/README.md#tensorflow)
- [x] [TensorFlow Lite](https://github.com/canturan10/satellighte/blob/master/deployment/README.md#tensorflow-lite)

_For more information, please refer to the [Deployment](https://github.com/canturan10/satellighte/blob/master/deployment)_

<!-- TRAINING -->
## Training

To training, follow these steps:

For installing Satellighte, please refer to the [Installation](#installation).

```bash
python training/eurosat_training.py
```

For optional arguments,

```bash
python training/eurosat_training.py --help
```

<!-- TESTS -->
## Tests

During development, you might like to have tests run.

Install dependencies

```bash
pip install -e ".[test]"
```

### Linting Tests

```bash
pytest satellighte --pylint --pylint-error-types=EF
```

### Document Tests

```bash
pytest satellighte --doctest-modules
```

### Coverage Tests

```bash
pytest --doctest-modules --cov satellighte --cov-report term
```

<!-- CONTRIBUTING -->
## Contributing

To contribute to `Satellighte`, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin`
5. Create the pull request.

Alternatively see the `GitHub` documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

<!-- CONTRIBUTORS -->
## Contributors

<table style="width:100%">
    <tr>
        <td align="center">
            <a href="https://github.com/canturan10">
                <h3>
                    Oğuzcan Turan
                </h3>
                <img src="https://avatars0.githubusercontent.com/u/34894012?s=460&u=722268bba03389384f9d673d3920abacf12a6ea6&v=4&s=200"
                    width="200px;" alt="Oğuzcan Turan" /><br>
                <a href="https://www.linkedin.com/in/canturan10/">
                    <img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=Linkedin&logoColor=white"
                        width="75px;" alt="Linkedin" />
                </a>
                <a href="https://canturan10.github.io/">
                    <img src="https://img.shields.io/badge/-Portfolio-lightgrey?style=flat&logo=opera&logoColor=white"
                        width="75px;" alt="Portfolio" />
                </a>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/canturan10">
                <h3>
                    You ?
                </h3>
                <img src="https://raw.githubusercontent.com/canturan10/readme-template/master/src/you.png"
                    width="200px;" alt="Oğuzcan Turan" /><br>
                <a href="#">
                    <img src="https://img.shields.io/badge/-Reserved%20Place-red?style=flat&logoColor=white"
                        width="110px;" alt="Reserved" />
                </a>
            </a>
        </td>
    </tr>
</table>

<!-- CONTACT -->
## Contact

If you want to contact me you can reach me at [can.turan.10@gmail.com](mailto:can.turan.10@gmail.com).

<!-- LICENSE -->
## License

This project is licensed under `MIT` license. See [`LICENSE`](LICENSE) for more information.

<!-- REFERENCES -->
## References

The references used in the development of the project are as follows.

- [Img Shields](https://shields.io)
- [GitHub Pages](https://pages.github.com)
- [FastFace](https://github.com/borhanMorphy/fastface)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [Torchvision](https://github.com/pytorch/vision)

<!-- CITATIONS -->
## Citations

```bibtex
@article{DBLP:journals/corr/ChengHL17,
  author    = {Gong Cheng and
               Junwei Han and
               Xiaoqiang Lu},
  title     = {Remote Sensing Image Scene Classification: Benchmark and State of
               the Art},
  journal   = {CoRR},
  volume    = {abs/1703.00121},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.00121},
  eprinttype = {arXiv},
  eprint    = {1703.00121},
  timestamp = {Mon, 02 Dec 2019 09:32:19 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/ChengHL17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{helber2019eurosat,
  title={Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019},
  publisher={IEEE}
}
```

```bibtex
@inproceedings{helber2018introducing,
  title={Introducing EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  booktitle={IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium},
  pages={204--207},
  year={2018},
  organization={IEEE}
}
```

```bibtex
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

Give a ⭐️ if this project helped you!
![-----------------------------------------------------](https://raw.githubusercontent.com/canturan10/readme-template/master/src/colored_4b.png)

_This readme file is made using the [readme-template](https://github.com/canturan10/readme-template)_
