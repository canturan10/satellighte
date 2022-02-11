import os
from typing import List, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive

from .base import BaseDataset


class EuroSAT(BaseDataset):
    """
    Eurosat is a dataset and deep learning benchmark for land use and land cover classification. The dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.
    """

    __phases__ = ("val", "train")

    def __init__(
        self,
        root_dir: str = None,
        phase: str = "val",
        transforms=None,
        **kwargs,
    ):
        if root_dir is None:
            root_dir = "TODO"

        self.root_dir = root_dir
        self.phase = phase
        self.download()

        ids, targets = self._split_dataset(phase)
        super().__init__(ids, targets, transforms=transforms, **kwargs)

    @property
    def num_classes(self) -> int:
        return len(self.__classes)

    @property
    def classes(self) -> List[str]:
        return self.__classes

    def name_to_id(self, name: str) -> int:
        return self.__classes.index(name)

    def id_to_name(self, idx: int) -> str:
        return self.__classes[idx]

    def _split_dataset(self, phase: str) -> Tuple:
        labels = []
        filenames = []

        data_dir = os.path.join(self.root_dir)

        for item in os.listdir(data_dir):
            f = os.path.join(data_dir, item)
            if os.path.isfile(f):
                continue
            for subitem in os.listdir(f):
                sub_f = os.path.join(f, subitem)
                filenames.append(sub_f)
                labels.append(item)

        filenames = np.asarray(filenames)
        labels = np.asarray(labels)

        labels = labels[filenames.argsort()]
        filenames = filenames[filenames.argsort()]

        # convert to integer labels
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(np.sort(np.unique(labels)))
        labels = label_encoder.transform(labels)
        labels = np.asarray(labels)

        # remember label encoding
        self.__classes = list(label_encoder.classes_)

        # split into a is_train and test set as provided data is not presplit
        x_train, x_test, y_train, y_test = train_test_split(
            filenames,
            labels,
            test_size=0.2,
            random_state=1,
            stratify=labels,
        )
        if phase == "train":
            return x_train.tolist(), y_train.tolist()
        else:
            return x_test.tolist(), y_test.tolist()

    def _check_exists(self) -> bool:
        """
        Check the Root directory is exists
        """
        return os.path.exists(self.root_dir)

    def download(self) -> None:
        """
        Download the dataset from the internet
        """

        if self._check_exists():
            return

        os.makedirs(self.root_dir, exist_ok=True)
        download_and_extract_archive(
            "https://drive.google.com/file/d/1_QZVrVVmybvY8_pJjCdnev9THLvS3739",
            download_root=self.root_dir,
            extract_root=self.root_dir,
            filename="eurosat.zip",
        )
        os.remove(os.path.join(self.root_dir, "eurosat.zip"))


if __name__ == "__main__":
    data = EuroSAT("satellighte/datas/eurosat")
    print(data[0])
