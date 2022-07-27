import os
from typing import List, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import extract_archive

from ..core import _download_file_from_url
from .base import BaseDataset, Identity


class RESISC45(BaseDataset):
    """
    RESISC45 dataset is a dataset for Remote Sensing Image Scene Classification (RESISC). It contains 31,500 RGB images of size 256×256 divided into 45 scene classes, each class containing 700 images.
    """

    __phases__ = (
        "train",
        "test",
        "val",
    )

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
        self.transforms = Identity() if transforms is None else transforms
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

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=1,
            stratify=y_train,
        )  # 0.25 x 0.8 = 0.2

        if phase == "train":
            return x_train.tolist(), y_train.tolist()
        elif phase == "test":
            return x_test.tolist(), y_test.tolist()
        elif phase == "val":
            return x_val.tolist(), y_val.tolist()
        else:
            raise ValueError("Unknown phase")

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
        _download_file_from_url(
            "https://drive.google.com/u/0/uc?id=1PCesRqeXYINcsulnTixVjR15xFNXropZ&export=download&confirm=t",
            os.path.join(self.root_dir, "resisc45.zip"),
        )
        extract_archive(
            os.path.join(self.root_dir, "resisc45.zip"),
            self.root_dir,
            remove_finished=True,
        )


if __name__ == "__main__":
    data = RESISC45("satellighte/datas/resisc45")
    print(data[0])
    print(data.classes)
    print(len(data.classes))
