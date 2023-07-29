from __future__ import annotations

import sys

sys.path.append("..")

import numpy as np
import torch
import os
import cv2
from typing import Tuple, Union, List
from common import Rectangle, ImageLabelLoader, RawImageProcessor, ModelImageProcessor
import json
from albumentations import Compose


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[Rectangle],
        model_processor: type[ModelImageProcessor],
    ):
        super().__init__()
        self.images = images
        self.labels = labels
        self.y = [label.get_center() for label in self.labels]
        self.model_processor = model_processor

    @classmethod
    def from_folder(
        cls,
        img_dir: str,
        labels_dir: str,
        raw_processor: type[RawImageProcessor],
        model_processor: type[ModelImageProcessor],
    ):
        """Creates dataset from images and labels locally stored."""

        loader = ImageLabelLoader(img_dir, labels_dir)
        filenames = loader.get_filenames()

        labels = []
        images = []
        for filename in filenames:
            img, label = loader.load(filename)
            len_y, len_x, _ = img.shape
            img = raw_processor.transform(img)
            img_len = img.shape
            new_len_y = img_len[0]
            new_len_x = img_len[1]

            # we must rescale the original coordinates to the new shape
            coordinates = label.to_dict()
            coordinates["x0"] = int(coordinates["x0"] * new_len_x / len_x)
            coordinates["x1"] = int(coordinates["x1"] * new_len_x / len_x)
            coordinates["y0"] = int(coordinates["y0"] * new_len_y / len_y)
            coordinates["y1"] = int(coordinates["y1"] * new_len_y / len_y)
            reshaped_label = Rectangle.from_dict(coordinates)

            labels.append(reshaped_label)
            images.append(img)
        return cls(images, labels, model_processor)

    def select(self, idx: List[int]) -> Dataset:
        """Returns a reduced dataset based on the provided indexes."""

        selected_images = [self.images[i] for i in idx]
        selected_labels = [self.labels[i] for i in idx]
        return Dataset(selected_images, selected_labels, self.model_processor)

    def augment(self, increase_factor: int, augmentation: type[Compose]) -> None:
        """Expands the current dataset by creating augmented images for each sample."""

        new_images = []
        new_y = []
        for image, coord in zip(self.images, self.y):
            for _ in range(increase_factor):
                transformed = augmentation(image=image, keypoints=[coord])
                new_images.append(transformed["image"])
                new_y.append(transformed["keypoints"][0])
        self.y += new_y
        self.images += new_images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx]
        len_x = x.shape[0]
        len_y = x.shape[1]
        x = self.model_processor.transform(x)

        y = torch.Tensor(self.y[idx])
        y[0] = y[0] / len_x
        y[1] = y[1] / len_y
        y = torch.Tensor(y)

        return x, y
