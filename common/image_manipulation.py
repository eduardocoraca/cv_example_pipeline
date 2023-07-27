from typing import Tuple, Union
from .classes import Rectangle
import numpy as np
import cv2
import os
from .interfaces import ImageProcessor
import torch
import json


class ImageLabelLoader:
    """A loader for a pair image, label."""

    def __init__(self, img_dir: str, labels_dir: str):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.images = os.listdir(self.img_dir)
        self.labels = os.listdir(self.labels_dir)
        self.filenames = [image.split(".")[0] for image in self.images]

    def get_filenames(self):
        return self.filenames

    def load(self, filename: str) -> Union[Tuple[np.ndarray, Rectangle], None]:
        """Loads the data for the filename (without extension)."""

        label_file = f"{filename}.json"
        img_file = f"{filename}.jpg"
        if label_file in self.labels and img_file in self.images:
            img = cv2.imread(f"{self.img_dir}/{img_file}")
            with open(f"{self.labels_dir}/{label_file}", "r") as f:
                label_json = json.load(f)
            label = Rectangle.from_dict(label_json)
        return img, label


class RawImageProcessor(ImageProcessor):
    """Processes raw image."""

    def __init__(self, size_x: int, size_y: int, to_gray: bool):
        self.size_x = size_x
        self.size_y = size_y
        self.to_gray = to_gray

    def transform(self, img: type[np.ndarray]) -> type[np.ndarray]:
        """Transforms the raw BGR image to a grayscale and reshaped version."""

        img_cp = img.copy()
        if self.to_gray:
            out_img = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
        out_img = cv2.resize(
            out_img, (self.size_x, self.size_y), interpolation=cv2.INTER_AREA
        )
        return out_img


class ModelImageProcessor(ImageProcessor):
    """Processes image per model requirements."""

    def __init__(self):
        pass

    def transform(self, img: type[np.ndarray]) -> type[torch.Tensor]:
        """Transforms the grayscale image to a torch Tensor."""

        x = torch.Tensor(img)
        x = x.tile(3, 1, 1)
        x = (x - x.min()) / (x.max() - x.min())  # normalizing to [0,1] range
        return x
