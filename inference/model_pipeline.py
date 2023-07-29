import sys

sys.path.append(".")

import torch
from common import RawImageProcessor, ModelImageProcessor
import numpy as np
from typing import Tuple

IMG_SHAPE = 250


class ModelPipeline:
    """Processes the raw image and runs through the model, provinding a coordinate pair."""

    def __init__(self):
        try:
            self.model = torch.load("training/saved_models/model.pkl")
            self.model.eval()
            self.model.cpu()
        except:
            raise Exception(
                "Could not load model, make sure there is a model.pkl located at training/saved_models/"
            )

        self.raw_processor = RawImageProcessor(IMG_SHAPE, IMG_SHAPE, to_gray=True)
        self.model_processor = ModelImageProcessor()

    def __call__(self, img: np.ndarray) -> Tuple[float, float]:
        """Run the pipeline and returns normalized coordinates in (0,1) range."""

        x = self.raw_processor.transform(img)
        x = self.model_processor.transform(x)
        x = x.unsqueeze(0)  # batch dim
        with torch.no_grad():
            y = self.model(x).squeeze().numpy()
        return y[0], y[1]
