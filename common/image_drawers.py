from .interfaces import ShapeDrawer
from .classes import Rectangle
import numpy as np
import cv2
from typing import Tuple


class PointDrawer(ShapeDrawer):
    """Draws points (filled circles with low radius)."""

    def draw(self, img: np.ndarray, coord: Tuple[int, int]) -> None:
        cv2.circle(img, coord, 5, color=(0, 255, 0), thickness=-1)


class RectangleDrawer(ShapeDrawer):
    """Draws rectangles."""

    def draw(self, img: np.ndarray, rectangle: type[Rectangle]) -> None:
        x0, y0, x1, y1 = rectangle.get_points()
        cv2.rectangle(
            img,
            (x0, y0),
            (x1, y1),
            color=(0, 0, 255),
            thickness=2,
        )
