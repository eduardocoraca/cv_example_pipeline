from .interfaces import ShapeDrawer
import numpy as np
import cv2
from typing import Tuple


class PointDrawer(ShapeDrawer):
    """Draws points (filled circles with low radius)."""

    def draw(self, img: np.ndarray, center_coord: Tuple[int, int]) -> None:
        cv2.circle(img, center_coord, 5, color=(0, 255, 0), thickness=-1)


class RectangleDrawer(ShapeDrawer):
    """Draws rectangles."""

    def draw(
        self, img: np.ndarray, center_coord: Tuple[int, int], widths: Tuple[int, int]
    ) -> None:
        x_center, y_center = center_coord
        dx, dy = widths
        dx_half = int(dx // 2)
        dy_half = int(dy // 2)
        x_left = x_center - dx_half
        x_right = x_center + dx_half
        y_lower = y_center - dy_half
        y_upper = y_center + dy_half
        cv2.rectangle(
            img,
            (y_upper, x_left),
            (y_lower, x_right),
            color=(0, 0, 255),
            thickness=2,
        )

    def draw_from_coord(
        self,
        img: np.ndarray,
        top_left_coord: Tuple[int, int],
        bottom_right_coord: Tuple[int, int],
    ) -> None:
        cv2.rectangle(
            img,
            top_left_coord,
            bottom_right_coord,
            thickness=2,
            color=(0, 255, 0),
        )
