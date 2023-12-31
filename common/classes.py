from typing import Tuple, List, Union
import numpy as np
from datetime import datetime


class Rectangle:
    """Rectangle object containing two points (x0,y0), (x1,y1)."""

    def __init__(self, x0: int = None, y0: int = None, x1: int = None, y1: int = None):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def add_point(self, x: int, y: int) -> None:
        if self.x1 != None and self.y1 != None:
            self.clear()

        if self.x0 != None and self.y0 != None:
            self.x1 = x
            self.y1 = y
        else:
            self.x0 = x
            self.y0 = y

    def contains(self, point: Tuple[int, int]) -> bool:
        """Checks if a point is within the rectangle."""

        x, y = point
        return (x >= self.x0) and (x <= self.x1) and (y >= self.y0) and (y <= self.y1)

    def is_complete(self) -> bool:
        """Checks if the rectangle has 2 defined points."""

        return (
            self.x0 != None and self.x1 != None and self.y0 != None and self.y1 != None
        )

    def clear(self):
        """Clears saved points."""

        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            x0=data.get("x0"),
            x1=data.get("x1"),
            y0=data.get("y0"),
            y1=data.get("y1"),
        )

    def to_dict(self):
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    def get_points(
        self,
    ) -> List[Union[int, float]]:
        return [self.x0, self.y0, self.x1, self.y1]

    def get_center(self) -> Tuple[int, int]:
        return int(abs(self.x1 + self.x0) // 2), int(abs(self.y1 + self.y0) // 2)


class Event:
    def __init__(
        self, timestamp: type[datetime], original_img: np.ndarray, event_img: np.ndarray
    ):
        self.original_img = original_img
        self.event_img = event_img
        self.timestamp = timestamp

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "original_img": self.original_img,
            "event_img": self.event_img,
        }
