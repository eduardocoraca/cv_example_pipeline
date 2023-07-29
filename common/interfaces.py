from abc import ABC, abstractmethod


class ImageProcessor(ABC):
    @abstractmethod
    def transform(self):
        pass


class ShapeDrawer(ABC):
    @abstractmethod
    def draw(self) -> None:
        pass
