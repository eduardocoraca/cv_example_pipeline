from abc import ABC, abstractmethod


class ImageProcessor(ABC):
    @abstractmethod
    def transform(self):
        pass
