import sys

sys.path.append(".")

from common import Rectangle
import cv2
import os
import numpy as np
from typing import Tuple
import json
import shutil


class ImageLabeler:
    def __init__(
        self,
        src_path: str = r"data\unlabeled",
        dst_path: str = r"data\labels",
        out_path: str = r"data\images",
    ):
        self.src_path = src_path
        self.dst_path = dst_path
        self.out_path = out_path
        self.__check_dir(self.src_path)
        self.__check_dir(self.dst_path)
        self.__check_dir(self.out_path)
        self.unlabeled_filenames = os.listdir(src_path)

        self.current_box = Rectangle()
        self.current_img = None

    def __check_dir(self, path: str) -> None:
        """Checks if directory exists and creates if it doesn't."""

        if not os.path.exists(path):
            os.makedirs(path)

    def __draw_and_save(self, event, x: int, y: int, flags, *userdata):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.current_box.clear()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.current_box.add_point(x, y)
            if self.current_box.is_complete():
                x0, y0, x1, y1 = self.current_box.get_points()
                rec = cv2.rectangle(
                    self.current_img.copy(),
                    (x0, y0),
                    (x1, y1),
                    thickness=2,
                    color=(0, 255, 0),
                )
                cv2.imshow(self.current_window, rec)
                cv2.waitKey(0)

        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.destroyAllWindows()

    def __ask_label(self, img: type[np.ndarray], file_path: str) -> bool:
        self.current_window = f"Raw image: {file_path}"
        cv2.namedWindow(self.current_window)
        cv2.imshow(self.current_window, img)
        cv2.setMouseCallback(self.current_window, self.__draw_and_save)
        cv2.waitKey(0)

        success = True if self.current_box.is_complete() else False
        return success

    def __save_label(self, label: type[Rectangle], dst: str, filename: str) -> None:
        label_filename = filename.split(".")[0]
        with open(f"{dst}\{label_filename}.json", "w") as f:
            points = self.current_box.to_dict()
            json.dump(points, f)

    def __move_file(self, filename: str) -> None:
        """Moves the image in file_path to the output path."""

        src_path = f"{self.src_path}\{filename}"
        dst_path = f"{self.out_path}\{filename}"
        shutil.move(src_path, dst_path)

    def label_one(self, filename: str) -> None:
        """Requests label for one image."""

        file_path = f"{self.src_path}\{filename}"
        img = cv2.imread(file_path)
        self.current_img = img.copy()
        success = self.__ask_label(img, file_path)

        if success:
            self.__move_file(filename)
            self.__save_label(self.current_box, dst=self.dst_path, filename=filename)
            print(f"File {filename} successfully labeled.")
        self.current_box.clear()

    def label_dir(self) -> None:
        """Requests label for all images in src_path."""

        filenames = os.listdir(self.src_path)
        filenames = [f for f in filenames if f.endswith(".jpg")]
        num_files = len(filenames)

        if num_files == 0:
            print("No files no label. Make sure images have .jpg extension.")

        for i, filename in enumerate(filenames):
            print(f"Labeling {i+1} from {num_files} images.")
            self.label_one(filename)
