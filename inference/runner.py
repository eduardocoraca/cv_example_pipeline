from imutils.video import VideoStream, FPS
import time
import cv2
from model_pipeline import ModelPipeline
import numpy as np
from common import PointDrawer, RectangleDrawer, Rectangle
from logging import RootLogger
from datetime import datetime
from typing import Tuple


class Runner:
    """Runs the model pipeline in real time using video stream from a webcam."""

    def __init__(
        self, logger: type[RootLogger], wait_seconds: int = 3, max_attempts: int = 2
    ):
        self.model_pipeline = ModelPipeline()
        self.logger = logger
        self.wait_seconds = wait_seconds  # waiting time in case of disconnection
        self.max_attemps = max_attempts  # maximum number of attemps to reconnect
        self.attemps_counter = 0

    def initialize_capture(self):
        """Initialize video capture via webcam."""

        try:
            self.vs = VideoStream(src=0).start()
            self.fps = FPS().start()
            self.logger.info(f"Started run at {datetime.now()}.")
        except:
            self.logger.error(f"Failed to start run at {datetime.now()}.")
            raise Exception("Could not initialize webcam.")

    def __attempt_capture(self) -> np.ndarray:
        """Attempts to capture a new frame and retries if not successfull."""

        img = self.vs.read()
        success = self.vs.grabbed
        if success:
            self.attemps_counter = 0
            return img
        elif self.attemps_counter <= self.max_attemps:
            self.logger.warning(f"Attempting new capture at {datetime.now()}.")
            time.sleep(self.wait_seconds)
            self.attemps_counter += 1
            self.attempt_capture()

    def run(self):
        point_drawer = PointDrawer()
        rectangle_drawer = RectangleDrawer()
        while True:
            # image capture
            img = self.__attempt_capture()
            Y, X = img.shape[0:-1]  # image size
            center_coord = int(Y // 2), int(X // 2)  # image center coordinates

            # loop break condition
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            # run model
            norm_coord = self.model_pipeline(img)  # object normalized coordinates
            obj_coord = (int(norm_coord[0] * X), int(norm_coord[1] * Y))

            # plot detected point
            point_drawer.draw(img, obj_coord)

            # plot the center of the image (region of interest)
            center_rectangle = Rectangle(
                x0=center_coord[1] - int(0.05 * X),
                y0=center_coord[0] - int(0.05 * Y),
                x1=center_coord[1] + int(0.05 * X),
                y1=center_coord[0] + int(0.05 * Y),
            )
            rectangle_drawer.draw(img, center_rectangle)
            cv2.imshow("Webcam", img)
            self.fps.update()

        cv2.destroyAllWindows()

    def stop_capture(self):
        self.fps.stop()
        self.vs.stop()
        self.logger.info(f"Stopped run at {datetime.now()}")
