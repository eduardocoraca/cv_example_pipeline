from imutils.video import VideoStream, FPS
import time
import cv2
from model_pipeline import ModelPipeline
import numpy as np
from common import PointDrawer, RectangleDrawer


class Runner:
    """Runs the model pipeline in real time using video stream from a webcam."""

    def __init__(self, wait_seconds: int = 3, max_attempts: int = 2):
        self.model_pipeline = ModelPipeline()
        self.wait_seconds = wait_seconds  # waiting time in case of disconnection
        self.max_attemps = max_attempts  # maximum number of attemps to reconnect
        self.attemps_counter = 0

    def initialize_capture(self):
        """Initialize video capture via webcam."""

        try:
            self.vs = VideoStream(src=0).start()
            self.fps = FPS().start()
        except:
            raise Exception("Could not initialize webcam.")

    def __attempt_capture(self) -> np.ndarray:
        """Attempts to capture a new frame and retries if not successfull."""

        img = self.vs.read()
        success = self.vs.grabbed
        if success:
            self.attemps_counter = 0
            return img
        elif self.attemps_counter <= self.max_attemps:
            time.sleep(self.wait_seconds)
            self.attemps_counter += 1
            self.attempt_capture()

    def run(self):
        point_drawer = PointDrawer()
        rectangle_drawer = RectangleDrawer()
        while True:
            # image capture
            img = self.__attempt_capture()
            W, L = img.shape[0:-1]  # image size

            # loop break condition
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            # run model
            norm_coord = self.model_pipeline(img)  # object normalized coordinates

            # plot detected point
            center_coord = int(W // 2), int(L // 2)  # image center coordinates
            obj_coord = (int(norm_coord[0] * L), int(norm_coord[1] * W))
            point_drawer.draw(img, obj_coord)

            # plot the center of the image (region of interest)
            rectangle_drawer.draw(img, center_coord, (int(0.1 * W), int(0.1 * L)))
            cv2.imshow("Webcam", img)

            self.fps.update()
        cv2.destroyAllWindows()

    def stop_capture(self):
        self.fps.stop()
        self.vs.stop()
