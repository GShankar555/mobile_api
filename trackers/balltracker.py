from utils import get_center_of_bbox
import cv2
import numpy as np

class BallTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.last_ball_position = None

    def update(self, ball_bbox):
        x, y = get_center_of_bbox(ball_bbox)
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if self.last_ball_position is None:
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kalman.correct(measurement)
        self.last_ball_position = (x, y)

    def predict(self):
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))