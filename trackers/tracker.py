from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO
import pickle
import os
import numpy as np
import cv2
import sys
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import logging
from datetime import datetime
from .balltracker import BallTracker
from sklearn.cluster import KMeans
from player_holdings import PlayerBallAssigner
sys.path.append('../')

logging.basicConfig(filename="basketball_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

class Tracker:
    def __init__(self, model_path: str, detections_path: str = "detections", conf_threshold: float = 0.7, max_age: int = 100, n_init: int = 5, iou_threshold: float = 0.2):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=max_age, n_init=n_init,nn_budget=100,max_cosine_distance=0.3,max_iou_distance=0.5)
        self.detections_path = detections_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.player_holding = None
        self.basket_text = None
        self.ball_tracker = BallTracker()
        self.player_score = defaultdict(int)
        self.basket_cooldown = 0
        self.player_ball = PlayerBallAssigner()

    def log_basket_event(self,frame_num, player_id, score_type="2-pointer"):
        event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"Frame {frame_num}: Basket made by Player {player_id}, Type: {score_type} at {event_time}"
        logging.info(log_message)
        print(log_message)

    def detect_frames(self, frames):
        if os.path.exists(self.detections_path):
            with open(self.detections_path, 'rb') as f:
                return pickle.load(f)
        detections = self.model(frames, conf=self.conf_threshold, verbose=False)
        with open(self.detections_path, 'wb') as f:
            pickle.dump(detections, f)
        return detections

    def get_dynamic_threshold(self,player_bbox, scale_factor=0.2):
        player_height = player_bbox[3] - player_bbox[1]
        return max(30, scale_factor * player_height)

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = {
            "players": [defaultdict(dict) for _ in range(len(frames))],
            "ball": [defaultdict(dict) for _ in range(len(frames))],
            "hoop": [defaultdict(dict) for _ in range(len(frames))],
            "2p": [defaultdict(dict) for _ in range(len(frames))]
        }
        cls_names_inv = {'2p': 0, 'ball': 1, 'hoop': 2, 'player': 3}
        for frame_num, detection in enumerate(detections):
            detections_list = []
            ball_bbox = None
            for det in detection:
                bbox = det.boxes.xyxy[0].tolist()
                cls_id = int(det.boxes.cls[0].item())
                conf = float(det.boxes.conf[0].item())

                if cls_id == cls_names_inv['player']:
                    detections_list.append(([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], conf, cls_id))
                elif cls_id == cls_names_inv['ball']:
                    ball_bbox = bbox
                elif cls_id == cls_names_inv['hoop']:
                    tracks["hoop"][frame_num][1] = {"bbox": bbox}
                elif cls_id == cls_names_inv['2p']:
                    if hasattr(det, "masks") and det.masks is not None:
                        mask_array = det.masks[0].xy[0]
                        tracks["2p"][frame_num]["mask"][1] = {"mask":mask_array}
                        tracks["2p"][frame_num]["bbox"][1] = {"bbox": bbox}
                    else:
                        tracks["2p"][frame_num][1] = {"bbox": bbox}

            track_results = self.tracker.update_tracks(detections_list, frame=frames[frame_num])
            for track in track_results:
                track_id = track.track_id
                bbox = track.to_ltrb()
                cls_id = track.det_class
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

            if ball_bbox:
                self.ball_tracker.update(ball_bbox)
                tracks["ball"][frame_num] = ball_bbox
            else:
                predicted_ball_pos = self.ball_tracker.predict()
                if predicted_ball_pos:
                    tracks["ball"][frame_num] = [
                        predicted_ball_pos[0] - 10,
                        predicted_ball_pos[1] - 10,
                        predicted_ball_pos[0] + 10,
                        predicted_ball_pos[1] + 10
                    ]

        return tracks

    def iou(self,boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def is_player_holding_ball(self, player_bbox, ball_bbox, iou_threshold=0.2):
        player_center = get_center_of_bbox(player_bbox)
        ball_center = get_center_of_bbox(ball_bbox)
        threshold_dist = self.get_dynamic_threshold(player_bbox)
        distance = np.linalg.norm(np.array(player_center) - np.array(ball_center))
        iou_value = self.iou(player_bbox, ball_bbox)

        return (distance < threshold_dist) or (iou_value > iou_threshold)
    
    def is_player_inside_2p(self, player_bbox, _2p_bbox):
        if player_bbox is None or _2p_bbox is None:
            return False
        x,y = get_foot_position(player_bbox)
        bx1, by1, bx2, by2 = _2p_bbox
        return bx1 <= x <= bx2 and by1 <= y <= by2
    
    def is_ball_inside_hoop(self, ball_bbox, rim_bbox):
        bx1, by1, bx2, by2 = ball_bbox
        rx1, ry1, rx2, ry2 = rim_bbox
        return bx1 >= rx1 or bx2 <= rx2 or by1 >= ry1 or by2 <= ry2

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        last_known_bbox = {}
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_num]
            ball_bbox = tracks["ball"][frame_num]
            rim_bbox = next(iter(tracks["hoop"][frame_num].values()), {}).get('bbox', None)
            _2p_mask = tracks["2p"][frame_num]["mask"].get(1, {}).get("mask", None)
            _2p_bbox = tracks["2p"][frame_num]["bbox"].get(1, {}).get("bbox", None)
            assigned_player = self.player_ball.assign_ball_to_player(player_dict, ball_bbox)
            
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                last_known_bbox[track_id] = bbox
                # x_min, y_min, x_max, y_max = map(int, bbox)
                # x_min = max(0, min(x_min, frame.shape[1] - 1))
                # y_min = max(0, min(y_min, frame.shape[0] - 1))
                # x_max = max(0, min(x_max, frame.shape[1]))
                # y_max = max(0, min(y_max, frame.shape[0]))

                # player_frame = frame[y_min:y_max, x_min:x_max]
                # top_half_image=  player_frame[: int(player_frame.shape[0]/2), :]
                # image_2d = top_half_image.reshape(-1, 3)
                # kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
                # kmeans.fit(image_2d)                
                # labels = kmeans.labels_
                # clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
                # corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
                # non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
                # player_cluster = 1-non_player_cluster
                # colors = list(map(int,kmeans.cluster_centers_[player_cluster]))
                frame = self.draw_ellipse(frame, player["bbox"], (255,0,0), track_id)                
            if assigned_player != -1:
                self.player_holding = assigned_player
                print(f"{assigned_player} holding ball")
            if ball_bbox is not None:
                frame = self.draw_triangle(frame, ball_bbox, (0, 255, 0))

            if rim_bbox is not None:
                cv2.rectangle(frame, (int(rim_bbox[0]), int(rim_bbox[1])), (int(rim_bbox[2]), int(rim_bbox[3])), (255, 255, 0), 2)

            if ball_bbox is not None and rim_bbox is not None:
                rim_x, rim_y = get_center_of_bbox(rim_bbox)
                ball_x, ball_y = get_center_of_bbox(ball_bbox)
                rim_width = get_bbox_width(rim_bbox)

                if self.basket_cooldown == 0 and self.is_ball_inside_hoop(ball_bbox, rim_bbox) and abs(ball_x - rim_x) < rim_width * 0.5 and self.player_holding is not None:
                    basket_type = "3p"
                    player_bbox = last_known_bbox.get(self.player_holding, None)
                    if player_bbox is not None and _2p_mask is not None and self.is_player_inside_2p(player_bbox, _2p_bbox):
                        basket_type = "2p"
                        self.player_score[self.player_holding] += 2
                    else:
                        self.player_score[self.player_holding] += 3
                    self.basket_text = f"Basket made by {self.player_holding} || score: {self.player_score[self.player_holding]} || {basket_type}"
                    self.log_basket_event(frame_num, self.player_holding, basket_type)
                    self.basket_cooldown = 30

            if self.basket_cooldown > 0:
                self.basket_cooldown -= 1

            if self.basket_text is not None:
                cv2.putText(frame, self.basket_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if _2p_mask is not None:
                points = np.array(_2p_mask, dtype=np.float32).reshape((-1, 2))
                points[:, 0] = gaussian_filter1d(points[:, 0], sigma=2)
                points[:, 1] = gaussian_filter1d(points[:, 1], sigma=2)
                smoothed_points = points.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [smoothed_points], isClosed=True, color=(0, 255, 0), thickness=2)

            output_video_frames.append(frame)

        return output_video_frames

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, (x_center, y2), (int(0.7*width), int(0.15 * width)), 0, -45, 235, color, 2)

        if track_id is not None:
            cv2.putText(frame, f"{track_id}", (x_center - 10, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        size = max(5, int(get_bbox_width(bbox) * 0.2))
        triangle_points = np.array([[x, y], [x - size, y - (2 * size)], [x + size, y - (2 * size)]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        return frame