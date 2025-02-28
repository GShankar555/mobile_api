from typing import Tuple, List

def get_center_of_bbox(bbox: List[float]) -> Tuple[int, int]:
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

def get_bbox_width(bbox: List[float]) -> float:
    return abs(bbox[2] - bbox[0])

def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def get_foot_position(bbox: List[float]) -> Tuple[int, int]:
    return (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))