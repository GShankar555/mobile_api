import sys 
sys.path.append('./')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_center = get_center_of_bbox(player['bbox'])
            distance = measure_distance(player_center, ball_position)
            if distance < self.max_player_ball_distance and distance < miniumum_distance:
                miniumum_distance = distance
                assigned_player = player_id
                
        return assigned_player