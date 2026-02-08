from collections import deque
import numpy as np

class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, turn_KP=1.7, turn_KI=0.3, turn_KD=0.6, turn_n=40, speed_KP=5.0, speed_KI=0.5, speed_KD=1.0, speed_n=40, max_throttle=0.75, brake_speed=0.05, brake_ratio=1.1, clip_delta=0.25, aim_dist=3.5, angle_thresh=0.3, dist_thresh=10): 
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

    def control_pid(self, path_waypoint, speed_waypoint, speed, target):
        num_pairs = len(path_waypoint) - 1
        best_norm = 1e5
        aim = path_waypoint[0]
        
        for i in range(num_pairs):
            norm = np.linalg.norm(path_waypoint[i])
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = path_waypoint[i]
                best_norm = norm

            midpoint = (path_waypoint[i+1] + path_waypoint[i]) / 2.0
            norm = np.linalg.norm(midpoint)
            if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
                aim = midpoint
                best_norm = norm

        aim_last = path_waypoint[-1] - path_waypoint[-2]

        desired_speed = 0.75 * np.linalg.norm(speed_waypoint[0]) * 2 + \
                        0.25 * np.linalg.norm(speed_waypoint[1] - speed_waypoint[0]) * 2

        angle = np.degrees(np.pi/2 - np.arctan2(aim[1], aim[0]))/90 if aim[1] > 0.02 else 0.0
        angle_last = np.degrees(np.pi/2 - np.arctan2(aim_last[1], aim_last[0]))/90
        angle_target = np.degrees(np.pi/2 - np.arctan2(target[1], target[0]))/90

        use_target_to_aim = False 
        angle_final = angle_target if use_target_to_aim else angle

        steer = np.clip(self.turn_controller.step(angle_final), -1.0, 1.0)
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta) if not brake else 0.0
        throttle = np.clip(throttle, 0.0, self.max_throttle)

        metadata = {
            'speed': float(speed),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(path_waypoint[3].astype(np.float64)) if len(path_waypoint) > 3 else (0,0),
            'wp_3': tuple(path_waypoint[2].astype(np.float64)) if len(path_waypoint) > 2 else (0,0),
            'wp_2': tuple(path_waypoint[1].astype(np.float64)),
            'wp_1': tuple(path_waypoint[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed),
            'angle': float(angle),
            'angle_last': float(angle_last),
            'angle_target': float(angle_target),
            'angle_final': float(angle_final),
            'delta': float(delta),
        }

        return steer, throttle, brake, metadata