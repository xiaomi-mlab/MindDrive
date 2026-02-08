CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
WAYPOINT_TOKEN = "<waypoint>"

# multi modal token
LEFT_WAYPOINT_TOKEN =  "<left_waypoint>"
RIGHT_WAYPOINT_TOKEN =  "<right_waypoint>"
STRAIGHT_WAYPOINT_TOKEN =  "<straight_waypoint>"
FOLLOW_WAYPOINT_TOKEN =  "<follow_waypoint>"
CHANGE_LEFT_WAYPOINT_TOKEN =  "<change_left_waypoint>"
CHANGE_RIGHT_WAYPOINT_TOKEN =  "<change_right_waypoint>"

# temporal waypoint token
WAYPOINT_TOKEN_0P5S =  "<waypoint_0p5s>"
WAYPOINT_TOKEN_1P0S =  "<waypoint_1p0s>"
WAYPOINT_TOKEN_1P5S =  "<waypoint_1p5s>"
WAYPOINT_TOKEN_2P0S =  "<waypoint_2p0s>"
WAYPOINT_TOKEN_2P5S =  "<waypoint_2p5s>"
WAYPOINT_TOKEN_3P0S =  "<waypoint_3p0s>"

# anchor modal waypoint token
WAYPOINT_ANCHOR_TOKEN_1 =  "<waypoint_anchor_1>"
WAYPOINT_ANCHOR_TOKEN_2 =  "<waypoint_anchor_2>"
WAYPOINT_ANCHOR_TOKEN_3 =  "<waypoint_anchor_3>"
WAYPOINT_ANCHOR_TOKEN_4 =  "<waypoint_anchor_4>"
WAYPOINT_ANCHOR_TOKEN_5 =  "<waypoint_anchor_5>"
WAYPOINT_ANCHOR_TOKEN_6 =  "<waypoint_anchor_6>"

# generate ego token
EGO_WAYPOINT_TOKEN = "<waypoint_ego>"
EGO_PATH_WAYPOINT_TOKEN = "<path_waypoint_ego>"

# meta-action
MAINTAIN_MODERATE_SPEED = "<maintain_moderate_speed>"
STOP = "<stop>"
MAINTAIN_SLOW_SPEED = "<maintain_slow_speed>"
SPEED_UP = "<speed_up>"
SLOW_DOWN = '<slow_down>'
MAIN_FAST_SPEED = '<maintain_fast_speed>'
SLOW_DOWN_RAPIDLY = '<slow_down_rapidly>'

LANEFOLLOW = "<lanefollow>"
STRAIGHT = '<straight>'
TURN_LEFT = '<turn_left>'
CHANGE_LANE_LEFT = '<change_lane_left>'
TURN_RIGHT = '<turn_right>'
CHANGE_LANE_RIGHT = '<change_lane_right>'