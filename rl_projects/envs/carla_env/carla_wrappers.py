import carla
import numpy as np
import weakref
import math

def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[:truncate - 1] + u"\u2026") if len(name) > truncate else name


def get_displacement_vector(car_pos, waypoint_pos, theta):
    """
    Calculates the displacement vector from the car to a waypoint, taking into account the orientation of the car.

    Parameters:
        car_pos (numpy.ndarray): 1D numpy array of shape (3,) representing the x, y, z coordinates of the car.
        waypoint_pos (numpy.ndarray): 1D numpy array of shape (3,) representing the x, y, z coordinates of the waypoint.
        theta (float): Angle in radians representing the orientation of the car.

    Returns:
        numpy.ndarray: 1D numpy array of shape (3,) representing the displacement vector from the car to the waypoint,
        with the car as the origin and the y-axis pointing in the direction of the car's orientation.
    """
    # Calculate the relative position of the waypoint with respect to the car
    relative_pos = waypoint_pos - car_pos

    theta = theta
    # Construct the rotation transformation matrix
    R = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    T = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    # Apply the rotation matrix to the relative position vector
    waypoint_car = R @ relative_pos
    waypoint_car = T @ waypoint_car
    # Set values very close to zero to exactly zero
    waypoint_car[np.abs(waypoint_car) < 10e-10] = 0

    return waypoint_car


def angle_diff(v0, v1):
    """
    Calculates the signed angle difference between 2D vectors v0 and v1.
    It returns the angle difference in radians between v0 and v1.
    The v0 is the reference for the sign of the angle
    """
    v0_xy = v0[:2]
    v1_xy = v1[:2]
    v0_xy_norm = np.linalg.norm(v0_xy)
    v1_xy_norm = np.linalg.norm(v1_xy)
    if v0_xy_norm == 0 or v1_xy_norm == 0:
        return 0

    v0_xy_u = v0_xy / v0_xy_norm
    v1_xy_u = v1_xy / v1_xy_norm
    dot_product = np.dot(v0_xy_u, v1_xy_u)
    angle = np.arccos(dot_product)

    # Calculate the sign of the angle using the cross product
    cross_product = np.cross(v0_xy_u, v1_xy_u)
    if cross_product < 0:
        angle = -angle
    if abs(angle) >= 2.3:
        return 0
    return round(angle, 2)


def distance_to_line(A, B, p):
    p[2] = 0
    num = np.linalg.norm(np.cross(B - A, A - p))
    denom = np.linalg.norm(B - A)
    if np.isclose(denom, 0):
        return np.linalg.norm(p - A)
    return num / denom


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


def smooth_action(old_value, new_value, smooth_factor):
    return old_value * smooth_factor + new_value * (1.0 - smooth_factor)


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2].astype(int)


sensor_transforms = {
    "spectator": carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
    "lidar": carla.Transform(carla.Location(x=0.0, z=2.4)),
    "dashboard": carla.Transform(carla.Location(z=10, x=10), carla.Rotation(pitch=-90)),
    "birdview0": carla.Transform(carla.Location(x=0, y=0, z=16), carla.Rotation(pitch=-90)),
    "birdview1": carla.Transform(carla.Location(z=6.4, x=5.2, y=-1.6), carla.Rotation(pitch=-90)),
    "fpv": carla.Transform(carla.Location(x=0.6, y=-0.5, z=1.2)),
    "CAM_FRONT": carla.Transform(carla.Location(x=0.8, y=0.0, z=1.6)),
    "CAM_FRONT_LEFT": carla.Transform(carla.Location(x=0.27, y=-0.55, z=1.6),carla.Rotation(yaw=-55)),
    "CAM_FRONT_RIGHT": carla.Transform(carla.Location(x=0.27, y=0.55, z=1.6),carla.Rotation(yaw=55)),
    "CAM_BACK": carla.Transform(carla.Location(x=-2.0, y=0.0, z=1.6),carla.Rotation(yaw=180.0)),
    "CAM_BACK_LEFT": carla.Transform(carla.Location(x=-0.32, y=-0.55, z=1.6),carla.Rotation(yaw=-110.0)),
    "CAM_BACK_RIGHT": carla.Transform(carla.Location(x=-0.32, y=0.55, z=1.6),carla.Rotation(yaw=110.0)),
    "imu": carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0)),
    "gps": carla.Transform(carla.Location(x=-1.4, y=0.0, z=0.0)),
}


# ===============================================================================
# CarlaActorBase
# ===============================================================================

class CarlaActorBase(object):
    def __init__(self, world, actor):
        self.world = world
        self.actor = actor
        self.destroyed = False

    def destroy(self):
        pass
    def get_carla_actor(self):
        return self.actor

    def tick(self):
        pass

    def __getattr__(self, name):
        """Relay missing methods to underlying carla actor"""
        return getattr(self.actor, name)


# ===============================================================================
# Lidar
# ===============================================================================

class Lidar(CarlaActorBase):
    def __init__(self, world, width=120, height=120, transform=carla.Transform(), on_recv_image=None,
                 attach_to=None):
        self._width = width
        self._height = height
        self.on_recv_image = on_recv_image
        self.range = 20
        # Setup lidar blueprint
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('points_per_second', '50000')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', str(self.range))
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('horizontal_fov', '110')
        lidar_bp.set_attribute('lower_fov', '-20')
        lidar_bp.set_attribute('rotation_frequency', '30')

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(lidar_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda lidar_data: Lidar.process_lidar_input(weak_self, lidar_data))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_lidar_input(weak_self, raw):
        self = weak_self()
        if not self:
            return
        if callable(self.on_recv_image):
            lidar_range = 2.0 * float(self.range)

            points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._width, self._height) / lidar_range
            lidar_data += (0.5 * self._width, 0.5 * self._height)
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._width, self._height, 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.on_recv_image(lidar_img)
        """
        if callable(self.on_recv_image):
            lidar_data.convert(self.color_converter)
            array = np.frombuffer(lidar_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (lidar_data.height, lidar_data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.on_recv_image(array)
            """


# ===============================================================================
# CollisionSensor
# ===============================================================================

class CollisionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_collision_fn):
        self.on_collision_fn = on_collision_fn

        # Collision history
        self.history = []

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.collision") # 设置传感器的蓝图

        # Create and setup sensor 将传感器弄到车辆上
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: CollisionSensor.on_collision(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_collision_fn
        if callable(self.on_collision_fn):
            self.on_collision_fn(event)


# ===============================================================================
# LaneInvasionSensor
# ===============================================================================

class LaneInvasionSensor(CarlaActorBase):
    def __init__(self, world, vehicle, on_invasion_fn):
        self.on_invasion_fn = on_invasion_fn

        # Setup sensor blueprint
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")

        # Create sensor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle.get_carla_actor())
        actor.listen(lambda event: LaneInvasionSensor.on_invasion(weak_self, event))

        super().__init__(world, actor)

    @staticmethod
    def on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return

        # Call on_invasion_fn
        if callable(self.on_invasion_fn):
            self.on_invasion_fn(event)


# ===============================================================================
# Camera
# ===============================================================================

class Camera6V(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 attach_to=None, on_recv_image=None, fov=110, sensor_tick=10, cam_type='cam',
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw, custom_palette=False):
        self.on_recv_image = on_recv_image
        self.cam_type = cam_type
        self.color_converter = color_converter

        self.custom_palette = custom_palette
        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", f"{fov}")
        # camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera6V.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            print('no img')
            return
        if callable(self.on_recv_image):

            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if self.custom_palette:
                classes = {
                    0: [0, 0, 0],  # None
                    1: [0, 0, 0],  # Buildings
                    2: [0, 0, 0],  # Fences
                    3: [0, 0, 0],  # Other
                    4: [0, 0, 0],  # Pedestrians
                    5: [0, 0, 0],  # Poles
                    6: [157, 234, 50],  # RoadLines
                    7: [50, 64, 128],  # Roads
                    8: [255, 255, 255],  # Sidewalks
                    9: [0, 0, 0],  # Vegetation
                    10: [255, 0, 0],  # Vehicles
                    11: [0, 0, 0],  # Walls
                    12: [0, 0, 0]  # TrafficSigns
                }
                segimg = np.round((array[:, :, 0])).astype(np.uint8)
                array = array.copy()
                for j in range(array.shape[0]):
                    for i in range(array.shape[1]):
                        r_id = segimg[j, i]
                        if r_id <= 12:
                            array[j, i] = classes[segimg[j, i]]
                        else:
                            array[j, i] = classes[0]

            self.on_recv_image(array, self.cam_type)
        else:
            print('no callable')

    def destroy(self):
        super().destroy()

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(),
                 attach_to=None, on_recv_image=None, fov=110, sensor_tick=10, cam_type='cam',
                 camera_type="sensor.camera.rgb", color_converter=carla.ColorConverter.Raw, custom_palette=False):
        
        self.on_recv_image = on_recv_image
        self.cam_type = cam_type
        self.color_converter = color_converter

        self.custom_palette = custom_palette
        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find(camera_type)
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", f"{fov}")
        # camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            print('no img')
            return
        if callable(self.on_recv_image):

            image.convert(self.color_converter)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if self.custom_palette:
                classes = {
                    0: [0, 0, 0],  # None
                    1: [0, 0, 0],  # Buildings
                    2: [0, 0, 0],  # Fences
                    3: [0, 0, 0],  # Other
                    4: [0, 0, 0],  # Pedestrians
                    5: [0, 0, 0],  # Poles
                    6: [157, 234, 50],  # RoadLines
                    7: [50, 64, 128],  # Roads
                    8: [255, 255, 255],  # Sidewalks
                    9: [0, 0, 0],  # Vegetation
                    10: [255, 0, 0],  # Vehicles
                    11: [0, 0, 0],  # Walls
                    12: [0, 0, 0]  # TrafficSigns
                }
                segimg = np.round((array[:, :, 0])).astype(np.uint8)
                array = array.copy()
                for j in range(array.shape[0]):
                    for i in range(array.shape[1]):
                        r_id = segimg[j, i]
                        if r_id <= 12:
                            array[j, i] = classes[segimg[j, i]]
                        else:
                            array[j, i] = classes[0]

            self.on_recv_image(array, self.cam_type)
        else:
            print('no callable')

    def destroy(self):
        super().destroy()

# ===============================================================================
# Vehicle
# ===============================================================================

class Vehicle(CarlaActorBase):
    def __init__(self, world, transform=carla.Transform(),
                 on_collision_fn=None, on_invasion_fn=None,
                 vehicle_type="vehicle.tesla.model3", is_ego=False):
        # Setup vehicle blueprint
        vehicle_bp = world.get_blueprint_library().find(vehicle_type) 
        if is_ego:
            vehicle_bp.set_attribute('role_name', 'hero')
            # vehicle_bp.set_attribute('color', '0, 255, 0')
            color = vehicle_bp.get_attribute("color").recommended_values[0]
        else:
            color = vehicle_bp.get_attribute("color").recommended_values[0]
            vehicle_bp.set_attribute("color", color)

        # Create vehicle actor
        actor = world.spawn_actor(vehicle_bp, transform) # spawn_actor创建对象  Transform stating a location and rotation for the actor.
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)

        # Maintain vehicle control
        self.control = carla.VehicleControl() # Carla API 控制指令

        if callable(on_collision_fn): # 回调函数
            self.collision_sensor = CollisionSensor(world, self, on_collision_fn=on_collision_fn)
        if callable(on_invasion_fn):
            self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn=on_invasion_fn)

    def tick(self):
        self.actor.apply_control(self.control)

    def get_speed(self):
        """
        Return current vehicle speed in km/h
        """
        velocity = self.get_velocity()
        return 3.6 * np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def get_angle(self, waypoint):
        fwd = vector(self.get_velocity())
        wp_fwd = vector(waypoint.transform.rotation.get_forward_vector())
        return angle_diff(wp_fwd, fwd)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

    def set_autopilot(self, is_true, *args, **kwargs):
        self.actor.set_autopilot(is_true, *args, **kwargs)


# ===============================================================================
# my_Vehicle
# ===============================================================================

class My_Vehicle(CarlaActorBase):
    def __init__(self, world, actor, on_collision_fn=None, on_invasion_fn=None, is_ego=False):

        super().__init__(world, actor)
        self.actor = actor
        self.control = carla.VehicleControl()

        # if callable(on_collision_fn):
        #     self.collision_sensor = CollisionSensor(world, self, on_collision_fn)
        # if callable(on_invasion_fn):
        #     self.lane_sensor = LaneInvasionSensor(world, self, on_invasion_fn)


    # def tick(self):
    #     self.actor.apply_control(self.control)

    def get_closest_waypoint(self):
        return self.world.map.get_waypoint(self.get_transform().location, project_to_road=True)

    def set_autopilot(self, is_true, *args, **kwargs):
        self.actor.set_autopilot(is_true, *args, **kwargs)

# ===============================================================================
# World
# ===============================================================================

class World():
    def __init__(self, client, town='Town02'):
        self.world = client.load_world(town) # Creates a new world with default settings using map_name map. All actors in the current world will be destroyed.
        self.map = self.get_map() 
        self.actor_list = []

    def tick(self):
        for actor in list(self.actor_list):
            actor.tick()

        self.world.tick()

    def destroy(self):
        print("Destroying all spawned actors")
        for actor in list(self.actor_list):
            actor.destroy()

    def get_carla_world(self):
        return self.world

    def __getattr__(self, name): # 缺失的方法去找world
        """Relay missing methods to underlying carla object"""
        return getattr(self.world, name)

    
# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(CarlaActorBase):
    def __init__(self, world, transform = carla.Transform(), parent_actor = None):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, transform, attach_to=self._parent.get_carla_actor())
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(CarlaActorBase):
    def __init__(self, world, transform = carla.Transform(), parent_actor = None):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, transform, attach_to=self._parent.get_carla_actor())
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)