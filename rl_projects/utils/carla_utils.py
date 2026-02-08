import numpy as np
import os
import subprocess
import time
import psutil

def launch_carla(args):
    CARLA_ROOT = os.environ["CARLA_ROOT"]
    carla_path = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
    port = args.port
    launch_command = [carla_path]
    launch_command += ['-RenderOffScreen']
    launch_command += ['-nosound']
    launch_command += [f'-carla-world-port={port}']
    launch_command += [f'-graphicsadapter=0']
    launch_command = " ".join(launch_command)
    launch_command = f"su - carla -c \"{launch_command}\""
    print("Running command:")
    print("".join(launch_command))
    carla_process = subprocess.Popen(launch_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("Waiting for CARLA to initialize\n")
    time.sleep(20)
    if carla_process.poll() is None:
        print("CARLA process is running")
    else:
        print("CARLA process has exited with code", carla_process.poll())


def kill_existing_carla():
    """更彻底的进程清理"""
    print("Terminating CARLA processes...")
    
    # 第一步：标准终止
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], check=False)
    subprocess.run(["pkill", "-9", "-f", "carla-rpc-port"], check=False)
    
    # 第二步：精确查找残留进程
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'CarlaUE4' in ''.join(proc.info['cmdline'] or []):
                proc.kill()
            if 'carla' in proc.info['name'].lower():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # 第三步：清理共享内存（关键步骤）
    subprocess.run(["ipcs", "-m"], check=False)
    subprocess.run(["ipcrm", "-a"], check=False)  # 清除所有IPC资源
    print("CARLA processes and shared memory cleared")


def get_rotation_matrix(yaw):
    """
    根据偏航角计算旋转矩阵
    :param yaw: 偏航角（度）
    :return: 旋转矩阵
    """
    yaw_rad = np.radians(-yaw)
    rotation_matrix = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    return rotation_matrix


def world_to_vehicle(world_point, vehicle_location, rotation_matrix):
    """
    将全局坐标系下的点转换到自车坐标系
    :param world_point: 全局坐标系下的点 [x, y, z]
    :param vehicle_location: 自车在全局坐标系下的位置
    :param rotation_matrix: 旋转矩阵
    :return: 自车坐标系下的点
    """
    # 平移
    translated_point = [
        world_point[0] - vehicle_location.x,
        world_point[1] - vehicle_location.y,
        world_point[2] - vehicle_location.z
    ]
    # 旋转
    vehicle_point = np.dot(rotation_matrix, translated_point)
    return vehicle_point


def convert_waypoints_to_vehicle(waypoints, vehicle_transform):
    """
    将 GlobalRoutePlanner 生成的路点转换到自车坐标系
    :param waypoints: GlobalRoutePlanner 生成的路点列表
    :param vehicle_transform: 自车的 transform 对象
    :return: 自车坐标系下的路点列表
    """
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    yaw = vehicle_rotation.yaw
    rotation_matrix = get_rotation_matrix(yaw)

    vehicle_waypoints = []
    for waypoint in waypoints:
        world_point = [
            waypoint[0].transform.location.x,
            waypoint[0].transform.location.y,
            waypoint[0].transform.location.z
        ]
        vehicle_point = world_to_vehicle(world_point, vehicle_location, rotation_matrix)
        vehicle_waypoints.append(vehicle_point)

    return vehicle_waypoints