import numpy as np

G = 9.8  # 重力加速度 (m/s^2)
VELOCITY_SMOKE_SINK = 3.0  # 烟幕云团下沉速度 (m/s) [cite: 5]


def calculate_smoke_center_pos(
        drone_initial_pos: np.ndarray,
        drone_velocity_vec: np.ndarray,
        time_to_release: float,
        time_to_detonate: float,
        time_after_detonation: float
) -> np.ndarray:
    """
    计算在起爆后特定时刻，烟幕云团球心的三维坐标。

    Args:
        drone_initial_pos (np.ndarray): 无人机初始位置坐标 [x, y, z]。
        drone_velocity_vec (np.ndarray): 无人机飞行的速度向量 [vx, vy, vz]。
        time_to_release (float): 无人机从接到指令到投放烟幕弹所需的时间 (s)。
        time_to_detonate (float): 烟幕弹从投放到起爆所需的时间 (s)。
        time_after_detonation (float): 烟幕弹起爆后经过的时间 (s)。

    Returns:
        np.ndarray: 该时刻烟幕云团球心的坐标 [x, y, z]。
    """
    # 1. 计算无人机投放烟幕弹时的位置
    # 无人机是等高度匀速直线飞行
    release_pos = drone_initial_pos + drone_velocity_vec * time_to_release

    # 2. 计算烟幕弹起爆时的位置
    # 烟幕弹脱离后做平抛运动（初速度为无人机的速度）
    # 水平位移
    horizontal_displacement = drone_velocity_vec * time_to_detonate
    # 垂直位移（只受重力影响）
    vertical_displacement = np.array([0, 0, -0.5 * G * (time_to_detonate ** 2)])

    detonation_pos = release_pos + horizontal_displacement + vertical_displacement

    # 3. 计算起爆后，由于下沉导致的最终烟幕球心位置
    # 烟幕云团匀速下沉
    sinking_displacement = np.array([0, 0, -VELOCITY_SMOKE_SINK * time_after_detonation])

    final_smoke_pos = detonation_pos + sinking_displacement

    return final_smoke_pos


def calculate_missile_pos(
        missile_initial_pos: np.ndarray,
        target_pos: np.ndarray,
        missile_speed: float,
        time: float
) -> np.ndarray:
    """
    计算给定时刻导弹的三维坐标。

    Args:
        missile_initial_pos (np.ndarray): 导弹的初始位置坐标 [x, y, z]。
        target_pos (np.ndarray): 导弹的目标位置坐标 [x, y, z]。
        missile_speed (float): 导弹的飞行速度 (m/s)。
        time (float): 从发现导弹开始经过的时间 (s)。

    Returns:
        np.ndarray: 该时刻导弹的坐标 [x, y, z]。
    """
    # 1. 计算导弹飞行的方向向量（单位向量）
    direction_vec = target_pos - missile_initial_pos
    distance_to_target = np.linalg.norm(direction_vec)
    unit_direction_vec = direction_vec / distance_to_target

    # 2. 计算在该时间段内飞行的距离
    distance_traveled = missile_speed * time

    # 3. 计算当前时刻的导弹位置
    current_missile_pos = missile_initial_pos + unit_direction_vec * distance_traveled

    return current_missile_pos