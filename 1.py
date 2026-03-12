import numpy as np
#from utils import *
from utils import *
# --- 问题中给出的常量 ---
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


def is_point_in_sphere(point: np.ndarray,
                          sphere_center: np.ndarray,
                          sphere_radius: float) -> bool:
    """
    判断一个点是否在一个球体内部（包含球面），输入为 NumPy 数组。

    此函数利用 NumPy 的向量化操作来高效计算，性能优于使用 Python 列表的循环。

    Args:
        point (np.ndarray): 要检查的点的坐标 (NumPy 数组)。
        sphere_center (np.ndarray): 球心的坐标 (NumPy 数组)。
        sphere_radius (float): 球的半径。

    Returns:
        bool: 如果点在球体内部或球面上，则返回 True；否则返回 False。

    Raises:
        ValueError: 如果点和球心的形状不匹配。
    """
    # 确保输入的形状一致
    if point.shape != sphere_center.shape:
        raise ValueError("点和球心的坐标数组形状必须相同。")

    # 1. 计算点和球心的向量差 (向量化操作)
    diff_vector = point - sphere_center

    # 2. 计算距离的平方 (向量化操作)
    # np.dot(vector, vector) 是计算向量内积，等价于向量各元素平方和。
    # 这比 np.sum(diff_vector**2) 的效率通常更高。
    squared_distance = np.dot(diff_vector, diff_vector)

    # 3. 计算半径的平方
    squared_radius = sphere_radius ** 2

    # 4. 比较并返回结果
    return squared_distance <= squared_radius


if __name__=="__main__":
    M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])

    drone_initial_pos = np.array([17800.0, 0.0, 1800.0])

    time = 0
    for time_after_denote in [i * 0.01 for i in range(2001)]:
        m_pos = calculate_missile_pos(M1_INITIAL_POS, np.array([0, 0, 0]), 300, time_after_denote + 1.5 + 3.6)
        d_pos = calculate_smoke_center_pos(drone_initial_pos, np.array([-120, 0, 0]), 1.5, 3.6, time_after_denote)
        if (is_point_in_sphere(m_pos, d_pos, 10)):
            time += 0.01
            print(m_pos, d_pos)
            print(time_after_denote)
            continue
        if(check_containment(m_pos,d_pos)):
            time += 0.01
            print(m_pos, d_pos)
            print(time_after_denote)

    print(time)
