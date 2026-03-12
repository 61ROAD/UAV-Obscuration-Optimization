import numpy as np
from scipy.optimize import differential_evolution
import time as timer
import itertools  # 导入itertools用于创建笛卡尔积
# 确保您的环境中存在 get_pos 和 utils 这两个文件
from get_pos import *
from utils import *
import matplotlib.pyplot as plt


def calculate_total_shelter_time(params):
    """
    目标函数：根据给定的决策变量，计算总的遮蔽时间。
    此版本中无人机飞行速度和方向都可变。
    """
    # 1. 从输入参数中解析决策变量 (现在是4个)
    drone_speed, drone_direction_angle, time_to_release, time_to_detonate = params

    # 2. 构建无人机的速度向量 (使用角度)
    vx = drone_speed * np.cos(drone_direction_angle)
    vy = drone_speed * np.sin(drone_direction_angle)
    drone_velocity_vec = np.array([vx, vy, 0])

    # --- 仿真参数 ---
    time_step = 0.1
    total_shelter_time = 0

    # 3. 运行仿真循环 (与之前相同)
    M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
    M1_TARGET_POS = np.array([0.0, 0.0, 0.0])
    M1_SPEED = 300.0
    drone_initial_pos = np.array([17800.0, 0.0, 1800.0])

    t_start_smoke = time_to_release + time_to_detonate
    for t in np.arange(0, 21, time_step):
        missile_pos = calculate_missile_pos(M1_INITIAL_POS, M1_TARGET_POS, M1_SPEED, t + t_start_smoke)
        smoke_pos = calculate_smoke_center_pos(drone_initial_pos, drone_velocity_vec,
                                               time_to_release, time_to_detonate,
                                               t)
        if is_point_in_sphere(missile_pos, smoke_pos, 10):
            total_shelter_time += time_step
        elif check_containment(missile_pos, smoke_pos):
            total_shelter_time += time_step

    return -total_shelter_time


def run_two_stage_optimization():
    """
    执行两阶段优化：
    1. 粗粒度的网格搜索，使用自动生成的数百组工况评估，确定最佳角度大致范围。
    2. 在最佳范围内进行精细化的差分进化优化。
    """
    # --- 首先，计算出完整的、最大的角度搜索范围 ---
    drone_pos_xy = np.array([17800.0, 0.0])
    real_target_pos_xy = np.array([0.0, 200.0])
    missile_initial_pos_xy = np.array([20000.0, 0.0])

    angle_to_target = np.arctan2(real_target_pos_xy[1] - drone_pos_xy[1], real_target_pos_xy[0] - drone_pos_xy[0])
    angle_to_missile = np.arctan2(missile_initial_pos_xy[1] - drone_pos_xy[1],
                                  missile_initial_pos_xy[0] - drone_pos_xy[0])

    full_angle_bounds_rad = (min(angle_to_missile, angle_to_target),np.pi) #max(angle_to_missile, angle_to_target))

    # ==================== 阶段一：网格搜索 (自动生成多工况) ====================
    print("=" * 60)
    print("阶段一：开始进行角度范围的网格搜索 (自动生成多工况)...")
    print(f"完整搜索扇面: ({np.rad2deg(full_angle_bounds_rad[0]):.2f}°, {np.rad2deg(full_angle_bounds_rad[1]):.2f}°)")

    # --- 网格搜索的可调参数 ---
    num_grids = 20  # 将整个角度范围分成20个网格

    # ==================== 自动生成100+组工况 ====================
    # 为每个参数定义采样点的数量
    num_speed_samples = 5
    num_release_time_samples = 5
    num_detonate_time_samples = 5

    # 使用np.linspace在各自的边界内生成等间距的采样点
    speed_points = np.linspace(70, 140, num_speed_samples)
    release_time_points = np.linspace(0.1, 40.0, num_release_time_samples)
    detonate_time_points = np.linspace(0.1, 40.0, num_detonate_time_samples)

    # 使用itertools.product创建所有参数组合的笛卡尔积
    # 这将生成 5 * 5 * 5 = 125 组测试工况
    scenarios = list(itertools.product(speed_points, release_time_points, detonate_time_points))

    print(f"自动生成 {len(scenarios)} 组工况来评估每个角度的潜力。")
    # ==========================================================

    best_grid_score = -1
    best_grid_center_angle_rad = 0

    # 遍历每个网格
    grid_step = (full_angle_bounds_rad[1] - full_angle_bounds_rad[0]) / num_grids
    for i in range(num_grids):
        current_center_angle_rad = full_angle_bounds_rad[0] + (i + 0.5) * grid_step
        max_score_for_this_grid = -1

        # 用所有自动生成的工况测试当前角度
        for speed, release_time, detonate_time in scenarios:
            params_for_grid_search = [speed, current_center_angle_rad, release_time, detonate_time]
            score = -calculate_total_shelter_time(params_for_grid_search)
            if score > max_score_for_this_grid:
                max_score_for_this_grid = score

        # 打印进度，每5个网格打印一次，避免刷屏
        if (i + 1) % 5 == 0 or i == 0 or i == num_grids - 1:
            print(f"  测试网格 {i + 1}/{num_grids} (角度: {np.rad2deg(current_center_angle_rad):.2f}°): "
                  f"潜力得分 = {max_score_for_this_grid:.4f}s")

        if max_score_for_this_grid > best_grid_score:
            best_grid_score = max_score_for_this_grid
            best_grid_center_angle_rad = current_center_angle_rad

    print(f"\n网格搜索完成！最有潜力的区域中心角度为: {np.rad2deg(best_grid_center_angle_rad):.2f}°")

    # ==================== 阶段二：精细化优化 ====================
    print("=" * 60)
    print("阶段二：在最有潜力的区域内进行差分进化优化...")

    sub_region_width_degrees = 30.0
    sub_region_width_rad = np.deg2rad(sub_region_width_degrees)

    refined_angle_bounds_rad = (
        best_grid_center_angle_rad - sub_region_width_rad / 2,
        best_grid_center_angle_rad + sub_region_width_rad / 2
    )
    refined_angle_bounds_rad = (
        max(refined_angle_bounds_rad[0], full_angle_bounds_rad[0]),
        min(refined_angle_bounds_rad[1], full_angle_bounds_rad[1])
    )

    print(
        f"新的精细化搜索角度范围: ({np.rad2deg(refined_angle_bounds_rad[0]):.2f}°, {np.rad2deg(refined_angle_bounds_rad[1]):.2f}°)")
    print("-" * 60)

    bounds = [
        (70, 140),
        refined_angle_bounds_rad,
        (0.1, 40.0),
        (0.1, 40.0)
    ]

    print("开始进行优化，这可能需要几分钟时间...")
    start_time = timer.time()

    result = differential_evolution(
        calculate_total_shelter_time,
        bounds,
        strategy='best1bin', maxiter=5000, popsize=50, tol=0.01,
        mutation=(0.5, 1), recombination=0.7, disp=True, workers=-1, seed=1025,
    )

    end_time = timer.time()
    print(f"优化完成，耗时: {end_time - start_time:.2f} 秒")

    best_params = result.x
    max_shelter_time = -result.fun

    print("\n--- 最终优化结果 ---")
    print(f"最大遮蔽时间: {max_shelter_time:.4f} 秒")
    print("最优参数组合:")
    print(f"  - 无人机飞行速度: {best_params[0]:.4f} m/s")
    print(f"  - 无人机飞行角度: {np.rad2deg(best_params[1]):.4f} 度")
    print(f"  - 投放准备时间: {best_params[2]:.4f} s")
    print(f"  - 投放后起爆时间: {best_params[3]:.4f} s")

    best_speed, best_angle = best_params[0], best_params[1]
    best_vx = best_speed * np.cos(best_angle)
    best_vy = best_speed * np.sin(best_angle)
    print("\n推算出的最优飞行数据:")
    print(f"  - 无人机速度向量: [{best_vx:.2f}, {best_vy:.2f}, 0.0]")
    drone_initial_pos = np.array([17800.0, 0.0, 1800.0])
    release_pos = drone_initial_pos + np.array([best_vx, best_vy, 0]) * best_params[2]
    print(f"  - 烟幕弹投放点坐标: [{release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f}]")


if __name__ == "__main__":
    run_two_stage_optimization()