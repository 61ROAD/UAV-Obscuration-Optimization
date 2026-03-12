import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import time as timer
from tqdm import tqdm

# 假设您的 get_pos.py 和 utils.py 文件与此脚本在同一目录下
from get_pos import *
from utils import *

# --- 全局常量设定 (无需修改) ---
M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
M1_TARGET_POS = np.array([0.0, 0.0, 0.0])
M1_SPEED = 300.0
DRONE_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])
MISSILE_TOTAL_DISTANCE = np.linalg.norm(M1_INITIAL_POS - M1_TARGET_POS)
MAX_MISSILE_FLIGHT_TIME = MISSILE_TOTAL_DISTANCE / M1_SPEED


# --- 核心函数 (无需修改) ---
def calculate_shelter_time_for_strategy(strategy_params, time_step=0.1):
    """
    【核心模拟引擎】
    计算总遮蔽时间，并分别记录每枚烟幕弹的贡献时长。
    """
    drone_speed = strategy_params['speed']
    drone_angle = strategy_params['angle']
    bombs = strategy_params['bombs']
    individual_shelter_times = [0.0] * len(bombs)
    vx = drone_speed * np.cos(drone_angle)
    vy = drone_speed * np.sin(drone_angle)
    drone_velocity_vec = np.array([vx, vy, 0])

    if not bombs:
        return 0.0, individual_shelter_times

    detonation_times = [b['release'] + b['detonate_delay'] for b in bombs]
    if not detonation_times:
        return 0.0, individual_shelter_times

    total_shelter_time = 0
    sim_start_time = min(detonation_times)
    sim_end_time = min(max(detonation_times) + 20.0, MAX_MISSILE_FLIGHT_TIME)

    if sim_start_time >= sim_end_time:
        return 0.0, individual_shelter_times

    for t in np.arange(sim_start_time, sim_end_time, time_step):
        missile_pos = calculate_missile_pos(M1_INITIAL_POS, M1_TARGET_POS, M1_SPEED, t)
        is_sheltered_this_step = False
        sorted_bombs_with_indices = sorted(enumerate(bombs), key=lambda x: x[1]['release'])

        for original_index, bomb in sorted_bombs_with_indices:
            detonation_time = bomb['release'] + bomb['detonate_delay']
            time_since_detonation = t - detonation_time
            if 0 <= time_since_detonation <= 20.0:
                smoke_pos = calculate_smoke_center_pos(
                    DRONE_INITIAL_POS, drone_velocity_vec,
                    bomb['release'], bomb['detonate_delay'],
                    time_since_detonation
                )
                if is_point_in_sphere(missile_pos, smoke_pos, 10) or \
                        check_containment(missile_pos, smoke_pos):
                    is_sheltered_this_step = True
                    individual_shelter_times[original_index] += time_step
                    break
        if is_sheltered_this_step:
            total_shelter_time += time_step
    return total_shelter_time, individual_shelter_times


def objective_function_unified(params, refined_angle_bounds):
    """
    【统一优化目标函数】
    """
    drone_speed, drone_angle, r1, d1, r2, d2, r3, d3 = params
    if not (refined_angle_bounds[0] <= drone_angle <= refined_angle_bounds[1]):
        return 2000.0
    if r2 < r1 + 1.0 or r3 < r2 + 1.0:
        return 1000.0
    strategy = {
        'speed': drone_speed,
        'angle': drone_angle,
        'bombs': [
            {'release': r1, 'detonate_delay': d1},
            {'release': r2, 'detonate_delay': d2},
            {'release': r3, 'detonate_delay': d3},
        ]
    }
    score, _ = calculate_shelter_time_for_strategy(strategy)
    return -score


# --- 新增的优化执行函数 ---
def run_optimization(angle_bounds):
    """
    将单次差分进化优化封装成一个函数。
    接收无人机的角度边界作为输入。
    """
    bounds = [
        (70, 140),  # 无人机速度
        angle_bounds,  # 无人机角度 (动态)
        (0.1, 5.0), (0.1, 5.0),  # 弹1: 投放时间, 起爆延迟 (给予更合理的范围)
        (1.1, 15.0), (0.1, 5.0),  # 弹2: 投放时间, 起爆延迟
        (2.1, 25.0), (0.1, 5.0),  # 弹3: 投放时间, 起爆延迟
    ]

    print("  正在使用差分进化算法进行优化...")
    result = differential_evolution(
        objective_function_unified,
        bounds,
        args=(angle_bounds,),
        strategy='best1bin',
        maxiter=120, popsize=300, tol=0.01,
        mutation=(0.5, 1), recombination=0.7,
        disp=True, workers=-1, seed=2025
    )
    return result


# --- 主执行函数 (已按您要求的“真·二分搜索”逻辑重写) ---
def run_binary_search_optimization():
    """
    执行基于二分法思想的迭代优化策略。
    """
    # 步骤1: 计算初始搜索扇区 (与之前相同)
    drone_pos_xy = DRONE_INITIAL_POS[:2]
    missile_pos_xy = M1_INITIAL_POS[:2]
    false_target_pos_xy = M1_TARGET_POS[:2]
    delta_missile = missile_pos_xy - drone_pos_xy
    angle_to_missile = np.arctan2(delta_missile[1], delta_missile[0])
    delta_target = false_target_pos_xy - drone_pos_xy
    angle_to_target = np.arctan2(delta_target[1], delta_target[0])

    # 初始化二分搜索的边界
    current_low_angle = min(angle_to_missile, angle_to_target)
    current_high_angle = max(angle_to_missile, angle_to_target)

    print(f"无人机初始搜索扇区已确定: ({np.rad2deg(current_low_angle):.2f}, {np.rad2deg(current_high_angle):.2f}) 度")

    # 步骤2: 设置迭代参数
    total_iterations = 4  # 总共执行4轮优化
    global_best_result = None

    # 步骤3: 开始二分迭代优化循环
    for i in range(total_iterations):
        print("\n" + "= " * 25 + f" 第 {i + 1}/{total_iterations} 轮二分优化 " + "= " * 25)

        # 定义本轮的搜索边界
        current_bounds = (current_low_angle, current_high_angle)
        print(f"    - 本轮搜索角度范围: ({np.rad2deg(current_bounds[0]):.2f}, {np.rad2deg(current_bounds[1]):.2f}) 度")

        # 执行优化
        start_time = timer.time()
        result = run_optimization(current_bounds)
        end_time = timer.time()
        print(f"第 {i + 1} 轮优化完成！耗时: {end_time - start_time:.2f} 秒，本轮最优得分: {-result.fun:.4f}")

        # 更新全局最优解 (很重要，因为优化有随机性，窄区间不一定总能重复找到最优解)
        if global_best_result is None or result.fun < global_best_result.fun:
            print("  - 发现新的全局最优解！")
            global_best_result = result

        # --- 核心二分逻辑：为下一轮更新搜索边界 ---
        best_angle_this_run = result.x[1]
        mid_point_angle = (current_low_angle + current_high_angle) / 2.0

        if best_angle_this_run > mid_point_angle:
            # 最优解在区间的上半部分，下一轮搜索上半部分
            current_low_angle = mid_point_angle
            print \
                (f"  - 最优解位于上半区，下一轮将搜索 ({np.rad2deg(current_low_angle):.2f}, {np.rad2deg(current_high_angle):.2f})")
        else:
            # 最优解在区间的下半部分，下一轮搜索下半部分
            current_high_angle = mid_point_angle
            print \
                (f"  - 最优解位于下半区，下一轮将搜索 ({np.rad2deg(current_low_angle):.2f}, {np.rad2deg(current_high_angle):.2f})")

    # ============================================================
    # 最终结果展示 (已修改，增加坐标计算和打印)
    # ============================================================
    print("\n" + "= " * 60)
    print("                所有迭代完成，最终最优策略如下：")
    print("= " * 60)

    best_params = global_best_result.x
    final_strategy = {
        'speed': best_params[0],
        'angle': best_params[1],
        'bombs': [
            {'release': best_params[2], 'detonate_delay': best_params[3]},
            {'release': best_params[4], 'detonate_delay': best_params[5]},
            {'release': best_params[6], 'detonate_delay': best_params[7]},
        ]
    }

    max_shelter_time, individual_times = calculate_shelter_time_for_strategy(final_strategy)
    for i, bomb in enumerate(final_strategy['bombs']):
        bomb['contribution'] = individual_times[i]

    print(f"最大总遮蔽时间 (三枚弹): {max_shelter_time:.4f} 秒\n")
    print("无人机飞行策略:")
    print(f"  - 飞行速度: {final_strategy['speed']:.4f} m/s")
    print(f"  - 飞行角度: {np.rad2deg(final_strategy['angle']):.4f} 度\n")

    # --- 新增代码块：计算并打印坐标 ---
    # 首先计算出最终的无人机速度向量
    final_vx = final_strategy['speed'] * np.cos(final_strategy['angle'])
    final_vy = final_strategy['speed'] * np.sin(final_strategy['angle'])
    final_drone_velocity_vec = np.array([final_vx, final_vy, 0])

    print("烟幕弹投放详情:")
    sorted_bombs = sorted(final_strategy['bombs'], key=lambda b: b['release'])
    for i, bomb in enumerate(sorted_bombs):
        # 计算投放点坐标
        release_pos = DRONE_INITIAL_POS + final_drone_velocity_vec * bomb['release']

        # 计算起爆点坐标 (即爆炸发生瞬间，time_since_detonation=0)
        detonation_pos = calculate_smoke_center_pos(
            DRONE_INITIAL_POS,
            final_drone_velocity_vec,
            bomb['release'],
            bomb['detonate_delay'],
            0  # time_since_detonation
        )

        print(f"--- 烟幕弹 {i + 1} (按投放顺序) ---")
        print(f"  - 投放准备时间: {bomb['release']:.4f} s")
        print(f"  - 投放后起爆时间: {bomb['detonate_delay']:.4f} s")
        print(f"  - 投放点坐标: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
        print(f"  - 起爆点坐标: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")
        print(f"  - 本枚导弹贡献时长: {bomb['contribution']:.4f} s")


if __name__ == "__main__":
    print(f"导弹从初始位置 {M1_INITIAL_POS} 飞向目标 {M1_TARGET_POS}。")
    print(f"总飞行距离: {MISSILE_TOTAL_DISTANCE:.2f} m, 飞行速度: {M1_SPEED} m/s。")
    print(f"=> 导弹击中目标所需总时间: {MAX_MISSILE_FLIGHT_TIME:.4f} s。遮蔽时间计算将以此为上限。")
    # 调用新的主函数
    run_binary_search_optimization()