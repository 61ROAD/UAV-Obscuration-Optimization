import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import time

# (之前已修正，无需改动) 导入您的工具函数
from utils import check_containment, is_point_in_sphere
from get_pos import calculate_missile_pos, calculate_smoke_center_pos

# --- 1. 全局常量和初始条件设定 (无需改动) ---
M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
M1_SPEED = 300.0
FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])

UAV_INITIAL_POSITIONS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}

TRUE_TARGET_CENTER_XY = np.array([0.0, 200.0])
TRUE_TARGET_RADIUS = 7.0
TRUE_TARGET_HEIGHT = 10.0
SMOKE_RADIUS = 10.0
SMOKE_SINK_SPEED = 3.0
SMOKE_DURATION = 20.0
GRAVITY = 9.8

target_points = []
for z in [0, TRUE_TARGET_HEIGHT / 2, TRUE_TARGET_HEIGHT]:
    target_points.append(np.array([TRUE_TARGET_CENTER_XY[0], TRUE_TARGET_CENTER_XY[1], z]))
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        x = TRUE_TARGET_CENTER_XY[0] + TRUE_TARGET_RADIUS * np.cos(angle)
        y = TRUE_TARGET_CENTER_XY[1] + TRUE_TARGET_RADIUS * np.sin(angle)
        target_points.append(np.array([x, y, z]))


# --- 2. 核心函数 (无需改动) ---
def check_shelter_for_target(missile_pos, active_smoke_spheres):
    if not active_smoke_spheres:
        return False
    for smoke_center, smoke_radius in active_smoke_spheres:
        sc = np.array(smoke_center, dtype=float)
        try:
            if all(is_point_in_sphere(tp, sc, smoke_radius) for tp in target_points):
                return True
            if check_containment(missile_pos, sc, smoke_radius):
                return True
        except Exception:
            continue
    return False


def calculate_shelter_time_for_strategy(params):
    strategies = []
    uav_ids = ['FY1', 'FY2', 'FY3']
    for i in range(3):
        uav_id = uav_ids[i]
        speed, angle, t_release, t_detonate = params[i * 4: i * 4 + 4]
        strategies.append({
            'uav_initial_pos': UAV_INITIAL_POSITIONS[uav_id],
            'drone_velocity_vec': np.array([speed * np.cos(angle), speed * np.sin(angle), 0]),
            't_release': t_release,
            't_detonate': t_detonate,
            'detonation_time': t_release + t_detonate,
            'original_index': i
        })

    total_shelter_time = 0
    individual_shelter_times = [0.0, 0.0, 0.0]
    time_step = 0.1
    detonation_times = [s['detonation_time'] for s in strategies]
    if not detonation_times: return 0.0, individual_shelter_times
    sim_start_time = min(detonation_times)
    sim_end_time = max(detonation_times) + SMOKE_DURATION
    sorted_strategies = sorted(strategies, key=lambda s: s['detonation_time'])

    for t in np.arange(sim_start_time, sim_end_time, time_step):
        missile_pos = calculate_missile_pos(M1_INITIAL_POS, FALSE_TARGET_POS, M1_SPEED, t)
        active_smoke_spheres = []
        active_strategies = []
        for strat in sorted_strategies:
            time_since_detonation = t - strat['detonation_time']
            if 0 <= time_since_detonation <= SMOKE_DURATION:
                smoke_center = calculate_smoke_center_pos(
                    drone_initial_pos=strat['uav_initial_pos'],
                    drone_velocity_vec=strat['drone_velocity_vec'],
                    time_to_release=strat['t_release'],
                    time_to_detonate=strat['t_detonate'],
                    time_after_detonation=time_since_detonation
                )
                active_smoke_spheres.append((smoke_center, SMOKE_RADIUS))
                active_strategies.append(strat)
        if check_shelter_for_target(missile_pos, active_smoke_spheres):
            total_shelter_time += time_step
            if active_strategies:
                contributor_index = active_strategies[0]['original_index']
                individual_shelter_times[contributor_index] += time_step
    return total_shelter_time, individual_shelter_times


def objective_function(params):
    score, _ = calculate_shelter_time_for_strategy(params)
    return -score


# --- 3. 新增的优化执行函数 (无需改动) ---
def run_optimization(angle_bounds_fy1, angle_bounds_fy2, angle_bounds_fy3):
    """
    将单次差分进化优化封装成一个函数。
    接收三个无人机的角度边界作为输入。
    """
    bounds = [
        # --- FY1 (速度和时间边界是固定的) ---
        (70, 140),
        angle_bounds_fy1,  # 角度 a1 (动态)
        (0.1, 10.0),
        (0.1, 10.0),
        # --- FY2 ---
        (70, 140),
        angle_bounds_fy2,  # 角度 a2 (动态)
        (0.1, 20.0),
        (0.1, 10.0),
        # --- FY3 ---
        (70, 140),
        angle_bounds_fy3,  # 角度 a3 (动态)
        (0.1, 40.0),
        (0.1, 10.0)
    ]

    print("正在使用差分进化算法进行优化...")
    result = differential_evolution(
        objective_function,
        bounds,
        strategy='best1bin', maxiter=200, popsize=20,
        tol=0.01, mutation=(0.5, 1), recombination=0.7,
        disp=True, workers=-1, seed=42
    )
    return result


# --- 4. 修改后的主执行函数 (无需改动) ---
if __name__ == "__main__":
    print("开始执行基于二分搜索的迭代优化策略...")

    # 步骤1: 计算每个无人机的初始搜索扇区
    missile_pos_xy = M1_INITIAL_POS[:2]
    false_target_pos_xy = FALSE_TARGET_POS[:2]

    # 初始化二分搜索的边界
    current_angle_bounds = {}
    for uav_id, pos in UAV_INITIAL_POSITIONS.items():
        uav_pos_xy = pos[:2]
        delta_missile = missile_pos_xy - uav_pos_xy
        angle_to_missile = np.arctan2(delta_missile[1], delta_missile[0])
        delta_target = false_target_pos_xy - uav_pos_xy
        angle_to_target = np.arctan2(delta_target[1], delta_target[0])

        lower_bound = min(angle_to_missile, angle_to_target)
        upper_bound = max(angle_to_missile, angle_to_target)
        current_angle_bounds[uav_id] = (lower_bound, upper_bound)

        print(f"无人机 {uav_id} 初始搜索扇区: ({np.rad2deg(lower_bound):.2f}, {np.rad2deg(upper_bound):.2f}) 度")

    # 步骤2: 设置迭代参数
    total_iterations = 4  # 1次大范围搜索 + 3次二分
    global_best_result = None

    # 步骤3: 开始二分迭代优化循环
    for i in range(total_iterations):
        print("\n" + "=" * 25 + f" 第 {i + 1}/{total_iterations} 轮二分优化 " + "=" * 25)

        # 打印本轮优化的边界
        for uav_id, bounds in current_angle_bounds.items():
            print(f"    - {uav_id} 本轮搜索角度范围: ({np.rad2deg(bounds[0]):.2f}, {np.rad2deg(bounds[1]):.2f}) 度")

        # 执行优化
        start_time = time.time()
        result = run_optimization(
            current_angle_bounds['FY1'],
            current_angle_bounds['FY2'],
            current_angle_bounds['FY3']
        )
        end_time = time.time()

        print(f"第 {i + 1} 轮优化完成！耗时: {end_time - start_time:.2f} 秒，本轮最优得分: {-result.fun:.4f}")

        # 更新全局最优解
        if global_best_result is None or result.fun < global_best_result.fun:
            print("  - 发现新的全局最优解！")
            global_best_result = result

        # --- 核心二分逻辑: 为下一轮更新每个无人机的搜索边界 ---
        if i < total_iterations - 1:  # 最后一轮不需要再计算
            best_params = result.x
            best_angles = {
                'FY1': best_params[1],
                'FY2': best_params[5],
                'FY3': best_params[9]
            }

            next_angle_bounds = {}
            for uav_id, bounds in current_angle_bounds.items():
                low, high = bounds
                mid = (low + high) / 2.0
                best_angle_for_uav = best_angles[uav_id]

                if best_angle_for_uav > mid:
                    next_angle_bounds[uav_id] = (mid, high)  # 搜索上半区
                else:
                    next_angle_bounds[uav_id] = (low, mid)  # 搜索下半区

            current_angle_bounds = next_angle_bounds  # 更新边界以供下一轮使用

    # =================================================================
    # --- 5. 结果处理与输出 (已按您的图片格式要求修改) ---
    # =================================================================
    print("\n" + "=" * 60)
    print("                  所有迭代完成，最终最优协同策略")
    print("=" * 60)

    best_params = global_best_result.x
    max_shelter_time, individual_times = calculate_shelter_time_for_strategy(best_params)

    print(f"\n最大总遮蔽时间: {-global_best_result.fun:.4f} 秒\n")

    # 用于存储最终要写入Excel的数据
    output_data = []
    # 结果表格的最终列顺序
    final_columns = [
        '无人机编号', '无人机运动方向', '无人机运动速度 (m/s)',
        '烟幕干扰弹投放点的x坐标(m)', '烟幕干扰弹投放点的y坐标(m)', '烟幕干扰弹投放点的z坐标(m)',
        '烟幕干扰弹起爆点的x坐标(m)', '烟幕干扰弹起爆点的y坐标(m)', '烟幕干扰弹起爆点的z坐标(m)',
        '有效干扰时长 (s)'
    ]

    uav_ids = ['FY1', 'FY2', 'FY3']
    for i in range(3):
        uav_id = uav_ids[i]
        speed, angle_rad, t_release, t_detonate = best_params[i * 4: i * 4 + 4]
        angle_deg = np.rad2deg(angle_rad)
        uav_initial_pos = UAV_INITIAL_POSITIONS[uav_id]
        velocity_vec = np.array([speed * np.cos(angle_rad), speed * np.sin(angle_rad), 0])

        # 计算投放点和起爆点坐标
        release_pos = uav_initial_pos + velocity_vec * t_release
        detonation_pos = calculate_smoke_center_pos(
            drone_initial_pos=uav_initial_pos,
            drone_velocity_vec=velocity_vec,
            time_to_release=t_release,
            time_to_detonate=t_detonate,
            time_after_detonation=0  # 爆炸瞬间
        )

        # --- 在控制台打印详细信息 (保留原有格式) ---
        print(f"--- 无人机 {uav_id} 策略 ---")
        print(f"  - 飞行速度: {speed:.4f} m/s")
        print(f"  - 飞行方向: {angle_deg:.4f} 度")
        print(f"  - 投放准备时间: {t_release:.4f} s")
        print(f"  - 起爆延迟时间: {t_detonate:.4f} s")
        print(f"  - 投放点坐标: ({release_pos[0]:.2f}, {release_pos[1]:.2f}, {release_pos[2]:.2f})")
        print(f"  - 起爆点坐标: ({detonation_pos[0]:.2f}, {detonation_pos[1]:.2f}, {detonation_pos[2]:.2f})")
        print(f"  - 本弹贡献时长: {individual_times[i]:.4f} s\n")

        # --- 准备用于写入Excel的一行数据 ---
        output_data.append({
            '无人机编号': uav_id,
            '无人机运动方向': f"{angle_deg:.4f} 度",
            '无人机运动速度 (m/s)': speed,
            '烟幕干扰弹投放点的x坐标(m)': release_pos[0],
            '烟幕干扰弹投放点的y坐标(m)': release_pos[1],
            '烟幕干扰弹投放点的z坐标(m)': release_pos[2],
            '烟幕干扰弹起爆点的x坐标(m)': detonation_pos[0],
            '烟幕干扰弹起爆点的y坐标(m)': detonation_pos[1],
            '烟幕干扰弹起爆点的z坐标(m)': detonation_pos[2],
            '有效干扰时长 (s)': individual_times[i]
        })

    # 创建DataFrame并按照最终列顺序排列
    df = pd.DataFrame(output_data)
    df = df[final_columns]

    # 定义输出文件名
    output_filename = 'uav_strategy_results.xlsx'
    # 保存到Excel
    df.to_excel(output_filename, index=False, float_format="%.4f")

    print(f"最终策略已成功保存到文件: {output_filename}")