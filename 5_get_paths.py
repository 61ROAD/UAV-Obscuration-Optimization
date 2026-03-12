import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pickle
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# 导入已有的函数
from get_pos import calculate_smoke_center_pos, calculate_missile_pos
from utils import check_containment

# ==================== 常量定义 ====================
MISSILE_SPEED = 300.0  # m/s
DRONE_V_MIN, DRONE_V_MAX = 70.0, 140.0  # m/s
R_TARGET, H_TARGET = 7.0, 10.0  # m
SMOKE_R = 10.0  # m
SMOKE_SINK = 3.0  # m/s
SMOKE_T_EFF = 20.0  # s
DROP_INTERVAL = 1.0  # s
G = 9.8  # m/s^2

# 初始位置
CENTER_TRUE = np.array([0.0, 200.0, 0.0])
CENTER_FALSE = np.array([0.0, 0.0, 0.0])

INIT_M = {
    "M1": np.array([20000, 0, 2000]),
    "M2": np.array([19000, 600, 2100]),
    "M3": np.array([18000, -600, 1900])
}

INIT_D = {
    "FY1": np.array([17800, 0, 1800]),
    "FY2": np.array([12000, 1400, 1400]),
    "FY3": np.array([6000, -3000, 700]),
    "FY4": np.array([11000, 2000, 1800]),
    "FY5": np.array([13000, -2000, 1300])
}

# 时间离散化
DT = 0.1  # 时间步长
T_MAX = 100.0  # 最大仿真时间

# ==================== 数据结构 ====================
@dataclass
class Column:
    """列（投放事件）的数据结构"""
    drone: str
    path_idx: int
    missile: str
    tau: float  # 投放时刻
    delta: float  # 起爆延时
    ton: float  # 生效起始
    toff: float  # 生效截止
    cover_time: float  # 覆盖时长
    velocity: float  # 飞行速度
    theta: float  # 飞行方向角（弧度）
    
@dataclass
class Path:
    """候选路径"""
    drone: str
    velocity: float  # m/s
    theta: float  # 方向角（弧度）
    direction_vec: np.ndarray  # 单位方向向量

# ==================== 辅助函数 ====================
def get_direction_bounds(drone_pos: np.ndarray, missile_pos: np.ndarray) -> Tuple[float, float]:
    """
    获取无人机飞行方向的角度范围（仅考虑x-y平面）
    从无人机位置到假目标方向 到 无人机位置到导弹方向
    """
    # 到假目标的方向
    to_false = CENTER_FALSE[:2] - drone_pos[:2]
    angle_to_false = np.arctan2(to_false[1], to_false[0])
    
    # 到导弹的方向
    to_missile = missile_pos[:2] - drone_pos[:2]
    angle_to_missile = np.arctan2(to_missile[1], to_missile[0])
    
    # 确保角度在[0, 2π]范围内
    angle_to_false = angle_to_false % (2 * np.pi)
    angle_to_missile = angle_to_missile % (2 * np.pi)
    
    # 返回角度范围（可能需要处理跨越0度的情况）
    return angle_to_false, angle_to_missile

def generate_candidate_paths(drone_name: str, num_angles: int = 20, num_velocities: int = 6) -> List[Path]:
    """
    为指定无人机生成候选路径集
    """
    paths = []
    drone_pos = INIT_D[drone_name]
    
    # 速度网格
    velocities = np.linspace(DRONE_V_MIN, DRONE_V_MAX, num_velocities)
    
    # 对每个导弹计算角度范围
    all_angles = []
    for missile_name, missile_pos in INIT_M.items():
        angle_false, angle_missile = get_direction_bounds(drone_pos, missile_pos)
        
        # 生成该范围内的角度
        if abs(angle_missile - angle_false) < np.pi:
            # 不跨越0度
            angles = np.linspace(min(angle_false, angle_missile), 
                               max(angle_false, angle_missile), 
                               num_angles // 3)
        else:
            # 跨越0度的情况
            if angle_false < angle_missile:
                angles1 = np.linspace(0, angle_false, num_angles // 6)
                angles2 = np.linspace(angle_missile, 2*np.pi, num_angles // 6)
                angles = np.concatenate([angles1, angles2])
            else:
                angles1 = np.linspace(0, angle_missile, num_angles // 6)
                angles2 = np.linspace(angle_false, 2*np.pi, num_angles // 6)
                angles = np.concatenate([angles1, angles2])
        
        all_angles.extend(angles)
    
    # 去重并排序
    all_angles = np.unique(all_angles)
    
    # 生成路径
    path_idx = 0
    for v in velocities:
        for theta in all_angles:
            direction_vec = np.array([np.cos(theta), np.sin(theta), 0])
            paths.append(Path(
                drone=drone_name,
                velocity=v,
                theta=theta,
                direction_vec=direction_vec
            ))
            path_idx += 1
    
    return paths

def check_coverage_at_time(smoke_pos: np.ndarray, missile_pos: np.ndarray, 
                          target_center: np.ndarray = CENTER_TRUE) -> bool:
    """
    检查烟幕是否完全遮蔽导弹视线
    """
    return check_containment(missile_pos, smoke_pos, R_TARGET, H_TARGET, 
                           (target_center[0], target_center[1])) or is_point_in_sphere(missile_pos,smoke_pos,10)

def evaluate_column(drone_pos: np.ndarray, path: Path, missile_name: str,
                   tau: float, delta: float) -> Tuple[float, float, float]:
    """
    评估一个投放事件的覆盖效果
    返回: (覆盖开始时间, 覆盖结束时间, 总覆盖时长)
    """
    missile_pos_init = INIT_M[missile_name]
    velocity_vec = path.direction_vec * path.velocity
    
    # 检查烟幕有效期内的覆盖情况
    t_start = tau + delta
    t_end = min(t_start + SMOKE_T_EFF, T_MAX)
    
    coverage_start = None
    coverage_end = None
    
    # 以较小步长检查覆盖
    for t in np.arange(t_start, t_end, DT):
        # 导弹位置
        missile_pos = calculate_missile_pos(missile_pos_init, CENTER_FALSE, 
                                           MISSILE_SPEED, t)
        
        # 烟幕中心位置
        time_after_det = t - (tau + delta)
        if time_after_det >= 0:
            smoke_pos = calculate_smoke_center_pos(
                drone_pos, velocity_vec, tau, delta, time_after_det
            )
            
            # 检查覆盖
            if check_coverage_at_time(smoke_pos, missile_pos):
                if coverage_start is None:
                    coverage_start = t
                coverage_end = t
            elif coverage_start is not None:
                # 覆盖中断
                break
    
    if coverage_start is None:
        return 0, 0, 0
    
    coverage_time = (coverage_end - coverage_start) if coverage_end else 0
    return coverage_start, coverage_end, coverage_time

def search_optimal_release(drone_name: str, path: Path, missile_name: str,
                          min_coverage: float = 0.5) -> List[Column]:
    """
    使用差分进化搜索最优投放参数
    """
    drone_pos = INIT_D[drone_name]
    
    def objective(params):
        tau, delta = params
        _, _, coverage_time = evaluate_column(drone_pos, path, missile_name, tau, delta)
        return -coverage_time  # 最大化覆盖时间
    
    # 约束范围
    bounds = [
        (0.1, 30.0),  # tau: 投放时刻
        (1.0, 10.0)   # delta: 起爆延时
    ]
    
    # 差分进化优化
    result = differential_evolution(
        objective, bounds,
        maxiter=50,
        popsize=15,
        tol=0.01,
        seed=42,
        workers=1
    )
    
    columns = []
    if -result.fun >= min_coverage:  # 如果覆盖时间足够长
        tau, delta = result.x
        ton, toff, coverage_time = evaluate_column(drone_pos, path, missile_name, tau, delta)
        
        if coverage_time >= min_coverage:
            columns.append(Column(
                drone=drone_name,
                path_idx=path.velocity * 1000 + path.theta,  # 简单的索引编码
                missile=missile_name,
                tau=tau,
                delta=delta,
                ton=ton,
                toff=toff,
                cover_time=coverage_time,
                velocity=path.velocity,
                theta=path.theta
            ))
    
    return columns

def generate_initial_columns(drone_name: str, paths: List[Path], 
                            missiles: List[str] = ["M1", "M2", "M3"],
                            max_columns_per_combo: int = 3) -> List[Column]:
    """
    为无人机生成初始列池
    """
    all_columns = []
    
    print(f"为 {drone_name} 生成列...")
    for i, path in enumerate(paths):
        if i % 10 == 0:
            print(f"  处理路径 {i+1}/{len(paths)}")
        
        for missile in missiles:
            # 搜索该组合的最优投放
            columns = search_optimal_release(drone_name, path, missile)
            all_columns.extend(columns[:max_columns_per_combo])
    
    print(f"  生成了 {len(all_columns)} 个有效列")
    return all_columns

# ==================== 主函数 ====================
def main():
    """生成并保存所有无人机的候选列"""
    
    all_columns = {}
    all_paths = {}
    
    for drone_name in INIT_D.keys():
        print(f"\n处理无人机 {drone_name}")
        
        # 生成候选路径
        paths = generate_candidate_paths(drone_name, num_angles=15, num_velocities=5)
        all_paths[drone_name] = paths
        print(f"  生成了 {len(paths)} 条候选路径")
        
        # 生成初始列
        columns = generate_initial_columns(drone_name, paths)
        all_columns[drone_name] = columns
    
    # 保存结果
    print("\n保存结果...")
    
    # 保存为pickle格式（保留对象结构）
    with open('columns_pool.pkl', 'wb') as f:
        pickle.dump(all_columns, f)
    
    with open('paths_pool.pkl', 'wb') as f:
        pickle.dump(all_paths, f)
    
    # 同时保存为CSV格式（便于查看）
    columns_data = []
    for drone_name, columns in all_columns.items():
        for col in columns:
            columns_data.append({
                'drone': col.drone,
                'missile': col.missile,
                'velocity': col.velocity,
                'theta': col.theta,
                'tau': col.tau,
                'delta': col.delta,
                'cover_time': col.cover_time,
                'ton': col.ton,
                'toff': col.toff
            })
    
    df = pd.DataFrame(columns_data)
    df.to_csv('columns_summary.csv', index=False)
    
    print(f"总共生成了 {len(columns_data)} 个候选列")
    print("结果已保存到 columns_pool.pkl, paths_pool.pkl 和 columns_summary.csv")
    
    # 打印统计信息
    print("\n统计信息:")
    for drone_name, columns in all_columns.items():
        missile_counts = {}
        for col in columns:
            missile_counts[col.missile] = missile_counts.get(col.missile, 0) + 1
        print(f"  {drone_name}: 总计 {len(columns)} 列")
        for missile, count in missile_counts.items():
            print(f"    -> {missile}: {count} 列")

if __name__ == "__main__":
    main()