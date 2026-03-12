import numpy as np
import pandas as pd
import pickle
from pulp import *
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

from get_pos import calculate_smoke_center_pos, calculate_missile_pos
from utils import check_containment, is_point_in_sphere


# ==================== 数据类定义 ====================
@dataclass
class Column:
    drone: str;
    path_idx: int;
    missile: str;
    tau: float;
    delta: float;
    ton: float;
    toff: float;
    cover_time: float;
    velocity: float;
    theta: float;
    quality_score: float = 0.0;
    coverage_slots: List = field(default_factory=list)


@dataclass
class Path:
    drone: str;
    velocity: float;
    theta: float;
    direction_vec: np.ndarray;
    path_id: int = 0


# ==================== 常量定义 ====================
MISSILE_SPEED = 300.0
CENTER_FALSE = np.array([0.0, 0.0, 0.0])
CENTER_TRUE = np.array([0.0, 200.0, 0.0])
DT = 0.05
T_MAX = 100.0
DROP_INTERVAL = 1.0
G = 9.8
SMOKE_R = 10.0
SMOKE_T_EFF = 20.0
R_TARGET = 7.0
H_TARGET = 10.0

INIT_M = {"M1": np.array([20000, 0, 2000]), "M2": np.array([19000, 600, 2100]), "M3": np.array([18000, -600, 1900])}
INIT_D = {"FY1": np.array([17800, 0, 1800]), "FY2": np.array([12000, 1400, 1400]), "FY3": np.array([6000, -3000, 700]),
          "FY4": np.array([11000, 2000, 1800]), "FY5": np.array([13000, -2000, 1300])}


class CorrectedMasterProblem:
    def __init__(self, columns_dict: Dict, paths_dict: Dict):
        self.columns_dict = columns_dict
        self.paths_dict = paths_dict
        self.organize_columns_by_path()
        self.time_grid = np.arange(0, T_MAX, DT)
        self.time_slots = len(self.time_grid)

    def organize_columns_by_path(self):
        self.path_columns = {};
        self.path_list = []
        for drone, cols in self.columns_dict.items():
            drone_paths = {}
            for col in cols:
                path_key = (round(col.velocity, 1), round(col.theta, 4))
                if path_key not in drone_paths:
                    drone_paths[path_key] = [];
                    self.path_list.append((drone, path_key))
                drone_paths[path_key].append(col)
            self.path_columns[drone] = drone_paths
        print(f"识别出 {len(self.path_list)} 条不同路径")

    def build_coverage_matrix(self):
        self.coverage = {}
        for drone, paths in self.path_columns.items():
            for path_key, cols in paths.items():
                for col_idx, col in enumerate(cols):
                    for t_idx, t in enumerate(self.time_grid):
                        if self.verify_coverage(col, t):
                            self.coverage[(drone, path_key, col_idx, col.missile, t_idx)] = 1

    def verify_coverage(self, col, t):
        if t < col.ton or t > col.toff: return False
        drone_pos = INIT_D[col.drone];
        missile_pos_init = INIT_M[col.missile]
        velocity_vec = np.array([col.velocity * np.cos(col.theta), col.velocity * np.sin(col.theta), 0])
        missile_pos = calculate_missile_pos(missile_pos_init, CENTER_FALSE, MISSILE_SPEED, t)
        time_after_det = t - col.ton
        smoke_pos = calculate_smoke_center_pos(drone_pos, velocity_vec, col.tau, col.delta, time_after_det)
        return check_containment(missile_pos, smoke_pos, R_TARGET, H_TARGET,
                                 (CENTER_TRUE[0], CENTER_TRUE[1])) or is_point_in_sphere(missile_pos, smoke_pos, 10)

    def solve(self, max_per_drone=3):
        print("\n构建覆盖矩阵...");
        self.build_coverage_matrix()
        print("\n创建优化模型...");
        prob = LpProblem("Q5_PathConsistent", LpMaximize)

        y = {k: LpVariable(f"y_{k[0]}_{k[1]}", cat='Binary') for k in self.path_list}
        x = {(d, pk, i): LpVariable(f"x_{d}_{pk}_{i}", cat='Binary') for d, p in self.path_columns.items() for pk, c in
             p.items() for i in range(len(c))}
        Z = {(m, t): LpVariable(f"Z_{m}_{t}", cat='Binary') for m in INIT_M.keys() for t in range(self.time_slots)}

        prob += lpSum(Z.values()) * DT

        print("  添加路径唯一性约束...")
        for drone in INIT_D.keys():
            paths = [(d, p) for d, p in self.path_list if d == drone]
            if paths: prob += lpSum(y[dp] for dp in paths) == 1

        print("  添加路径-列关联约束...");
        [prob.__iadd__(var_x <= y[(d, pk)]) for (d, pk, _), var_x in x.items()]
        print("  添加投放数量约束...")
        for drone in INIT_D.keys():
            vars = [v for (d, _, _), v in x.items() if d == drone];
            if vars: prob += lpSum(vars) <= max_per_drone

        print("  添加投放间隔约束...")
        for d, p in self.path_columns.items():
            for pk, c in p.items():
                for i in range(len(c)):
                    for j in range(i + 1, len(c)):
                        if abs(c[i].tau - c[j].tau) < DROP_INTERVAL: prob += x[(d, pk, i)] + x[(d, pk, j)] <= 1

        print("  添加覆盖约束...")
        for m in INIT_M.keys():
            for t in range(self.time_slots):
                cx = [x[k[:3]] for k in self.coverage if k[3] == m and k[4] == t]
                if cx:
                    prob += Z[(m, t)] <= lpSum(cx)
                else:
                    prob += Z[(m, t)] == 0

        print("\n开始求解...");
        prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))

        if LpStatus[prob.status] in ['Optimal', 'Feasible']:
            selected_cols = [(d, self.path_columns[d][pk][ci]) for (d, pk, ci), v in x.items() if value(v) > 0.5]
            coverage = {m: sum(value(Z[(m, t)]) * DT for t in range(self.time_slots)) for m in INIT_M.keys()}
            return selected_cols, coverage, y
        else:
            print(f"求解失败: {LpStatus[prob.status]}");
            return [], {}, None

    # ==================== 函数已重构 (START) ====================
    def export_solution(self, selected_columns, coverage):
        """
        导出结果到Excel，严格按照指定的模板格式。
        首先创建包含所有无人机和3个弹药槽的完整框架，然后用解填充。
        """
        # 1. 将求解器输出的有效投放事件处理成DataFrame
        results_data = []
        if selected_columns:
            for drone, col in selected_columns:
                drone_pos = INIT_D[drone]
                velocity_vec = np.array([col.velocity * np.cos(col.theta), col.velocity * np.sin(col.theta), 0])
                release_pos = drone_pos + velocity_vec * col.tau
                detonation_pos = calculate_smoke_center_pos(drone_pos, velocity_vec, col.tau, col.delta, 0)

                results_data.append({
                    '无人机编号': drone,
                    '无人机运动速度 (m/s)': col.velocity,
                    '干扰的导弹编号': col.missile,
                    '投放时刻(s)': col.tau,  # 临时用于排序
                    '有效干扰时长 (s)': col.cover_time,
                    '烟幕干扰弹投放点的x坐标 (m)': release_pos[0],
                    '烟幕干扰弹投放点的y坐标 (m)': release_pos[1],
                    '烟幕干扰弹投放点的z坐标 (m)': release_pos[2],
                    '烟幕干扰弹起爆点的x坐标 (m)': detonation_pos[0],
                    '烟幕干扰弹起爆点的y坐标 (m)': detonation_pos[1],
                    '烟幕干扰弹起爆点的z坐标 (m)': detonation_pos[2],
                })

        if not results_data:
            df_results = pd.DataFrame(columns=results_data[0].keys() if results_data else [])
        else:
            df_results = pd.DataFrame(results_data)
            # 2. 为每个无人机的投放事件分配编号 (1, 2, 3)
            df_results = df_results.sort_values(['无人机编号', '投放时刻(s)'])
            df_results['烟幕干扰弹编号'] = df_results.groupby('无人机编号').cumcount() + 1

        # 3. 创建一个符合最终格式要求的模板DataFrame
        drones = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
        bombs = [1, 2, 3]
        template_data = [{'无人机编号': d, '烟幕干扰弹编号': b} for d in drones for b in bombs]
        df_template = pd.DataFrame(template_data)

        # 4. 将计算结果合并到模板中
        final_df = pd.merge(df_template, df_results, on=['无人机编号', '烟幕干扰弹编号'], how='left')

        # 5. 格式化最终的DataFrame
        # 对于一个被使用的无人机，其速度对它所有的弹药槽都是一样的
        final_df['无人机运动速度 (m/s)'] = final_df.groupby('无人机编号')['无人机运动速度 (m/s)'].transform('first')

        # 添加图片中要求的空列
        final_df['无人机运动方向'] = np.nan

        # 确定最终列顺序
        final_columns_order = [
            '无人机编号', '无人机运动方向', '无人机运动速度 (m/s)', '烟幕干扰弹编号',
            '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)', '烟幕干扰弹投放点的z坐标 (m)',
            '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)', '烟幕干扰弹起爆点的z坐标 (m)',
            '有效干扰时长 (s)', '干扰的导弹编号'
        ]
        final_df = final_df[final_columns_order]

        # 对浮点数进行格式化，使其在Excel中更美观
        float_cols = [
            '无人机运动速度 (m/s)', '烟幕干扰弹投放点的x坐标 (m)', '烟幕干扰弹投放点的y坐标 (m)',
            '烟幕干扰弹投放点的z坐标 (m)', '烟幕干扰弹起爆点的x坐标 (m)', '烟幕干扰弹起爆点的y坐标 (m)',
            '烟幕干扰弹起爆点的z坐标 (m)', '有效干扰时长 (s)'
        ]
        # 使用 applymap 处理 NaN 值，避免格式化错误
        for col in float_cols:
            final_df[col] = final_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else '')

        # 6. 保存到Excel
        output_filename = 'result3.xlsx'
        final_df.to_excel(output_filename, index=False)
        print(f"\n✅ 解决方案已成功导出到 {output_filename}")

        return final_df
    # ==================== 函数已重构 (END) ====================
    # <<< MODIFICATION END >>>


# <<< MODIFICATION START: 主函数已更新，增加结果清理步骤 >>>
def main():
    print("=" * 60);
    print("问题5 - 路径一致性修正版");
    print("=" * 60)

    with open('merged_columns.pkl', 'rb') as f:
        columns_dict = pickle.load(f)
    with open('merged_paths.pkl', 'rb') as f:
        paths_dict = pickle.load(f)

    solver = CorrectedMasterProblem(columns_dict, paths_dict)
    selected_columns, coverage, y_vars = solver.solve(max_per_drone=3)

    if not selected_columns:
        print("未能找到有效解。");
        return

    print("\n" + "=" * 20 + " 校正解：强制路径唯一性 " + "=" * 20)
    winning_paths = {}
    for (drone, path_key), y_var in y_vars.items():
        if value(y_var) > 0.5:
            if drone not in winning_paths or value(y_var) > winning_paths[drone]['score']:
                winning_paths[drone] = {'path': path_key, 'score': value(y_var)}

    for drone, data in winning_paths.items():
        pk = data['path'];
        print(f"  - {drone} 唯一路径: v={pk[0]}, θ={np.degrees(pk[1]):.1f}°")

    cleaned_selected_columns = []
    for drone, col in selected_columns:
        current_path_key = (round(col.velocity, 1), round(col.theta, 4))
        if drone in winning_paths and current_path_key == winning_paths[drone]['path']:
            cleaned_selected_columns.append((drone, col))

    if cleaned_selected_columns:
        df = solver.export_solution(cleaned_selected_columns, coverage)
        print("\n各导弹总遮蔽时间:")
        for m, t in coverage.items(): print(f"  - {m}: {t:.2f} 秒")
        print(f"总遮蔽时间: {sum(coverage.values()):.2f} 秒")


# <<< MODIFICATION END >>>

if __name__ == "__main__":
    main()