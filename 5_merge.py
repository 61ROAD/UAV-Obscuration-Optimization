# merge_with_classes.py
import pickle
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Dict

# ==================== 定义数据类 ====================
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
    quality_score: float = 0.0  # 质量评分
    coverage_slots: List = field(default_factory=list)  # 覆盖的时间槽
    
@dataclass
class Path:
    """候选路径"""
    drone: str
    velocity: float
    theta: float
    direction_vec: np.ndarray
    path_id: int = 0

# ==================== 合并函数 ====================
def load_columns_with_fallback(filename):
    """尝试加载列文件，如果失败则从CSV重建"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"  成功加载 {filename}")
        return data
    except FileNotFoundError:
        print(f"  文件 {filename} 不存在")
        return {}
    except Exception as e:
        print(f"  加载 {filename} 失败: {e}")
        print(f"  尝试从CSV重建...")
        return {}

def rebuild_from_csv_files():
    """从CSV文件重建列数据"""
    all_columns = {}
    
    # 查找所有相关的CSV文件
    csv_files = [
        'columns_summary.csv',
        'fy4_columns_summary.csv', 
        'enhanced_columns_summary.csv',
        'candidate_paths.csv'
    ]
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n从 {csv_file} 读取数据...")
            df = pd.read_csv(csv_file)
            
            # 检查是否是列数据（有特定字段）
            if 'drone' in df.columns and 'missile' in df.columns:
                for _, row in df.iterrows():
                    drone = row['drone']
                    if drone not in all_columns:
                        all_columns[drone] = []
                    
                    # 创建Column对象
                    col = Column(
                        drone=drone,
                        path_idx=int(row.get('path_idx', 0)),
                        missile=row.get('missile', 'M1'),
                        tau=float(row.get('tau', 0)),
                        delta=float(row.get('delta', 0)),
                        ton=float(row.get('ton', 0)),
                        toff=float(row.get('toff', 0)),
                        cover_time=float(row.get('cover_time', 0)),
                        velocity=float(row.get('velocity', 100)),
                        theta=np.radians(float(row.get('theta_deg', 0))) if 'theta_deg' in row else float(row.get('theta', 0)),
                        quality_score=float(row.get('quality_score', 0)) if 'quality_score' in row else 0.0,
                        coverage_slots=[]
                    )
                    all_columns[drone].append(col)
                
                print(f"  从 {csv_file} 加载了 {len(df)} 条记录")
    
    return all_columns

def rebuild_paths_from_csv():
    """从CSV文件重建路径数据"""
    all_paths = {}
    
    if os.path.exists('candidate_paths.csv'):
        print("\n从 candidate_paths.csv 重建路径...")
        df = pd.read_csv('candidate_paths.csv')
        
        for _, row in df.iterrows():
            drone = row['drone']
            if drone not in all_paths:
                all_paths[drone] = []
            
            path = Path(
                drone=drone,
                velocity=float(row.get('speed', row.get('velocity', 100))),
                theta=np.radians(float(row.get('angle_deg', 0))) if 'angle_deg' in row else float(row.get('angle_rad', row.get('theta', 0))),
                direction_vec=np.array([float(row.get('vx', 0)), float(row.get('vy', 0)), float(row.get('vz', 0))]),
                path_id=int(row.get('path_id', 0))
            )
            all_paths[drone].append(path)
        
        print(f"  重建了 {len(df)} 条路径")
    
    return all_paths

def merge_all_data():
    """主合并函数"""
    print("="*60)
    print("开始合并数据（带类定义）")
    print("="*60)
    
    # 尝试直接加载pkl文件
    all_columns = {}
    all_paths = {}
    
    # 尝试加载主文件
    for filename in ['columns_pool.pkl', 'enhanced_columns_pool.pkl']:
        if os.path.exists(filename):
            print(f"\n尝试加载 {filename}...")
            data = load_columns_with_fallback(filename)
            if data and isinstance(data, dict):
                all_columns.update(data)
    
    for filename in ['candidate_paths.pkl', 'enhanced_paths_pool.pkl', 'paths_pool.pkl']:
        if os.path.exists(filename):
            print(f"\n尝试加载 {filename}...")
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                if data and isinstance(data, dict):
                    all_paths.update(data)
                    print(f"  成功加载路径数据")
            except Exception as e:
                print(f"  加载失败: {e}")
    
    # 尝试加载FY4文件
    for filename in ['fy4_columns_pool.pkl', 'FY4_columns_pool.pkl']:
        if os.path.exists(filename):
            print(f"\n尝试加载 {filename}...")
            data = load_columns_with_fallback(filename)
            if data and isinstance(data, dict):
                if 'FY4' in data:
                    all_columns['FY4'] = data['FY4']
                else:
                    all_columns.update(data)
    
    for filename in ['fy4_paths_pool.pkl', 'FY4_paths_pool.pkl']:
        if os.path.exists(filename):
            print(f"\n尝试加载 {filename}...")
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                if data and isinstance(data, dict):
                    if 'FY4' in data:
                        all_paths['FY4'] = data['FY4']
                    else:
                        all_paths.update(data)
            except Exception as e:
                print(f"  加载失败: {e}")
    
    # 如果列数据为空，从CSV重建
    if not any(all_columns.values()):
        print("\n列数据为空，从CSV文件重建...")
        all_columns = rebuild_from_csv_files()
    
    # 如果路径数据为空，从CSV重建
    if not any(all_paths.values()):
        print("\n路径数据为空，从CSV文件重建...")
        all_paths = rebuild_paths_from_csv()
    
    # 确保所有5架无人机都有条目
    for drone in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        if drone not in all_columns:
            all_columns[drone] = []
        if drone not in all_paths:
            all_paths[drone] = []
    
    # 保存合并结果
    save_results(all_columns, all_paths)
    
    return all_columns, all_paths

def save_results(all_columns, all_paths):
    """保存合并结果"""
    
    # 保存pkl文件
    with open('merged_columns.pkl', 'wb') as f:
        pickle.dump(all_columns, f)
    print("\n保存: merged_columns.pkl")
    
    with open('merged_paths.pkl', 'wb') as f:
        pickle.dump(all_paths, f)
    print("保存: merged_paths.pkl")
    
    # 生成统计CSV
    summary_data = []
    for drone, columns in all_columns.items():
        for col in columns:
            summary_data.append({
                'drone': col.drone,
                'missile': col.missile,
                'velocity': col.velocity,
                'theta_deg': np.degrees(col.theta),
                'tau': col.tau,
                'delta': col.delta,
                'cover_time': col.cover_time,
                'quality_score': getattr(col, 'quality_score', 0),
                'ton': col.ton,
                'toff': col.toff
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv('merged_summary.csv', index=False)
        print(f"保存: merged_summary.csv ({len(df)} 行)")
    
    # 打印统计
    print("\n" + "="*60)
    print("合并统计:")
    print("="*60)
    
    total_columns = 0
    total_paths = 0
    
    for drone in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        col_count = len(all_columns.get(drone, []))
        path_count = len(all_paths.get(drone, []))
        total_columns += col_count
        total_paths += path_count
        
        status = "✓" if col_count > 0 else "✗"
        print(f"{drone}: 列[{status}] {col_count:4d} | 路径 {path_count:4d}")
        
        # 显示导弹分布
        if col_count > 0:
            missile_dist = {}
            for col in all_columns[drone]:
                missile = col.missile
                missile_dist[missile] = missile_dist.get(missile, 0) + 1
            print(f"      导弹分布: {missile_dist}")
    
    print(f"\n总计: {total_columns} 个列, {total_paths} 条路径")
    
    if total_columns == 0:
        print("\n⚠ 警告: 没有找到任何列数据！")
        print("请确保以下文件之一存在:")
        print("  - columns_summary.csv")
        print("  - fy4_columns_summary.csv")
        print("  - enhanced_columns_summary.csv")

if __name__ == "__main__":
    # 显示当前目录的文件
    import glob
    print("当前目录文件:")
    for pattern in ['*.pkl', '*.csv']:
        files = glob.glob(pattern)
        if files:
            print(f"\n{pattern}:")
            for f in sorted(files):
                size = os.path.getsize(f) / 1024
                print(f"  - {f} ({size:.1f} KB)")
    
    print("\n" + "="*60)
    
    # 执行合并
    all_columns, all_paths = merge_all_data()
    
    # 验证
    if any(len(cols) > 0 for cols in all_columns.values()):
        print("\n✓ 合并成功！")
        print("下一步: 运行主问题求解器")
        print("  注意: 主问题求解器需要使用 'merged_columns.pkl' 和 'merged_paths.pkl'")
    else:
        print("\n✗ 合并失败 - 没有找到列数据")
        print("请提供以下文件之一:")
        print("  1. 原始的pkl文件（需要包含Column类定义）")
        print("  2. CSV文件（columns_summary.csv 或 fy4_columns_summary.csv）")