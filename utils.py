import numpy as np

def cone_from_sphere(missile_pos, smoke_center, smoke_radius):
    """
    根据导弹位置和烟幕球得到圆锥的表达式
    - 圆锥顶点: missile_pos
    - 锥轴: missile_pos -> smoke_center
    - 半顶角: arcsin(R / |MS|)
    返回 (apex, axis_unit, half_angle)
    """
    M = np.array(missile_pos, dtype=float)
    C = np.array(smoke_center, dtype=float)
    v = C - M
    d = np.linalg.norm(v)
    axis = v / d
    half_angle = np.arcsin(smoke_radius / d)
    return M, axis,half_angle


def point_in_cone(x, y, z, M, axis, cos2, eps=1e-12):
    r = np.array([x - M[0], y - M[1], z - M[2]])
    S = axis.dot(r)
    # 条件：在锥的前方 且 (a·r)^2 >= cos2 * ||r||^2
    return (S >= -eps) and (S*S >= cos2 * (r.dot(r)) - eps)

def check_containment(missile_pos,smoke_center,
                            cyl_radius=7.0, cyl_height=10.0,
                            cyl_center=(0.0,200.0), dtheta=0.1, eps=1e-9):
    apex, axis, half_angle, = cone_from_sphere(missile_pos,smoke_center,10)
    cos2 = np.cos(half_angle)**2
    M = np.array(apex, dtype=float)
    cx, cy = cyl_center

    thetas = np.arange(0, 2*np.pi, dtheta)
    for theta in thetas:
        x = cx + cyl_radius * np.cos(theta)
        y = cy + cyl_radius * np.sin(theta)

        dx = x - M[0]
        dy = y - M[1]
        dz0 = -M[2]

        c0 = axis[0]*dx + axis[1]*dy + axis[2]*dz0
        A = axis[2]**2 - cos2
        B = 2*(axis[2]*c0 - cos2*dz0)        # <-- 修正：用 c0*axis[2]
        C = c0*c0 - cos2*(dx*dx + dy*dy + dz0*dz0)

        # 退化 / 判别式不可用时，用端点判定（鲁棒且简单）
        if abs(A) < eps:
            # 线性或常数，退回到端点判断
            if not (point_in_cone(x,y,0.0,M,axis,cos2) and point_in_cone(x,y,cyl_height,M,axis,cos2)):
                return False
            else:
                continue

        disc = B*B - 4*A*C
        if disc < 0:
            # 没有实交点：可能整段都在内也可能都在外，退回到端点判断
            if not (point_in_cone(x,y,0.0,M,axis,cos2) and point_in_cone(x,y,cyl_height,M,axis,cos2)):
                return False
            else:
                continue

        z1 = (-B - np.sqrt(disc)) / (2*A)
        z2 = (-B + np.sqrt(disc)) / (2*A)
        zmin, zmax = min(z1, z2), max(z1, z2)

        # 要求区间 [0, h] 被包含在满足 f(z) >= 0 且 a·r >= 0 的区域。
        # 简单、安全的做法：检查端点是否都在圆锥内（若整段都在内，端点必然在内）。
        if not (point_in_cone(x,y,0.0,M,axis,cos2) and point_in_cone(x,y,cyl_height,M,axis,cos2)):
            return False

    return True

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


