def get_link_travel_time(flow_vector, link_idx, links):
    """路阻函数/行程时间函数：t = t0 * (1 + (Q/C))^2"""
    t0 = links[link_idx]['t0']
    C = links[link_idx]['capacity']
    Q = flow_vector[link_idx]
    return t0 * (1 + (Q / C) ) ** 2

def Beckmann_function(flow_vector, links):
    """ Beckmann势能函数Z(x)，是对路阻函数的积分"""
    total = 0.0
    for i, q in enumerate(flow_vector):
        C = links[i]['capacity']
        t0 = links[i]['t0']
        # 积分 ∫0^q t0*(1 + (x/C))^2 dx = t0*(q + q^2/C + q^3/(3*C^2))
        total += t0 * (q + (q ** 2) / C + (q ** 3) / (3 * C ** 2))
        
    return total

def get_total_travel_time(flow_vector, links):
    """
    计算所有出行者的总行程时间TTT。
    参数:
        flow_vector: list，每条 link 的流量 [q0, q1, ..., qn]
        links: list，每条 link 的信息（含 capacity, t0）
    返回:
        total_tt: float，总行程时间
    """
    total_travel_time = 0.0
    for i in range(len(links)):
        q = flow_vector[i]
        if q <= 0:
            continue
        # 使用 BPR 函数计算当前流量下的行程时间
        C = links[i]['capacity']
        t0 = links[i]['t0']
        t = t0 * (1 + (q / C)) ** 2   # BPR: β=2
        total_travel_time += q * t
    return total_travel_time

def line_search_newton(x, y, links, max_iter=20, tol=1e-8):
    """
    使用 Newton-Raphson 方法精确求解最优 alpha ∈ [0, 1]
    利用 phi'(alpha) = sum( (y_i - x_i) * t(q_i(alpha)) )
    """
    n = len(x)
    d = [y[i] - x[i] for i in range(n)]  # direction
    
    # 特殊情况：初始零解
    if all(v == 0 for v in x):
        return 1.0

    # 初始猜测：取 0.5 或上次经验（这里用 0.5）
    alpha = 0.5

    for _ in range(max_iter):
        # 限制 alpha 在 [0, 1] 内（防止发散）
        alpha = max(0.0, min(1.0, alpha))
        
        # 计算 q(alpha) = x + alpha * d
        q = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n)]
        
        # 计算 phi'(alpha) = sum( d_i * t(q_i) )
        phi_prime = 0.0
        for i in range(n):
            if abs(d[i]) < 1e-12:
                continue
            t_val = get_link_travel_time(q, i, links)  
            phi_prime += d[i] * t_val

        # 若导数接近零，已找到极小值
        if abs(phi_prime) < tol:
            break

        # 计算 phi''(alpha) = sum( d_i^2 * dt/dq )
        # dt/dq = t0 * 2 * (1 + q/C) * (1/C) = 2 * t0 / C * (1 + q/C)
        phi_double_prime = 0.0
        for i in range(n):
            if abs(d[i]) < 1e-12:
                continue
            C = links[i]['capacity']
            t0 = links[i]['t0']
            q_i = q[i]
            if C <= 0:
                continue
            dt_dq = 2 * t0 / C * (1 + q_i / C)
            phi_double_prime += d[i] * d[i] * dt_dq

        # 防止除零
        if phi_double_prime <= 0:
            phi_double_prime = 1e-12

        # Newton 更新
        alpha_new = alpha - phi_prime / phi_double_prime

        # 如果更新后变化很小，退出
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break

        alpha = alpha_new

    # 最终确保在 [0,1]
    alpha = max(0.0, min(1.0, alpha))
    return alpha