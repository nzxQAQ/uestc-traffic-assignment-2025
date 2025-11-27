import json
import math
import heapq
from collections import defaultdict

# ----------------------------
# 1. 加载数据
# ----------------------------

with open('data/network.json', 'r') as f:
    network = json.load(f)

with open('data/demand.json', 'r') as f:
    demand = json.load(f)

# 节点坐标映射
node_names = network['nodes']['name']
x_coords = network['nodes']['x']
y_coords = network['nodes']['y']
pos = {name: (x, y) for name, x, y in zip(node_names, x_coords, y_coords)}

# 计算欧氏距离（假设单位一致）
def euclidean_distance(u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 构建有向边列表
links = []
link_key_to_index = {}  # (u, v) -> index in links

for i, pair in enumerate(network['links']['between']):
    u, v = pair[0], pair[1]
    length = euclidean_distance(u, v)
    capacity = network['links']['capacity'][i]
    speedmax = network['links']['speedmax'][i]
    t0 = length / speedmax  # 自由流行程时间

    # 添加正向 u->v
    links.append({
        'from': u,
        'to': v,
        'length': length,
        'capacity': capacity,
        'speedmax': speedmax,
        't0': t0
    })
    link_key_to_index[(u, v)] = len(links) - 1

    # 添加反向 v->u（假设双向）
    links.append({
        'from': v,
        'to': u,
        'length': length,
        'capacity': capacity,
        'speedmax': speedmax,
        't0': t0
    })
    link_key_to_index[(v, u)] = len(links) - 1

n_links = len(links)

# 构建邻接表（用于 Dijkstra）
graph = defaultdict(list)
for idx, link in enumerate(links):
    graph[link['from']].append((link['to'], idx))  # (neighbor, link_index)

# ----------------------------
# 2. OD 需求整理
# ----------------------------

od_pairs = list(zip(demand['from'], demand['to']))
od_demand = {}
total_demand = 0
for o, d, amt in zip(demand['from'], demand['to'], demand['amount']):
    od_demand[(o, d)] = od_demand.get((o, d), 0) + amt
    total_demand += amt

print(f"Total OD demand: {total_demand}")

# ----------------------------
# 3. 辅助函数
# ----------------------------

def get_link_travel_time(flow, link_idx):
    """BPR 函数：t = t0 * (1 + (Q/C))^2"""
    C = links[link_idx]['capacity']
    t0 = links[link_idx]['t0']
    Q = flow[link_idx]
    return t0 * (1 + (Q / C) ) ** 2

def dijkstra_all_or_nothing(graph, od_demand, flow):
    """全有全无分配：返回新的流量向量 y"""
    y = [0.0] * n_links
    for (orig, dest), demand_val in od_demand.items():
        if demand_val <= 0:
            continue
        # Dijkstra 最短路径（基于当前 flow 的时间）
        dist = defaultdict(lambda: float('inf'))
        prev = {}
        prev_link = {}
        dist[orig] = 0
        pq = [(0, orig)]

        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            if u == dest:
                break
            for v, link_idx in graph[u]:
                tt = get_link_travel_time(flow, link_idx)
                new_dist = d + tt
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    prev_link[v] = link_idx
                    heapq.heappush(pq, (new_dist, v))

        # 回溯路径并分配流量
        if dist[dest] == float('inf'):
            print(f"Warning: No path from {orig} to {dest}")
            continue

        path_links = []
        curr = dest
        while curr != orig:
            link_idx = prev_link[curr]
            path_links.append(link_idx)
            curr = prev[curr]

        for lid in path_links:
            y[lid] += demand_val

    return y

def objective_function(flow):
    """计算总系统旅行时间（目标函数）"""
    total = 0.0
    for i, q in enumerate(flow):
        C = links[i]['capacity']
        t0 = links[i]['t0']
        # 积分 ∫0^q t0*(1 + (x/C))^2 dx = t0*(q + q^2/C + q^3/(3*C^2))
        total += t0 * (q + (q ** 2) / C + (q ** 3) / (3 * C ** 2))
        
    return total

# def line_search(x, y, max_iter=50):
#     """在方向 d = y - x 上寻找最优步长 alpha ∈ [0,1]"""
#     def phi(alpha):
#         flow = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n_links)]
#         return objective_function(flow)

#     # 使用黄金分割或简单网格搜索（这里用二分+导数近似）
#     a, b = 0.0, 1.0
#     phi_a = phi(a)
#     phi_b = phi(b)
#     for _ in range(max_iter):
#         if b - a < 1e-6:
#             break
#         m1 = a + (b - a) / 3
#         m2 = b - (b - a) / 3
#         phi_m1 = phi(m1)
#         phi_m2 = phi(m2)
#         if phi_m1 < phi_m2:
#             b = m2
#         else:
#             a = m1
#     return (a + b) / 2

def line_search(x, y):
    """
    在方向 d = y - x 上寻找最优步长 alpha ∈ [0, 1]
    使用高密度采样，尤其在 [0, 0.1] 区间，避免因采样稀疏错过最小值。
    """
    def phi(alpha):
        total = 0.0
        for i in range(len(links)):
            q = (1 - alpha) * x[i] + alpha * y[i]
            t0 = links[i]['t0']
            C = links[i]['capacity']
            # 正确的 BPR 积分：t(Q) = t0 * (1 + Q/C)^2
            total += t0 * (q + (q ** 2) / C + (q ** 3) / (3 * C ** 2))
        return total

    # 特殊处理初始零解：直接全赋值
    if all(v == 0 for v in x):
        return 1.0

    # 构建非均匀采样点：在 [0, 0.1] 高密度，其余均匀
    candidates = set()
    candidates.add(0.0)
    candidates.add(1.0)
    
    # [0, 0.1] 以 0.001 步长采样（100 个点）
    for i in range(1, 101):
        candidates.add(i * 0.001)  # 0.001 到 0.1
    
    # [0.1, 1.0] 以 0.01 步长采样（90 个点）
    for i in range(11, 101):
        candidates.add(i * 0.01)   # 0.11 到 1.0

    # 转为排序列表（虽非必需，但便于调试）
    candidates = sorted(candidates)

    # 寻找使 phi(alpha) 最小的 alpha
    best_alpha = 0.0
    best_val = phi(0.0)

    for a in candidates:
        val = phi(a)
        if val < best_val:
            best_val = val
            best_alpha = a

    # 安全机制：如果最优 alpha 过小但存在明显下降，可考虑保留；
    # 但此处信任采样结果。
    return best_alpha

# ----------------------------
# 4. Frank-Wolfe 主循环
# ----------------------------

# 初始化流量
x = [0.0] * n_links
max_iter = 100
epsilon = 1e-4

# for iteration in range(1, max_iter + 1):
#     # Step 1: 全有全无分配 → y
#     y = dijkstra_all_or_nothing(graph, od_demand, x)

#     # Step 2: 计算方向 d = y - x
#     d_norm = sum(abs(y[i] - x[i]) for i in range(n_links))
#     if d_norm < epsilon * total_demand:
#         print(f"Converged at iteration {iteration}")
#         break

#     # Step 3: 线搜索求最优 alpha
#     alpha = line_search(x, y)
#     # alpha = 0.5  # 简化处理，固定步长

#     # Step 4: 更新流量
#     x = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n_links)]

#     # 可选：打印进度
#     if iteration % 10 == 0 or iteration == 1:
#         obj_val = objective_function(x)
#         print(f"Iter {iteration}: Obj={obj_val:.2f}, Alpha={alpha:.4f}, DirNorm={d_norm:.2f}")

for iteration in range(1, max_iter + 1):
    # Step 1: 全有全无分配 → y
    y = dijkstra_all_or_nothing(graph, od_demand, x)

    # === 计算相对间隙 ===
    total_time_x = sum(get_link_travel_time(x, i) * x[i] for i in range(n_links))
    total_time_y = sum(get_link_travel_time(x, i) * y[i] for i in range(n_links))
    relative_gap = (total_time_x - total_time_y) / total_time_x if total_time_x > 1e-6 else float('inf')

    # === 如果相对间隙足够小，立即收敛 ===
    if relative_gap < 1e-4:  # 0.01% gap，标准阈值
        print(f"✅ Converged at iter {iteration} with relative gap = {relative_gap:.2e}")
        break

    # === 否则，继续迭代（即使 alpha 很小）===
    alpha = line_search(x, y)

    # 【可选】防卡死：如果 alpha 极小但 gap 仍大，强制小步长
    if alpha < 1e-5 and relative_gap > 1e-3:
        alpha = 0.01

    x = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n_links)]

    # === 日志 ===
    if iteration % 10 == 0 or alpha < 1e-4:
        obj_val = objective_function(x)
        dir_norm = sum(abs(y[i] - x[i]) for i in range(n_links))
        print(f"Iter {iteration}: Obj={obj_val:.2f}, Alpha={alpha:.6f}, RelGap={relative_gap:.2e}, DirNorm={dir_norm:.2f}")

# ----------------------------
# 5. 输出结果
# ----------------------------

print("\n=== Final Link Flows ===")
for i, link in enumerate(links):
    if x[i] > 1e-3:  # 只显示有流量的路段
        print(f"{link['from']}->{link['to']}: flow={x[i]:.2f}, capacity={link['capacity']}, "
              f"t0={link['t0']:.2f}, t={get_link_travel_time(x, i):.2f}")
        
# ----------------------------
# 6. 可视化网络（新增部分）
# ----------------------------

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def visualize_network(G, pos_dict):
    """
    可视化网络：绘制网络图，并显示每条边的流量。
    参数：
        G: networkx.DiGraph，边需包含 'Q' 属性（流量）
        pos_dict: dict，节点坐标字典
    """
    plt.figure(figsize=(10, 8))
    
    edges = list(G.edges())
    flows = np.array([G[u][v]['Q'] for u, v in edges])
    
    if len(flows) == 0:
        max_flow = 1
        min_flow = 0
    else:
        max_flow = flows.max()
        min_flow = flows.min()

    # 使用对数缩放计算线宽
    min_width = 0.5
    max_width = 5.0
    widths = []
    for flow in flows:
        if flow <= 0:
            width = min_width
        else:
            log_flow = np.log1p(flow)
            log_max = np.log1p(max_flow)
            width = min_width + (max_width - min_width) * (log_flow / log_max if log_max > 0 else 0)
        widths.append(width)

    # 颜色映射
    norm = mcolors.Normalize(vmin=min_flow, vmax=max_flow)
    cmap = cm.viridis  
    edge_colors = [cmap(norm(flow)) for flow in flows]

    # 计算标签偏移（避免双向边标签重叠）
    label_offsets = {}
    for u, v in G.edges():
        if G.has_edge(v, u):
            x1, y1 = pos_dict[u]
            x2, y2 = pos_dict[v]
            dx = x2 - x1
            dy = y2 - y1
            length = (dx*dx + dy*dy) ** 0.5
            if length > 0:
                d = 1.5
                perp_x = dy / length
                perp_y = -dx / length
                label_offsets[(u, v)] = (-d * perp_x, -d * perp_y)
                label_offsets[(v, u)] = ( d * perp_x,  d * perp_y)
            else:
                label_offsets[(u, v)] = (-0.3, 0)
                label_offsets[(v, u)] = (0.3, 0)
        else:
            label_offsets[(u, v)] = (0, 0)

    ax = plt.gca()

    # 绘制边
    nx.draw_networkx_edges(
        G, pos_dict,
        edgelist=edges,
        width=widths,
        edge_color=edge_colors,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1',
        alpha=0.9
    )

    # 绘制流量标签
    for i, (u, v) in enumerate(edges):
        flow = flows[i]
        offset = label_offsets[(u, v)]
        
        mid_x = (pos_dict[u][0] + pos_dict[v][0]) / 2
        mid_y = (pos_dict[u][1] + pos_dict[v][1]) / 2
        label_x = mid_x + offset[0]
        label_y = mid_y + offset[1]
        
        ax.annotate(
            f'{flow:.1f}',
            xy=(label_x, label_y),
            fontsize=9,
            color='white',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            zorder=5
        )

    # 节点
    nx.draw_networkx_nodes(G, pos_dict, node_size=800, node_color='lightgray', edgecolors='black')
    nx.draw_networkx_labels(G, pos_dict, font_size=12, font_weight='bold')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Traffic Flow (Q)', fontsize=12)

    plt.gca().invert_yaxis()
    plt.title('Traffic Flow Allocation\n(Line width: log-scaled flow; Color: flow magnitude)', fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# === 构建 NetworkX 图 ===
G = nx.DiGraph()

# 添加节点（确保所有节点都在图中）
for node in node_names:
    G.add_node(node)

# 添加边并赋值流量 Q
for i, link in enumerate(links):
    u = link['from']
    v = link['to']
    flow_val = x[i]  # 最终流量
    # 只添加有流量或原始网络中存在的边（避免重复）
    if not G.has_edge(u, v):
        G.add_edge(u, v, Q=flow_val)
    else:
        # 理论上不会重复，因为 links 已去重
        G[u][v]['Q'] += flow_val

# 调用可视化函数
visualize_network(G, pos)