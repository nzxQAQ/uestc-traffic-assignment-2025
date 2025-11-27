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

# 计算欧氏距离
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

    # 添加反向 v->u
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
    # === 全有全无分配 → y ===
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
        
    # === 更新流量 ===
    x = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n_links)]

    # === 调试日志 ===
    if iteration % 10 == 0 or alpha < 1e-4:
        obj_val = objective_function(x)
        dir_norm = sum(abs(y[i] - x[i]) for i in range(n_links))
        print(f"Iter {iteration}: Obj={obj_val:.2f}, Alpha={alpha:.6f}, RelGap={relative_gap:.2e}, DirNorm={dir_norm:.2f}")

# ----------------------------
# 5. 输出结果
# ----------------------------

def compute_total_travel_time(flow_vector, links):
    """
    计算所有出行者的总行程时间。
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

print("\n=== Frank-Wolfe Flows ===")
for i, link in enumerate(links):
    if x[i] > 1e-3:  # 只显示有流量的路段
        print(f"{link['from']}->{link['to']}: flow={x[i]:.2f}, capacity={link['capacity']}, "
                f"t0={link['t0']:.2f}, t={get_link_travel_time(x, i):.2f}")
        
TTT_fw = compute_total_travel_time(x, links)
print(f"Total Travel Time (FW-TTT): {TTT_fw:.2f}")
        
# ----------------------------
# 6. 可视化
# ----------------------------
try:
    from visualize_network import visualize_network
    import networkx as nx

    # === 构建 NetworkX 图 ===
    G = nx.DiGraph()

    # 添加节点（确保所有节点都在图中）
    for node in node_names:
        G.add_node(node)

    # 添加边并赋值流量 Q
    for i, link in enumerate(links):
        u = link['from']
        v = link['to']
        q = x[i]  # 最终流量
        t = get_link_travel_time(x, i)
        # 只添加有流量或原始网络中存在的边（避免重复）
        if not G.has_edge(u, v):
            G.add_edge(u, v, Q=q, T=t)
        else:
            # 理论上不会重复，因为 links 已去重
            G[u][v]['Q'] += q
            G[u][v]['T'] += t

    # 调用可视化函数
    visualize_network(G, pos, TTT=TTT_fw)
except ImportError:
    print("visualize_network not available. Skipping visualization.")