# FW.py
import json
import math
import heapq
from collections import defaultdict
from calculate import  get_link_travel_time, get_total_travel_time, line_search_newton, Beckmann_function

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
# 3. 全有全无分配函数
# ----------------------------

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
                tt = get_link_travel_time(flow, link_idx, links)
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

# ----------------------------
# 4. Frank-Wolfe 主循环
# ----------------------------

# 初始化流量
x = [0.0] * n_links
max_iter = 500
epsilon = 1e-6

for iteration in range(1, max_iter + 1):
    # === 全有全无分配 → y ===
    y = dijkstra_all_or_nothing(graph, od_demand, x)
    t_current = [get_link_travel_time(x, i, links) for i in range(n_links)]

    # === 计算相对间隙 gap,用于评价解的精度 ===
    numerator = sum((x[i] - y[i]) * t_current[i] for i in range(n_links))
    denominator = sum(x[i] * t_current[i] for i in range(n_links))
    
    if denominator < 1e-12:
        relative_gap = float('inf')
    else:
        relative_gap = numerator / denominator

    # === 如果相对间隙足够小，立即收敛 ===
    if relative_gap < epsilon:  
        print(f"✅ Converged at iter {iteration} with relative gap = {relative_gap:.2e}")
        break

    # === 否则，继续迭代（即使 alpha 很小）===
    alpha = line_search_newton(x, y, links)

    # 【可选】防卡死：如果 alpha 极小但 gap 仍大，强制推进步长
    # if alpha <= 0.001 and relative_gap > 0.001:
    #     alpha = 0.01
        
    # === 更新流量 ===
    x = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n_links)]

    # === 调试日志 ===
    # if iteration % 10 == 0 or alpha < 1e-4:
    obj_val = Beckmann_function(x, links)
    TTT_cur = get_total_travel_time(x, links)
    dir_norm = sum(abs(y[i] - x[i]) for i in range(n_links))
    print(f"Iter {iteration}: Obj={obj_val:.6f}, Alpha={alpha:.6f}, "
        f"RelGap={relative_gap:.2e}, DirNorm={dir_norm:.2f}, TotalTime={TTT_cur:.2f}")

# ----------------------------
# 5. 输出结果
# ----------------------------

print("\n=== Frank-Wolfe Flows ===")
for i, link in enumerate(links):
    print(f"{link['from']}->{link['to']}: flow={x[i]:.2f}, capacity={link['capacity']}, "
            f"t0={link['t0']:.2f}, t={get_link_travel_time(x, i, links):.2f}")
        
TTT_fw = get_total_travel_time(x, links)
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
        t = get_link_travel_time(x, i, links)  # 最终行程时间
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