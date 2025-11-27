# IA.py
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

def euclidean_distance(u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 构建有向边列表
links = []
for i, pair in enumerate(network['links']['between']):
    u, v = pair[0], pair[1]
    length = euclidean_distance(u, v)
    capacity = network['links']['capacity'][i]
    speedmax = network['links']['speedmax'][i]
    t0 = length / speedmax
    # 正向
    links.append({'from': u, 'to': v, 'length': length, 'capacity': capacity, 'speedmax': speedmax, 't0': t0})
    # 反向
    links.append({'from': v, 'to': u, 'length': length, 'capacity': capacity, 'speedmax': speedmax, 't0': t0})

n_links = len(links)

# 构建邻接表（动态权重）
graph = defaultdict(list)
for idx, link in enumerate(links):
    graph[link['from']].append((link['to'], idx))

# ----------------------------
# 2. OD 需求整理
# ----------------------------
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
    return t0 * (1 + (Q / C)) ** 2

def dijkstra_all_or_nothing_with_flow(graph, od_demand_partial, flow_current):
    """基于当前 flow 的行程时间，执行一次 AON 分配"""
    y = [0.0] * n_links
    for (orig, dest), demand_val in od_demand_partial.items():
        if demand_val <= 0:
            continue
        dist = defaultdict(lambda: float('inf'))
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
                tt = get_link_travel_time(flow_current, link_idx)
                new_dist = d + tt
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev_link[v] = link_idx
                    heapq.heappush(pq, (new_dist, v))
        if dist[dest] == float('inf'):
            print(f"Warning: No path from {orig} to {dest}")
            continue
        curr = dest
        path_links = []
        while curr != orig:
            lid = prev_link[curr]
            path_links.append(lid)
            curr = links[lid]['from']
        for lid in path_links:
            y[lid] += demand_val
    return y

def compute_total_travel_time(flow_vector):
    total = 0.0
    for i, q in enumerate(flow_vector):
        if q > 0:
            t = get_link_travel_time(flow_vector, i)
            total += q * t
    return total

# ----------------------------
# 4. 增量分配主逻辑
# ----------------------------
K = 5  # 分成 5 次分配（可调）
step_demand = {od: amt / K for od, amt in od_demand.items()}

# 初始化流量
x = [0.0] * n_links

print(f"\n=== Incremental Assignment (K={K}) ===")
for k in range(1, K + 1):
    print(f"Step {k}/{K}: assigning {100/K:.1f}% of demand...")
    # 基于当前 x 的行程时间，分配 step_demand
    y_k = dijkstra_all_or_nothing_with_flow(graph, step_demand, x)
    # 累加到总流量
    x = [x[i] + y_k[i] for i in range(n_links)]

# ----------------------------
# 5. 输出结果
# ----------------------------
print("\n=== Incremental Assignment Link Flows ===")
for i, link in enumerate(links):
    if x[i] > 1e-3:
        t_val = get_link_travel_time(x, i)
        print(f"{link['from']}->{link['to']}: flow={x[i]:.2f}, t={t_val:.2f}")

TTT_inc = compute_total_travel_time(x)
print(f"\n📊 Total Travel Time (Incremental): {TTT_inc:.2f} veh·h")

# ----------------------------
# 6. 可视化
# ----------------------------
try:
    from visualize_network import visualize_network
    import networkx as nx

    G = nx.DiGraph()
    for node in node_names:
        G.add_node(node)
    for i, link in enumerate(links):
        u, v = link['from'], link['to']
        q = x[i]
        t = get_link_travel_time(x, i)
        G.add_edge(u, v, Q=q, T=t)

    visualize_network(G, pos, TTT_inc)
except ImportError:
    print("visualize_network not available. Skipping visualization.")