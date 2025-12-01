# AON.py
import json
import math
import heapq
from collections import defaultdict
from calculate import get_link_travel_time, get_total_travel_time

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

# 构建有向边列表（仅用于构建图）
links = []
link_key_to_index = {}

for i, pair in enumerate(network['links']['between']):
    u, v = pair[0], pair[1]
    length = euclidean_distance(u, v)
    capacity = network['links']['capacity'][i]
    speedmax = network['links']['speedmax'][i]
    t0 = length / speedmax
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

# 构建邻接表
graph = defaultdict(list)
for idx, link in enumerate(links):
    graph[link['from']].append((link['to'], idx))  # (neighbor, link_index)

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
# 3. 全有全无分配函数
# ----------------------------
def dijkstra_all_or_nothing(graph, od_demand, n_links):
    """基于自由流时间 t0 的全有全无分配"""
    flow = [0.0] * n_links
    for (orig, dest), demand_val in od_demand.items():
        if demand_val <= 0:
            continue
        # Dijkstra
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
                tt = links[link_idx]['t0']  # 使用自由流时间！
                if d + tt < dist[v]:
                    dist[v] = d + tt
                    prev_link[v] = link_idx
                    heapq.heappush(pq, (dist[v], v))
        # 回溯路径
        if dist[dest] == float('inf'):
            print(f"Warning: No path from {orig} to {dest}")
            continue
        curr = dest
        path_links = []
        while curr != orig:
            lid = prev_link[curr]
            path_links.append(lid)
            # 找前驱节点（通过 link 反推）
            u = links[lid]['from']
            curr = u
        for lid in path_links:
            flow[lid] += demand_val
    return flow

# ----------------------------
# 4. 执行 AON 分配
# ----------------------------
flow_aon = dijkstra_all_or_nothing(graph, od_demand, n_links)

# ----------------------------
# 5. 输出结果
# ----------------------------

print("\n=== All-or-Nothing Link Flows (based on free-flow time) ===")
for i, link in enumerate(links):
    if flow_aon[i] > 1e-3:
        print(f"{link['from']}->{link['to']}: flow={flow_aon[i]:.2f}, capacity={link['capacity']}, "
                f"t0={link['t0']:.2f}, t={get_link_travel_time(flow_aon, i, links):.2f}")
        
TTT_aon = get_total_travel_time(flow_aon, links)
print(f"Total Travel Time (AON-TTT): {TTT_aon:.2f}")
        

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

    # 添加边并赋值流量 Q 和行程时间 T
    for i, link in enumerate(links):
        u = link['from']
        v = link['to']
        q = flow_aon[i]
        t = get_link_travel_time(flow_aon, i, links)
        if not G.has_edge(u, v):
            G.add_edge(u, v, Q=q, T=t)
        else:
            # 理论上不会重复
            G[u][v]['Q'] += q
            G[u][v]['T'] += t

    # 调用可视化函数
    visualize_network(G, pos, TTT=TTT_aon)
except ImportError:
    print("visualize_network not available. Skipping visualization.")