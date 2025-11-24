import json
import numpy as np
import networkx as nx
import time
from collections import defaultdict
from tqdm import *
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from algorithm.optimal_minimized import find_optimal_alpha
from algorithm.dijkstra_shortest import dijkstra

def build_network_from_json(network_file, demand_file):
    # 读取 network.json
    with open(network_file, 'r', encoding='utf-8') as f:
        network = json.load(f)
    
    # 读取 demand.json
    with open(demand_file, 'r', encoding='utf-8') as f:
        demand = json.load(f)

    # 构建节点坐标字典（用于后续可视化）
    node_names = network['nodes']['name']
    xs = network['nodes']['x']
    ys = network['nodes']['y']
    pos_dict = {name: (x, y) for name, x, y in zip(node_names, xs, ys)}

    # 构建有向图 G
    G = nx.DiGraph()

    # 添加路段（links）
    links = network['links']
    for i, link_str in enumerate(links['between']):
        u = link_str[0]  # 起点
        v = link_str[1]  # 终点
        capacity = links['capacity'][i]
        speedmax = links['speedmax'][i]

        # 计算欧氏距离作为路段长度
        x1, y1 = pos_dict[u]
        x2, y2 = pos_dict[v]
        length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

        # 自由流时间 t0 = length / speedmax
        t0 = length / speedmax if speedmax > 0 else float('inf')

        # 添加边到图中（路网是双向的）
        G.add_edge(u, v, t0=t0, C=capacity)
        G.add_edge(v, u, t0=t0, C=capacity)

    # 构建 OD 需求字典
    od_pairs = {}
    for o, d, q in zip(demand['from'], demand['to'], demand['amount']):
        od_pairs[(o, d)] = q

    # 初始化流量 Q
    for u, v in G.edges():
        G[u][v]['Q'] = 0.0

    return G, od_pairs, pos_dict

def all_or_nothing_assignment(G, od_pairs):
    """
    全有全无分配：根据当前图 G 的阻抗（边属性 't'），将 OD 流量加载到最短路径。
    
    参数:
        G: networkx.DiGraph，边需包含 't' 属性（旅行时间）
        od_pairs: dict，格式 {origin: [(dest, flow), ...]}
    
    返回:
        y: np.ndarray，边流量向量
    """
    # 获取边列表并建立索引映射（顺序固定）
    edges = list(G.edges())
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    
    # 初始化流量向量
    y = np.zeros(len(edges))
    
    # 对每个起点 o 计算最短路径树
    for o in od_pairs:
        try:
            paths = dijkstra(G, o, weight='t')  # 假设 dijkstra 支持 weight 参数
            for d, q in od_pairs[o]:
                if d in paths:
                    path = paths[d]
                    # 遍历路径上的每一段边
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        if (u, v) in edge_to_idx:  # 安全检查
                            y[edge_to_idx[(u, v)]] += q
        except nx.NetworkXNoPath:
            continue  # 跳过不可达的 OD 对
    
    return y


def FW_allocation(G, od_pairs, max_iter, tol):
    """
    Ford-Warshall 分配算法：根据当前图 G 的阻抗（边属性 't'），将 OD 流量加载到最短路径。
    
    参数:
        G: networkx.DiGraph，边需包含 't' 属性（旅行时间）
        od_pairs: dict，格式 {origin: [(dest, flow), ...]}
        max_iter: int，最大迭代次数
        tol: float，收敛阈值
    返回:
        G: networkx.DiGraph，边属性 'Q' 更新为分配后的流量
    """
    # 预处理边列表和属性数组
    edges = list(G.edges())
    t0_arr = np.array([G[u][v]['t0'] for u, v in edges])
    C_arr = np.array([G[u][v]['C'] for u, v in edges])
    od_groups = defaultdict(list)
    for (o, d), q in od_pairs.items():
        od_groups[o].append((d, q))

    # 计算当前阻抗
    for iter in tqdm(range(max_iter)):
        time.sleep(0.05)
        for u, v in G.edges():
            Q = G[u][v]['Q']
            C = G[u][v]['C']
            # 行程时间函数
            G[u][v]['t'] = G[u][v]['t0'] * (1 + Q / C)**2 

        y = all_or_nothing_assignment(G, od_groups)

        # 计算当前流量数组和方向向量
        Q_current = np.array([G[u][v]['Q'] for u, v in edges])
        d = y - Q_current
        
        # 定义向量化目标函数
        def obj(alpha):
            new_Q = Q_current + alpha * d
            if np.any(new_Q >= C_arr):
                return np.inf
            else:
                travel_times = t0_arr * (1 + new_Q / C_arr)**2  # 行程时间函数
                return np.sum(travel_times)
        
        # 优化步长
        alpha = find_optimal_alpha(obj, bounds=(0, 1))
        if alpha < tol:
            print(f'Converged after {iter+1} iterations.')
            break
        # 更新流量
        Q_new = Q_current + alpha * d
        for i, (u, v) in enumerate(edges):
            G[u][v]['Q'] = Q_new[i]

    return G


def visulize_network(G, pos_dict):
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

    # 使用对数缩放计算线宽（保持原逻辑）
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

    # === 颜色映射：根据流量大小使用连续 colormap（如 plasma, viridis, 或 RdBu）===
    norm = mcolors.Normalize(vmin=min_flow, vmax=max_flow)
    cmap = cm.viridis  

    edge_colors = [cmap(norm(flow)) for flow in flows]

    # === 计算标签偏移 ===
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

    # 绘制边（使用统一颜色映射）
    nx.draw_networkx_edges(
        G, pos_dict,
        edgelist=edges,
        width=widths,
        edge_color=edge_colors,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1',
        alpha=0.9
    )

    # 绘制流量标签（颜色可选白色或黑色以提高可读性）
    for i, (u, v) in enumerate(edges):
        flow = flows[i]
        offset = label_offsets[(u, v)]
        
        mid_x = (pos_dict[u][0] + pos_dict[v][0]) / 2
        mid_y = (pos_dict[u][1] + pos_dict[v][1]) / 2
        label_x = mid_x + offset[0]
        label_y = mid_y + offset[1]
        
        # 标签文字颜色：深色背景用白色，浅色用黑色（这里简单统一用白色+黑边）
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

    # 添加 colorbar
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='data/network.json', help='Path to network.json')
    parser.add_argument('--demand', type=str, default='data/demand.json', help='Path to demand.json')
    parser.add_argument('--max_iter', type=int, default=200)  # 迭代次数max_iter越大，精度越高
    parser.add_argument('--tol', type=float, default=1e-9)   # 迭代终止条件tol越小，精度越高
    args = parser.parse_args()

    # 构建网络
    G, od_pairs, pos_dict = build_network_from_json(args.network, args.demand)

    # 执行 FW 分配
    G_assigned = FW_allocation(G, od_pairs, args.max_iter, args.tol)

    # 可视化
    visulize_network(G_assigned, pos_dict)