# FW.py
import heapq
from collections import defaultdict
from calculate import line_search_newton, get_link_travel_time, Beckmann_function, get_total_travel_time
from data_load import load_network_and_demand, build_graph_and_links
# ----------------------------

def dijkstra_shortest_path(graph, links, origin, dest, travel_times):
    """基于给定 travel_times 返回最短路径的 link 索引列表"""
    dist = defaultdict(lambda: float('inf'))
    prev_link = {}
    dist[origin] = 0
    pq = [(0, origin)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == dest:
            break
        for v, link_idx in graph[u]:
            tt = travel_times[link_idx]
            new_dist = d + tt
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev_link[v] = link_idx
                heapq.heappush(pq, (new_dist, v))
    
    if dist[dest] == float('inf'):
        return None  # 无路径
    
    # 回溯路径
    path_links = []
    curr = dest
    while curr != origin:
        link_idx = prev_link[curr]
        path_links.append(link_idx)
        curr = links[link_idx]['from']
    return path_links

def all_or_nothing_assignment(graph, links, od_demand, travel_times):
    """全有全无分配：返回流量向量 y"""
    n_links = len(links)
    y = [0.0] * n_links
    for (orig, dest), demand_val in od_demand.items():
        if demand_val <= 0:
            continue
        path = dijkstra_shortest_path(graph, links, orig, dest, travel_times)
        if path is None:
            print(f"Warning: No path from {orig} to {dest}")
            continue
        for lid in path:
            y[lid] += demand_val
    return y

def initialize_with_free_flow(graph, links, od_demand):
    """用自由流时间做 AON 初始化"""
    free_flow_tt = [link['t0'] for link in links]
    return all_or_nothing_assignment(graph, links, od_demand, free_flow_tt)

def frank_wolfe_traffic_assignment(
    network_file='data/network.json',
    demand_file='data/demand.json',
    max_iter=500,
    epsilon=1e-6,
    verbose=True
):
    """
    执行 Frank-Wolfe 用户均衡交通分配
    
    Returns:
        dict: {
            'flow': x,
            'total_travel_time': TTT,
            'Beckmann_value': Z(x),
            'iterations': iteration,
            'converged': bool,
            'graph': graph,
            'links': links,
            'pos': pos,
            'node_names': node_names
        }
    """
    # 1. 加载数据
    network, demand = load_network_and_demand(network_file, demand_file)
    
    # 2. 构建图结构
    graph, links, pos, node_names, n_links = build_graph_and_links(network)
    
    # 3. 整理 OD 需求
    od_demand = {}
    for o, d, amt in zip(demand['from'], demand['to'], demand['amount']):
        od_demand[(o, d)] = od_demand.get((o, d), 0) + amt

    # 4. 初始化
    x = initialize_with_free_flow(graph, links, od_demand)

    best_obj = float('inf')
    stagnation_count = 0
    converged = False

    # 5. 主循环
    for iteration in range(1, max_iter + 1):
        # 当前阻抗
        t_current = [get_link_travel_time(x, i, links) for i in range(n_links)]
        
        # 全有全无分配
        y = all_or_nothing_assignment(graph, links, od_demand, t_current)
        
        # 相对间隙
        numerator = sum((x[i] - y[i]) * t_current[i] for i in range(n_links))
        denominator = sum(x[i] * t_current[i] for i in range(n_links))
        relative_gap = numerator / denominator if denominator > 1e-12 else float('inf')
        
        # 收敛检查
        if relative_gap >= 0 and relative_gap < epsilon:
            converged = True
            if verbose:
                print(f"✅ Converged at iter {iteration} with relative gap = {relative_gap:.2e}")
            break
        
        # 线搜索
        if iteration == 1 and all(v == 0 for v in x):
            alpha = 1.0
        else:
            alpha = line_search_newton(x, y, links)
            alpha = max(0.0, min(1.0, alpha))  # 保护
            
            if alpha < 1e-6 and relative_gap > 1e-3:
                alpha = min(0.1, 2.0 / (iteration + 1))
        
        # 更新
        x_new = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n_links)]
        obj_val = Beckmann_function(x_new, links)
        
        # 停滞检测
        if obj_val < best_obj - 1e-8:
            best_obj = obj_val
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        if stagnation_count >= 20:
            if verbose:
                print(f"⚠️ Stagnation detected at iteration {iteration}")
            break
        
        x = x_new
        
        # 日志
        if verbose and (iteration % 10 == 0 or iteration <= 5):
            TTT_cur = get_total_travel_time(x, links)
            dir_norm = sum(abs(y[i] - x[i]) for i in range(n_links))
            print(f"Iter {iteration:3d}: Beckmann Z(x)={obj_val:.2f}, Alpha={alpha:.6f}, "
                  f"Gap={relative_gap:.2e}, DirNorm={dir_norm:.2f}, TTT={TTT_cur:.2f}")
    
    # 最终结果
    final_TTT = get_total_travel_time(x, links)
    final_obj = Beckmann_function(x, links)
    
    return {
        'flow': x,
        'total_travel_time': final_TTT,
        'Beckmann_value': final_obj,
        'iterations': iteration,
        'converged': converged,
        'graph': graph,
        'links': links,
        'pos': pos,
        'node_names': node_names
    }

# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == '__main__':
    FW_result = frank_wolfe_traffic_assignment(verbose=True)
    
    print("\n=== Frank-Wolfe Flows ===")
    for i, link in enumerate(FW_result['links']):
        flow = FW_result['flow'][i]
        if flow > 1e-3:
            t_val = get_link_travel_time(FW_result['flow'], i, FW_result['links'])
            print(f"{link['from']}->{link['to']}: flow={flow:.2f}, "
                  f"capacity={link['capacity']}, t0={link['t0']:.2f}, t={t_val:.2f}")
    
    print(f"\nTotal Travel Time (FW-TTT): {FW_result['total_travel_time']:.2f},"
          f" Beckmann_value: {FW_result['Beckmann_value']:.2f}, ")
    
    # 可视化
    try:
        from visualize_network import visualize_network
        import networkx as nx
        
        G = nx.DiGraph()
        for node in FW_result['node_names']:
            G.add_node(node)
        
        for i, link in enumerate(FW_result['links']):
            u, v = link['from'], link['to']
            q = FW_result['flow'][i]
            t = get_link_travel_time(FW_result['flow'], i, FW_result['links'])
            G.add_edge(u, v, Q=q, T=t)
        
        visualize_network(G, FW_result['pos'], TTT=FW_result['total_travel_time'])
    except ImportError:
        print("visualize_network not available. Skipping visualization.")