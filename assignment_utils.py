# assignment_utils.py
import heapq
from collections import defaultdict

def dijkstra_shortest_path(graph, links, origin, dest, link_travel_times):
    """
    基于给定的 link_travel_times 执行 Dijkstra，返回最短路径上的 link 索引列表（从 origin 到 dest）
    
    Args:
        graph: 邻接表，graph[u] = [(v, link_idx), ...]
        links: 链路列表，links[i] = {'from': u, 'to': v, ...}
        origin, dest: 起点与终点节点 ID
        link_travel_times: list[float]，长度 = len(links)，表示每条链路的当前行程时间
    
    Returns:
        list[int]: 路径上的 link 索引列表（按从 dest 回溯到 origin 的顺序，即 [last_link, ..., first_link]）
        None: 若不可达
    """
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
            tt = link_travel_times[link_idx]
            new_dist = d + tt
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev_link[v] = link_idx
                heapq.heappush(pq, (new_dist, v))

    if dist[dest] == float('inf'):
        return None

    # 回溯路径
    path_links = []
    curr = dest
    while curr != origin:
        link_idx = prev_link[curr]
        path_links.append(link_idx)
        curr = links[link_idx]['from']
    return path_links


def all_or_nothing_assignment(graph, links, od_demand, link_travel_times):
    """
    执行一次全有全无（AON）分配
    
    Args:
        graph, links: 网络结构
        od_demand: dict[(orig, dest)] = demand_value
        link_travel_times: 当前各 link 的行程时间（用于最短路计算）
    
    Returns:
        list[float]: 每条 link 上的分配流量 y[i]
    """
    n_links = len(links)
    y = [0.0] * n_links

    for (orig, dest), demand_val in od_demand.items():
        if demand_val <= 0:
            continue
        path = dijkstra_shortest_path(graph, links, orig, dest, link_travel_times)
        if path is None:
            print(f"Warning: No path from {orig} to {dest}")
            continue
        for lid in path:
            y[lid] += demand_val

    return y