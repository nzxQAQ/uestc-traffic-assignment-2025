import json
import math
from collections import defaultdict
def load_network_and_demand(network_file='data/network.json', demand_file='data/demand.json'):
    with open(network_file, 'r') as f:
        network = json.load(f)
    with open(demand_file, 'r') as f:
        demand = json.load(f)
    return network, demand

def build_graph_and_links(network):
    # 节点坐标映射
    node_names = network['nodes']['name']
    x_coords = network['nodes']['x']
    y_coords = network['nodes']['y']
    pos = {name: (x, y) for name, x, y in zip(node_names, x_coords, y_coords)}

    # 欧氏距离
    def euclidean_distance(u, v):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # 构建有向边列表
    links = []
    link_key_to_index = {}
    for i, pair in enumerate(network['links']['between']):
        u, v = pair[0], pair[1]
        length = euclidean_distance(u, v)
        capacity = network['links']['capacity'][i]
        speedmax = network['links']['speedmax'][i]
        t0 = length / speedmax
        links.append({
            'from': u,
            'to': v,
            'length': length,
            'capacity': capacity,
            'speedmax': speedmax,
            't0': t0
        })
        link_key_to_index[(u, v)] = len(links) - 1
        
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
        graph[link['from']].append((link['to'], idx))

    return graph, links, pos, node_names, n_links