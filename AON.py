# AON.py
from calculate import get_link_travel_time, get_total_travel_time
from data_load import load_network_and_demand, build_graph_and_links
from assignment_utils import all_or_nothing_assignment
# ----------------------------
# 全有全无交通分配(AON)  
# ----------------------------

def All_or_Nothing_Traffic_Assignment(links, graph, pos, node_names, od_demand):
    """
    执行基于自由流时间的全有全无交通分配
    
    Returns:
        dict: {
            'flow': flow_vector,
            'total_travel_time': TTT,
            'graph': graph,
            'links': links,
            'pos': pos,
            'node_names': node_names
        }
    """

    # 构建自由流行程时间
    free_flow_tt = [link['t0'] for link in links]
    
    # 调用通用 AON
    flow_aon = all_or_nothing_assignment(graph, links, od_demand, free_flow_tt)
    
    TTT_aon = get_total_travel_time(flow_aon, links)

    return {
        'flow': flow_aon,
        'total_travel_time': TTT_aon,
        'graph': graph,
        'links': links,
        'pos': pos,
        'node_names': node_names
    }

# ----------------------------
# 主程序入口
# ----------------------------
if __name__ == '__main__':
    network_file='data/network.json'
    demand_file='data/demand.json'
    
    # 1. 加载数据
    network, demand = load_network_and_demand(network_file, demand_file)
    
    # 2. 构建图结构
    graph, links, pos, node_names, n_links = build_graph_and_links(network)
    
    # 3. 整理 OD 需求
    od_demand = {}
    for o, d, amt in zip(demand['from'], demand['to'], demand['amount']):
        od_demand[(o, d)] = od_demand.get((o, d), 0) + amt
    
    # 4. 执行 全有全无交通分配
    AON_result = All_or_Nothing_Traffic_Assignment(links, graph, pos, node_names, od_demand)

    print("\n=== All-or-Nothing Link Flows (based on free-flow time) ===")
    for i, link in enumerate(AON_result['links']):
        flow = AON_result['flow'][i]
        t_actual = get_link_travel_time(AON_result['flow'], i, AON_result['links'])
        print(f"{link['from']}->{link['to']}: flow={flow:.2f}, "
                f"capacity={link['capacity']}, t0={link['t0']:.2f}, t={t_actual:.2f}")

    print(f"\nTotal Travel Time (AON-TTT): {AON_result['total_travel_time']:.2f}")
    
    # 可视化
    try:
        from visualize_network import visualize_network
        import networkx as nx
        G = nx.DiGraph()
        for node in AON_result['node_names']:
            G.add_node(node)
        for i, link in enumerate(AON_result['links']):
            u, v = link['from'], link['to']
            q = AON_result['flow'][i]
            t = get_link_travel_time(AON_result['flow'], i, AON_result['links'])
            G.add_edge(u, v, Q=q, T=t)
        visualize_network(G, AON_result['pos'], TTT=AON_result['total_travel_time'], 
                        title="All-or-Nothing Assignment Result")
    except ImportError:
        print("visualize_network not available. Skipping visualization.")