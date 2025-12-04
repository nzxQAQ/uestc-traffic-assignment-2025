# IA.py
from calculate import get_link_travel_time, get_total_travel_time
from data_load import load_network_and_demand, build_graph_and_links
from assignment_utils import all_or_nothing_assignment
from visualize_network import visualize_network, build_network
# ----------------------------
# 增量交通分配（Incremental Assignment）
# ----------------------------
def Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand,
    K=1000
):
    """
    执行增量交通分配（Incremental Assignment）
    
    Args:
        K: 将总需求分成 K 份逐步分配
    
    Returns:
        dict: {
            'flow': x,
            'total_travel_time': TTT,
            'K': K,
            'graph': graph,
            'links': links,
            'pos': pos,
            'node_names': node_names
        }
    """
    
    print(f"\n=== Incremental Assignment (K={K}) ===")

    step_demand = {od: amt / K for od, amt in od_demand.items()}
    x = [0.0] * n_links

    for k in range(1, K + 1):
        # 获取当前行程时间列表
        t_current = [get_link_travel_time(x, i, links) for i in range(len(links))]   
        # 执行 AON 分配当前 step_demand 
        y_k = all_or_nothing_assignment(graph, links, step_demand, t_current)
        x = [x[i] + y_k[i] for i in range(n_links)]

    TTT_inc = get_total_travel_time(x, links)

    return {
        'flow': x,
        'total_travel_time': TTT_inc,
        'K': K,
        'graph': graph,
        'links': links,
        'pos': pos,
        'node_names': node_names
    }

# ----------------------------
# 测试入口
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
    
    # 4. 执行 增量分配(IA)
    K = 1000
    IA_title=f"Incremental Assignment Result(K = {K})"
    
    IA_result = Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand, K=K)
    
    # 5. 打印结果
    print("\n=== Incremental Assignment Link Flows ===")
    for i, link in enumerate(IA_result['links']):
        flow = IA_result['flow'][i]
        t_val = get_link_travel_time(IA_result['flow'], i, IA_result['links'])
        print(f"{link['from']}->{link['to']}: flow={flow:.2f}, t={t_val:.2f}")

    print(f"\n📊 Total Travel Time (Incremental): {IA_result['total_travel_time']:.2f} veh·h")

    # 6. 可视化
    try:
        G = build_network(IA_result)
        visualize_network(G, IA_result['pos'], TTT=IA_result['total_travel_time'], title=IA_title)
    except ImportError:
        print("visualize_network not available. Skipping visualization.")