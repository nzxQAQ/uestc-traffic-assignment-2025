# main.py
from data_load import load_network_and_demand, build_graph_and_links
from assignment_utils import dijkstra_shortest_path
from calculate import get_link_travel_time
from AON import All_or_Nothing_Traffic_Assignment
from IA import Incremental_Traffic_Assignment
from FW import Frank_Wolfe_Traffic_Assignment
from visualize_network import visualize_network, build_network, print_path

def main():
    print("=== 电子科技大学《交通规划原理》课程报告 - 软件测试 ===\n")
    
    network_file='data/network.json'
    demand_file='data/demand.json'
    
    # 加载数据
    network, demand = load_network_and_demand(network_file, demand_file)
    
    # 构建图结构
    graph, links, pos, node_names, n_links = build_graph_and_links(network)
    
    # 整理 OD 需求
    od_demand = {}
    for o, d, amt in zip(demand['from'], demand['to'], demand['amount']):
        od_demand[(o, d)] = od_demand.get((o, d), 0) + amt

    # 构建节点集合
    all_nodes = set(node_names)

    # ----------------------------
    # 问题1：不考虑拥堵时，任意两点间的最快路径是什么？
    # ----------------------------
    print("问题1：不考虑拥堵时，任意两点间的最快路径是什么？")
    print("注：如果有多条路径时间相等，仅打印其中一条。")
    free_flow_tt = [link['t0'] for link in links]
    for o in sorted(all_nodes):
        for d in sorted(all_nodes):
            if o == d:
                continue
            path = dijkstra_shortest_path(graph, links, o, d, free_flow_tt)
            path_str = print_path(links, path) if path else "不可达"
            print(f"  {o} to {d}: {path_str}")
    print()

    zero_od = {}
    aon_zero_res = All_or_Nothing_Traffic_Assignment(links, graph, pos, node_names, zero_od)
    # 可视化
    try:
        G = build_network(aon_zero_res)
        visualize_network(G, aon_zero_res['pos'], None, 
                        title="不考虑拥堵时，每个路段上的行程时间恒为自由流时间 t0")
    except ImportError:
        print("可视化不可用。跳过该步骤。")

    print("我们注意到有意思的细节：明明BE之间有直接连接的道路，但是BE之间的最快路径却是B→C→E，\n这是因为BC与CE道路上的限速为60，而BE上的限速为30.\n")

    # ----------------------------
    # 问题2：假设各路段流量已知，考虑拥堵效应，任意两点之间的最快路径是什么？
    # ----------------------------
    print("问题2：假设各路段流量已知，考虑拥堵效应，任意两点之间的最快路径是什么？")
    
    # 为了回答这个问题，我们使用 FW 分配后的流量作为“已知流量”
    fw_result = Frank_Wolfe_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand)
    flow_vector = fw_result['flow']
    
    # 考虑拥堵，计算当前拥堵下的行程时间
    congested_tt = [get_link_travel_time(flow_vector, i, links) for i in range(len(links))]
    for o in sorted(all_nodes):
        for d in sorted(all_nodes):
            if o == d:
                continue
            path = dijkstra_shortest_path(graph, links, o, d, congested_tt)
            path_str = print_path(links, path) if path else "不可达"
            print(f"  {o} to {d}: {path_str}")
    print()

    # ----------------------------
    # 问题3：仅 A→F 的 OD 对，分别执行AON、IA与Frank-Wolfe用户均衡分配
    # ----------------------------
    print("🔍 问题3：仅考虑 OD 对 A→F（需求=2000），执行 Frank-Wolfe 用户均衡分配")

    # 构造仅含 A→F 的需求字典
    single_od = {('A', 'F'): 2000}
    
    # 执行 AON 分配（仅此 single_od）
    aon_single_res = All_or_Nothing_Traffic_Assignment(links, graph, pos, node_names, single_od)

    # 执行 IA 分配（仅此 single_od）
    K1= 3
    K2= 1000
    ia_single_res_3 = Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, single_od, K1)
    ia_single_res_1000 = Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, single_od, K2)
    
    # 执行 FW 分配（仅此 single_od）
    fw_single_res = Frank_Wolfe_Traffic_Assignment(links, graph, pos, node_names, n_links, single_od)

    methods = [
        ("全有全无 (AON)", aon_single_res),
        ("增量分配 (IA, K=3)", ia_single_res_3),
        ("增量分配 (IA, K=1000)", ia_single_res_1000),
        ("Frank-Wolfe (UE)", fw_single_res)
    ]

    for name, res in methods:
        print(f"\n{name}:")
        print("  路段流量 (q) 与行程时间 (t):")
        for i, link in enumerate(links):
            q = res['flow'][i]
            t = get_link_travel_time(res['flow'], i, links)
            print(f"    {link['from']}→{link['to']}: q={q:6.1f}, t={t:.2f}")
        print(f"  总出行时间 (TTT): {res['total_travel_time']:.2f} veh·h")

    try:
        G = build_network(aon_single_res)
        visualize_network(G, aon_single_res['pos'], TTT=aon_single_res['total_travel_time'], 
                        title="仅考虑 A→F 时，全有全无 AON 分配结果")
        
        IA_title=f"仅考虑 A→F 时，增量分配 IA 分配结果(K = 3)" 
        G = build_network(ia_single_res_3)
        visualize_network(G, ia_single_res_3['pos'], TTT=ia_single_res_3['total_travel_time'], 
                        title=IA_title)
        
        IA_title=f"仅考虑 A→F 时，增量分配 IA 分配结果(K = 1000)" 
        G = build_network(ia_single_res_1000)
        visualize_network(G, ia_single_res_1000['pos'], TTT=ia_single_res_1000['total_travel_time'], 
                        title=IA_title)

        G = build_network(fw_single_res)
        visualize_network(G, fw_single_res['pos'], TTT=fw_single_res['total_travel_time'],
                        title="仅考虑 A→F 时，Frank-Wolfe 算法分配结果")
    except ImportError:
        print("可视化不可用。跳过该步骤。")

    # ----------------------------
    # 问题4：所有 OD 对，输出各算法下的路段流量和总出行时间
    # ----------------------------
    print("问题4：考虑所有 OD 对，比较不同分配方法的结果")
    
    # AON
    aon_res = All_or_Nothing_Traffic_Assignment(links, graph, pos, node_names, od_demand)
    
    # IA 
    K1 = 3
    K2 = 1000
    
    ia_res_3 = Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand, K1)
    ia_res_1000 = Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand, K2)
    
    # FW
    fw_res = Frank_Wolfe_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand)  

    methods = [
        ("全有全无 (AON)", aon_res),
        ("增量分配 (IA, K=3)", ia_res_3),
        ("增量分配 (IA, K=1000)", ia_res_1000),
        ("Frank-Wolfe (UE)", fw_res)
    ]

    for name, res in methods:
        print(f"\n{name}:")
        print("  路段流量 (q) 与行程时间 (t):")
        for i, link in enumerate(links):
            q = res['flow'][i]
            t = get_link_travel_time(res['flow'], i, links)
            print(f"    {link['from']}→{link['to']}: q={q:6.1f}, t={t:.2f}")
        print(f"  总出行时间 (TTT): {res['total_travel_time']:.2f} veh·h")

    try:
        G = build_network(aon_res)
        visualize_network(G, aon_res['pos'], TTT=aon_res['total_travel_time'], 
                        title="考虑所有 OD 对，全有全无 AON 分配结果")
        
        IA_title=f"考虑所有 OD 对，增量分配 IA 分配结果(K = {K1})" 
        G = build_network(ia_res_3)
        visualize_network(G, ia_res_3['pos'], TTT=ia_res_3['total_travel_time'], 
                        title=IA_title)
        IA_title=f"考虑所有 OD 对，增量分配 IA 分配结果(K = {K2})" 
        G = build_network(ia_res_1000)
        visualize_network(G, ia_res_1000['pos'], TTT=ia_res_1000['total_travel_time'], 
                        title=IA_title)

        G = build_network(fw_res)
        visualize_network(G, fw_res['pos'], TTT=fw_res['total_travel_time'],
                        title="考虑所有 OD 对，Frank-Wolfe 算法分配结果")
    except ImportError:
        print("可视化不可用。跳过该步骤。")
    
    print("\n所有测试问题已完成！")

if __name__ == '__main__':
    main()