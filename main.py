from calculate import get_link_travel_time
from FW import frank_wolfe_traffic_assignment

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
          f" Beckmann_value: {FW_result['Beckmann_value']:.2f} ")
    
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