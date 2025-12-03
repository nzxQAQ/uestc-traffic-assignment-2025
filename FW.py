# FW.py
from calculate import line_search_newton, get_link_travel_time, Beckmann_function, get_total_travel_time
from data_load import load_network_and_demand, build_graph_and_links
from assignment_utils import all_or_nothing_assignment
# ----------------------------
# Frank-Wolfe 用户均衡交通分配
# ----------------------------

def Frank_Wolfe_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand,
    max_iter=500,
    epsilon=1e-6,
    verbose=False):
    """
    按照教材描述的 Frank-Wolfe 算法实现
    
    参数:
        links: 路段列表
        graph: 网络图
        od_demand: OD需求字典
        max_iter: 最大迭代次数
        epsilon: 收敛阈值
        verbose: 是否打印详细信息
    
    返回:
        分配结果字典
    """
    
    # 步骤1: 初始化 - 自由流条件下的AON分配
    free_flow_tt = [link['t0'] for link in links]
    x = all_or_nothing_assignment(graph, links, od_demand, free_flow_tt)
    
    # 主循环
    for iteration in range(1, max_iter + 1):
        # 步骤2: 更新各路段的阻抗
        t_current = [get_link_travel_time(x, i, links) for i in range(n_links)]
        
        # 步骤3: 执行一次全有全无分配，寻找下一步迭代方向
        y = all_or_nothing_assignment(graph, links, od_demand, t_current)
        
        # 步骤4: 确定迭代步长（牛顿法）
        lambda_val = line_search_newton(x, y, links)
        lambda_val = max(0.0, min(1.0, lambda_val))  # 约束在[0,1]
        
        # 步骤5: 确定新的迭代起点
        x_new = [x[i] + lambda_val * (y[i] - x[i]) for i in range(n_links)]
        
        # 步骤6: 收敛性检验
        # 分子: sqrt(sum_a (x_a^{n+1} - x_a^n)^2)
        numerator = sum((x_new[i] - x[i]) ** 2 for i in range(n_links))
        numerator = (numerator ** 0.5) if numerator > 0 else 0
        
        # 分母: sum_a x_a^n
        denominator = sum(x)
        
        convergence_metric = numerator / denominator if denominator > 1e-12 else float('inf')
        
        # 打印迭代信息
        if verbose and (iteration % 5 == 0 or iteration <= 5 or convergence_metric < epsilon):
            Beckmann_val = Beckmann_function(x_new, links)
            TTT = get_total_travel_time(x_new, links)
            print(f"Iter {iteration:3d}: λ={lambda_val:.6f}, "
                  f"Conv={convergence_metric:.2e}, "
                  f"Beckmann_val={Beckmann_val:.2f}, TTT={TTT:.2f}")
        
        # 检查收敛
        if convergence_metric < epsilon:
            if verbose:
                print(f"✅ 收敛于迭代 {iteration}, 收敛指标 = {convergence_metric:.2e}")
            converged = True
            break
        
        x = x_new
    else:
        # 达到最大迭代次数仍未收敛
        converged = False
        if verbose:
            print(f"⚠️ 达到最大迭代次数 {max_iter}, 未收敛")
    
    # 计算最终结果
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
    
    # 4. 执行 Frank-Wolfe 分配
    FW_result = Frank_Wolfe_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand,verbose=True)
    
    print("\n=== Frank-Wolfe Flows ===")
    for i, link in enumerate(FW_result['links']):
        flow = FW_result['flow'][i]
        t_val = get_link_travel_time(FW_result['flow'], i, FW_result['links'])
        print(f"{link['from']}->{link['to']}: flow={flow:.2f}, "
                f"capacity={link['capacity']}, t0={link['t0']:.2f}, t={t_val:.2f}")
    
    print(f"\nTotal Travel Time (FW-TTT): {FW_result['total_travel_time']:.2f} Beckmann_value: {FW_result['Beckmann_value']:.2f} ")
    
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
        
        visualize_network(G, FW_result['pos'], TTT=FW_result['total_travel_time'],
                        title="Frank-Wolfe Assignment Result")
    except ImportError:
        print("visualize_network not available. Skipping visualization.")