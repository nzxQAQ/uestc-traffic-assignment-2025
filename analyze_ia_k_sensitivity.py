# analyze_IA_k_sensitivity.py
import matplotlib.pyplot as plt
from IA import Incremental_Traffic_Assignment
from data_load import load_network_and_demand, build_graph_and_links

def analyze_k_sensitivity(network_file='data/network.json', 
                         demand_file='data/demand.json',
                         max_k=200,
                         step_size=50):
    """
    分析K值对增量交通分配总行程时间的敏感性
    
    Args:
        network_file: 网络数据文件路径
        demand_file: 需求数据文件路径
        max_k: 最大K值
        step_size: K值采样步长（为了减少计算时间）
    """
    
    # 1. 加载数据
    print("加载数据...")
    network, demand = load_network_and_demand(network_file, demand_file)
    
    # 2. 构建图结构
    print("构建网络图...")
    graph, links, pos, node_names, n_links = build_graph_and_links(network)
    
    # 3. 整理OD需求
    od_demand = {}
    for o, d, amt in zip(demand['from'], demand['to'], demand['amount']):
        od_demand[(o, d)] = od_demand.get((o, d), 0) + amt
    
    # 4. 测试不同K值
    print(f"分析K值敏感性 (K从1到{max_k}, 步长{step_size})...")
    
    k_values = []
    ttt_values = []
    
    # 可以选择测试特定的K值点（例如小值区域更密集采样）
    test_k_values = list(range(1, max_k+1, step_size))
    
    for k in test_k_values:
        print(f"  计算 K={k}...", end="")
        
        # 执行增量分配
        result = Incremental_Traffic_Assignment(
            links, graph, pos, node_names, n_links, od_demand, K=k
        )
        
        k_values.append(k)
        ttt_values.append(result['total_travel_time'])
        print(f" TTT={result['total_travel_time']:.2f}")
    
    # 5. 绘制折线图
    plt.figure(figsize=(12, 7))
    
    # 主图：K与总行程时间关系
    plt.plot(k_values, ttt_values, 'b-o', linewidth=2, markersize=4, label='Total Travel Time')
    
    # 标记关键点
    plt.scatter(k_values[0], ttt_values[0], color='red', s=100, zorder=5, 
                label=f'K=1: {ttt_values[0]:.2f}')
    plt.scatter(k_values[-1], ttt_values[-1], color='green', s=100, zorder=5, 
                label=f'K={max_k}: {ttt_values[-1]:.2f}')
    
    # 寻找最小TTT对应的K值
    min_ttt_idx = ttt_values.index(min(ttt_values))
    plt.scatter(k_values[min_ttt_idx], ttt_values[min_ttt_idx], color='orange', s=100, zorder=5, 
                label=f'Min TTT (K={k_values[min_ttt_idx]}): {ttt_values[min_ttt_idx]:.2f}')
    
    # 设置图形属性
    plt.xlabel('K Value (Number of Increments)', fontsize=14)
    plt.ylabel('Total Travel Time (veh·h)', fontsize=14)
    plt.title(f'Sensitivity Analysis: K vs Total Travel Time (Incremental Assignment)', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # 添加文本说明
    plt.text(0.8, 0.8, f'K tested: {len(test_k_values)} points\nMin K={k_values[min_ttt_idx]}\nTTT range: {min(ttt_values):.2f} - {max(ttt_values):.2f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    
    # 6. 保存结果
    output_file = 'ia_k_sensitivity_analysis.png'
    plt.savefig(output_file, dpi=300)
    print(f"\n分析完成！图表已保存为: {output_file}")
    
    # 显示图表
    plt.show()
    
    # 7. 打印总结信息
    print("\n=== 敏感性分析总结 ===")
    print(f"测试的K值范围: 1 ~ {max_k}")
    print(f"总行程时间范围: {min(ttt_values):.2f} ~ {max(ttt_values):.2f} veh·h")
    print(f"最大TTT (K=1): {ttt_values[0]:.2f} veh·h")
    print(f"稳定TTT (K={max_k}): {ttt_values[-1]:.2f} veh·h")
    print(f"最小TTT出现在 K={k_values[min_ttt_idx]}: {ttt_values[min_ttt_idx]:.2f} veh·h")
    
    # 计算变化率
    if len(ttt_values) > 1:
        change_percent = ((ttt_values[-1] - ttt_values[0]) / ttt_values[0]) * 100
        print(f"K从1到{max_k}的总变化率: {change_percent:.2f}%")
    
    return k_values, ttt_values

if __name__ == '__main__':
    # 可以调整参数
    k_values, ttt_values = analyze_k_sensitivity(
        network_file='data/network.json',
        demand_file='data/demand.json',
        max_k=60,
        step_size=1  # 为了快速测试，可以增大步长
    )