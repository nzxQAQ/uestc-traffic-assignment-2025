from calculate import get_link_travel_time

import os
import re
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def build_network(res):
    G = nx.DiGraph()
    for node in res['node_names']:
        G.add_node(node)
    for i, link in enumerate(res['links']):
        u, v = link['from'], link['to']
        q = res['flow'][i]
        t = get_link_travel_time(res['flow'], i, res['links'])
        G.add_edge(u, v, Q=q, T=t)
    return G

def visualize_network(G, pos_dict, TTT, title="Traffic Assignment Result"):

    plt.figure(figsize=(12, 8))
    
    edges = list(G.edges())
    flows = np.array([G[u][v]['Q'] for u, v in edges])
    
    if len(flows) == 0:
        # 如果没有边，返回空列表
        return []  # 或者根据调用方式调整

    # 计算最大最小流量
    max_flow = flows.max()
    min_flow = flows.min()

    # 使用对数缩放计算线宽
    min_width = 0.5
    max_width = 5.0
    widths = []

    # 处理所有流量都为零的特殊情况
    if max_flow <= 0:
        # 所有流量都为零，使用最小线宽
        widths = [min_width] * len(flows)
    else:
        # 计算对数基准
        log_max = np.log1p(max_flow)
        
        for flow in flows:
            if flow <= 0:
                width = min_width
            else:
                log_flow = np.log1p(flow)
                # 确保不除以零，并且比例在[0,1]范围内
                width_ratio = log_flow / log_max if log_max > 0 else 0
                width_ratio = np.clip(width_ratio, 0, 1)  # 确保在合理范围内
                width = min_width + (max_width - min_width) * width_ratio
            widths.append(width)

    # 修改颜色映射部分
    norm = mcolors.Normalize(vmin=0, vmax=max_flow*1.1)
    cmap = cm.Blues

    # 设置颜色下限：最小流量也显示至少35%的颜色强度
    edge_colors = []
    for flow in flows:
        if flow <= 0:
            # 零流量使用灰色虚线
            color = (0.8, 0.8, 0.8, 0.3)  # 浅灰色，半透明
        else:
            # 设置颜色强度下限为0.35（避免太淡）
            normalized_flow = max(0.35, flow / (max_flow * 1.1))
            # 确保不超过1.0
            normalized_flow = min(1.0, normalized_flow)
            color = cmap(normalized_flow)
        edge_colors.append(color)

    # 计算标签偏移（避免双向边标签重叠）
    label_offsets = {}
    for u, v in G.edges():
        if G.has_edge(v, u):
            x1, y1 = pos_dict[u]
            x2, y2 = pos_dict[v]
            dx = x2 - x1
            dy = y2 - y1
            length = (dx*dx + dy*dy) ** 0.5
            if length > 0:
                d = 1.8
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

    # 绘制边
    nx.draw_networkx_edges(
        G, pos_dict,
        edgelist=edges,
        width=widths,
        edge_color=edge_colors,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1',
        alpha=0.9
    )

    # 绘制三行标签：路段名 + 流量 + 行程时间
    for i, (u, v) in enumerate(edges):
        q = G[u][v]['Q']
        t = G[u][v]['T']
        offset = label_offsets[(u, v)]
        
        mid_x = (pos_dict[u][0] + pos_dict[v][0]) / 2
        mid_y = (pos_dict[u][1] + pos_dict[v][1]) / 2
        label_x = mid_x + offset[0]
        label_y = mid_y + offset[1]
        
        # 三行文本：路段名 + 流量 + 行程时间
        label_text = f"{u}→{v}\nq={q:.2f}\nt={t:.2f}"
        ax.annotate(
            label_text,
            xy=(label_x, label_y),
            fontsize=10.0,  # 稍小以适应三行
            color='white',
            ha='center',
            va='center',
            bbox=dict(
            boxstyle="round,pad=0.3",  # pad 控制内边距
            facecolor="black", 
            alpha=0.75,
            linewidth=0.8  # 边框线宽
            ),
            zorder=5
        )

    # 节点
    nx.draw_networkx_nodes(G, pos_dict, node_size=800, node_color='lightgray', edgecolors='black')
    nx.draw_networkx_labels(G, pos_dict, font_size=12, font_weight='bold')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Traffic Flow (Q)', fontsize=12)

    plt.gca().invert_yaxis()
    plt.title(title, fontsize=14)
    
    # 第一个文本框：行程时间函数公式
    plt.figtext(
        0.5, 0.26,  # 上方的位置
        "行程时间函数 " + r"$t(q) = t_0 \left(1 + \frac{q}{C}\right)^2$",
        fontsize=13,
        ha='center',
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="lightblue",  # 不同颜色区分
            edgecolor="steelblue",
            alpha=0.8
        )
    )
    
    # === 显示 TTT ===
    if TTT is not None:
        label_text = (r"Total Travel Time (TTT) = $\sum q \cdot t$ = " + f"{TTT:.2f} veh·h")

        plt.figtext(
            0.5, 0.2,  # x=0.5（居中），y=0.2
            label_text,
            fontsize=12,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray")
        )

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    
    # ========== 保存图片功能 ==========
    try:
        # 使用标题作为文件名，清理非法字符
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)  # 替换非法字符
        safe_title = safe_title.replace(' ', '_')  # 空格替换为下划线
        save_path = f"./images/{safe_title}.png"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ 图片已保存: {save_path}")
        
    except Exception as e:
        print(f"❌ 保存图片失败: {str(e)}")

    plt.show()
    plt.close()

def print_path(links, path_link_indices):
    """将路径的 link 索引列表转换为节点序列"""
    if not path_link_indices:
        return "No path"
    nodes = []
    # 从起点开始回溯（path 是反向的：[last_link, ..., first_link]）
    current_node = links[path_link_indices[-1]]['from']
    nodes.append(current_node)
    for lid in reversed(path_link_indices):
        next_node = links[lid]['to']
        nodes.append(next_node)
        current_node = next_node
    return " → ".join(nodes)