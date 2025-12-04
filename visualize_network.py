from calculate import get_link_travel_time
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm
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
        max_flow = 1
        min_flow = 0
    else:
        max_flow = flows.max()
        min_flow = flows.min()

    # 使用对数缩放计算线宽
    min_width = 0.5
    max_width = 5.0
    widths = []
    for flow in flows:
        if flow <= 0:
            width = min_width
        else:
            log_flow = np.log1p(flow)
            log_max = np.log1p(max_flow)
            width = min_width + (max_width - min_width) * (log_flow / log_max if log_max > 0 else 0)
        widths.append(width)

    # 颜色映射
    norm = mcolors.Normalize(vmin=min_flow, vmax=max_flow)
    cmap = cm.viridis  
    edge_colors = [cmap(norm(flow)) for flow in flows]

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
        label_text = f"{u}→{v}\nq={q:.0f}\nt={t:.2f}"
        ax.annotate(
            label_text,
            xy=(label_x, label_y),
            fontsize=8.0,  # 稍小以适应三行
            color='white',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.65),
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
    plt.show()