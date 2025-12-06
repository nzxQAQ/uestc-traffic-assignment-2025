#import "./lib.typ": *
#show: project.with(
  title: "电子科技大学《交通规划原理》课程报告",
  author: "倪子煊2023091203002",
  date: auto,
  abstract: [
    本文针对《交通规划原理》课程要求，设计并实现了一套交通分配计算软件，支持全有全无（AON）、增量分配（IA）与 Frank-Wolfe 用户均衡（FW）三种算法。软件基于 Python 开发，采用模块化架构，实现了路网与需求数据解析、交通分配计算、结果定量评估（总出行时间 TTT）及可视化功能。通过对给定测试路网在单OD对与全OD矩阵场景下的实验，验证了各算法特性：AON 忽略拥堵导致非均衡解；IA 通过分批加载逼近均衡，其精度受分割数 $K$ 影响；FW 算法严格收敛至用户均衡状态。本工作不仅满足课程全部功能要求，也为理解 Wardrop 均衡原理提供了实践基础。
  ],
  keywords: ("交通分配", "路阻函数", "全有全无分配", "增量分配", "用户均衡", "Frank-Wolfe算法"),
)

= 交通分配问题的背景与意义

自20世纪50年代起，以美国底特律、芝加哥等大都市圈为代表的系统性交通调查与规划研究，推动了现代交通需求预测理论体系的建立。其中，“四阶段法”——即交通产生（Trip Generation）、交通分布（Trip Distribution）、方式划分（Modal Split）和交通流分配（Traffic Assignment）——逐渐成为交通规划领域的标准范式。在这一框架中，#strong[交通流分配]作为连接出行需求与路网供给的关键环节，其核心任务是将已知的起讫点（Origin–Destination, OD）交通量，依据路网拓扑结构与运行特性，合理地分配至各路段上，从而推演出全网流量分布与行程时间，为路网性能评估、扩容决策及政策模拟提供定量依据。

早期的交通分配模型多采用#strong[全有全无（All-or-Nothing, AON）]方法。该方法假设路网处于自由流状态，路段阻抗（通常以行驶时间表示）恒定不变，每个OD对的全部出行量均被指派至唯一最短路径上。尽管该方法计算简便，在低密度或非拥挤路网中具有一定适用性，但其忽略了交通流与路段阻抗之间的动态反馈关系，难以反映真实城市路网中的拥堵效应。

为克服上述局限，Wardrop（1952）提出了具有里程碑意义的网络平衡原理，奠定了现代交通分配理论的基础。其中，#strong[第一原理（用户均衡，User Equilibrium, UE）]指出：在均衡状态下，任意OD对之间所有被实际使用的路径具有相等且最小的行程时间；而未被使用的路径其行程时间不小于该最小值。这一原理深刻揭示了个体理性选择与系统整体状态之间的内在一致性，成为静态确定性交通分配模型的核心假设。

在建模实践中，路段阻抗通常通过#strong[路阻函数（Volume–Delay Function）]进行量化。其中最具代表性的是由美国公共道路局（Bureau of Public Roads, BPR）提出的BPR函数：

$ t_a = t_0 [1 + α (q_a / c_a)^β] $

式中，$t_a$ 为路段 $a$ 的实际行程时间，$t_0$ 为其自由流行程时间，$q_a$ 为路段流量，$c_a$ 为通行能力，$alpha$ 与 $beta$ 为经验参数。该函数有效刻画了流量增长引发的边际阻抗上升，是用户均衡模型中最常用的阻抗表达形式。

随着研究深入，学者们进一步认识到出行者对路径阻抗的感知存在异质性与不确定性，由此发展出#strong[随机用户均衡（Stochastic User Equilibrium, SUE）]理论。此外，面向时变需求与动态响应的#strong[动态交通分配（Dynamic Traffic Assignment, DTA）]以及融合多种交通方式的#strong[多模式网络建模]亦成为前沿方向。然而，基于Wardrop UE原理的静态确定性模型，因其理论严谨、算法成熟、计算效率较高，至今仍是交通规划实务中的主流工具。

因此，本报告聚焦于编程实现三类典型交通分配算法——全有全无（AON）、增量分配（Incremental Assignment, IA）与基于Frank-Wolfe算法的用户均衡分配——并在标准测试路网上进行对比分析，旨在深入理解不同模型对路网拥挤效应及出行者路径选择行为的刻画能力，为交通规划与管理提供科学支撑。

= 软件的功能模块划分与设计思路

为确保代码结构清晰、功能可扩展且便于验证，本软件采用“高内聚、低耦合”的模块化设计理念，将整体流程划分为六个相互协作但职责分明的核心模块。各模块通过标准化接口交互，共同完成从数据输入、算法执行到结果可视化的完整工作流。具体模块如下：

#strong[（1）数据加载与路网构建模块（`data_load.py`）]

该模块负责解析外部输入并构建内部路网表示：
- `load_network_and_demand()`：读取JSON格式的`network.json`（含节点坐标、路段连接关系、通行能力与限速）与`demand.json`（含OD对及其出行量）；
- `build_graph_and_links()`：基于欧氏距离计算路段长度 $l$，进而推导自由流行程时间 $t_0 = l / v_max}$；将每条无向路段扩展为两条方向相反的有向边，以支持双向通行；最终生成邻接表形式的图结构 `graph` 与包含物理属性的边列表 `links`，为后续算法提供统一数据基础。

#strong[（2）基础算法工具模块（`assignment_utils.py`）]

该模块封装路径搜索函数与全有全无分配函数：
- `dijkstra_shortest_path()`：基于给定的路段阻抗向量，采用Dijkstra算法求解最短路径，并返回路径上各边的索引序列；
- `all_or_nothing_assignment()`：遍历所有OD对，调用最短路径函数，将全部需求一次性分配至最短路径上，返回各边流量向量。该函数是AON算法的核心，同时也是IA与FW算法中迭代步骤的基础组件。

#strong[（3）数学计算模块（`calculate.py`）]

该模块集中处理所有与交通流理论相关的数值计算：
- `get_link_travel_time()`：计算路阻函数

  $ t(q) = t_0 (1 + q/c)^2 $

  其中路段的行程时间 $t$ 是流量 $q$ 的函数。 $t_0$ 是路段的自由流行程时间，等于路段长度 $l$ 除以限速 $v_max$。$q$ 是路段流量，$c$ 是路段通行能力，即出现拥堵前最大能够承受的流量

- `get_total_travel_time()`：路网总出行时间（Total Travel Time, TTT）是路网上所有路段流量 $q$ 与行程时间 $t$ 乘积的总和，定义为

  $ T T T = sum q dot t $

- `Beckmann_function()`：计算Frank-Wolfe算法所优化的目标函数——Beckmann势能函数，当其取得最小值时，对应达到均衡状态；
- `line_search_newton()`：采用Newton-Raphson法精确求解FW迭代中的最优步长 $lambda in [0,1]$，确保收敛效率与数值稳定性。

#strong[（4）交通分配算法模块（`AON.py`, `IA.py`, `FW.py`）]

该模块实现了三种具有代表性的分配策略，均遵循统一输出接口：
- 全有全无分配（AON）：只执行一次AON分配，忽略流量对阻抗的反馈；
- 增量分配（IA）：将总OD需求等分为 $K$ 份，逐次执行AON分配并累加流量，每次分配前更新路段阻抗，逐步逼近均衡状态；
- Frank-Wolfe用户均衡分配（FW）：以AON结果为初始解，迭代执行“阻抗更新 → AON方向搜索 → 步长优化 → 解更新 → 收敛检验”五步流程，直至满足预设精度 $epsilon$，理论上可收敛至Wardrop用户均衡解。

#strong[（5）可视化模块（`visualize_network.py`）]

该模块将抽象的分配结果转化为直观的图片：
- 根据分配结果`res`，构建NetworkX有向图 $G$（边属性含流量 $Q$ 与行程时间 $T$）；
- 采用对数缩放策略映射流量至线宽，Blues色谱映射流量至颜色，增强视觉辨识度；
- 在每条边上叠加三行标签（路段标识、流量 $q$、行程时间 $t$），并通过垂直偏移避免双向边标签重叠；
- 绘制带色标的路网图，并在图中嵌入TTT数值框，实现结果的一站式呈现。

#strong[（6）主程序（`main.py`）]

作为系统入口，该模块整合前述功能，系统性验证课程要求的各项能力：
- 分别在自由流与拥堵状态下查询任意两点间的最快路径；
- 针对单一OD对（如A→F）执行FW均衡分配，分析多路径使用情况；
- 对全网OD需求，对比AON、IA与FW三种算法的路段流量分布与所有出行者的总行程时间TTT指标；
- 自动调用可视化模块生成结果图示，辅助分析与展示。

#summary[上述模块化架构有效实现了算法逻辑、数据处理与可视化展示的解耦，不仅提升了代码的可读性与可维护性，也为未来引入更复杂模型（如SUE或动态分配）预留了良好的扩展接口。]

= 开发环境与依赖

本软件基于Python语言开发，注重可复现性与跨平台兼容性。具体开发与运行环境如下：

#strong[1. 编程语言]
- Python 3.8 或更高版本

#strong[2. 核心依赖库（见 `requirements.txt`）]
- `numpy==1.24.2`：提供高效的向量化运算，支撑流量向量、阻抗计算等数值密集型操作；
- `networkx==3.1`：用于构建与操作有向图，支持节点/边属性存储，是路网建模的基础；
- `matplotlib==3.7.5`：实现高质量的路网可视化，支持自定义字体（含中文）、颜色映射、线宽控制及文本标注。
- 注意，`networkx` 与 `matplotlib` 只与结果可视化有关。如果不需要可视化，可忽略该依赖。

上述依赖组合已在Windows 11操作系统下完成兼容性验证。

#strong[3. 代码获取与运行]

完整源代码已开源托管于GitHub仓库：

https://github.com/nzxQAQ/uestc-traffic-assignment-2025.git

可通过以下命令快速部署本软件：
```bash
pip install -r requirements.txt
python main.py
```

= 关键算法代码片段
为确保算法实现的透明性与可复现性，本节选取若干核心函数进行展示与说明。受限于篇幅，部分函数在文中会以伪代码或者省略号替代，完整代码请访问GitHub仓库。


#strong[1.最短路径搜索 与 全有全无分配（`assignment_utils.py`）]

Dijkstra算法是求解最短路径的基础。本模块实现返回路径上的边索引列表，便于后续流量累加：

```python
def dijkstra_shortest_path(graph, links, origin, dest, link_travel_times):
    """基于给定阻抗执行Dijkstra，返回最短路径的边索引列表"""
    # ...（初始化距离字典、优先队列等）
    while pq:
        d, u = heapq.heappop(pq)
        if u == dest: break
        for v, link_idx in graph[u]:
            tt = link_travel_times[link_idx]
            new_dist = d + tt
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev_link[v] = link_idx
                heapq.heappush(pq, (new_dist, v))
    # ...（回溯路径）
    return path_links
```

`all_or_nothing_assignment` 函数则遍历所有OD对，调用上述最短路径函数，完成一次完整的全有全无分配：

```python
def all_or_nothing_assignment(graph, links, od_demand, link_travel_times):
    """执行一次全有全无（AON）分配"""
    n_links = len(links)
    y = [0.0] * n_links
    for (orig, dest), demand_val in od_demand.items():
        if demand_val <= 0: continue
        path = dijkstra_shortest_path(graph, links, orig, dest, link_travel_times)
        if path is None:
            print(f"Warning: No path from {orig} to {dest}")
            continue
        for lid in path:
            y[lid] += demand_val
    return y
```

#strong[2.路阻函数t、总出行时间TTT 以及 Beckmann函数（`calculate.py`）]

本项目采用指定的路阻函数，其形式为 $ t = t_0 (1 + q/C)^2 $

路阻函数的计算在 `get_link_travel_time` 中实现：

```python
def get_link_travel_time(flow_vector, link_idx, links):
    """路阻函数/行程时间函数：t = t0 * (1 + (Q/C))^2"""
    t0 = links[link_idx]['t0']
    C = links[link_idx]['capacity']
    Q = flow_vector[link_idx]
    return t0 * (1 + (Q / C) ) ** 2
```
路网总出行时间`（Total Travel Time, TTT）`被定义为所有路段上“流量q与行程时间t之积”的总和，即：

$ T T T = sum q dot t $

总出行时间的计算在 `get_total_travel_time` 中实现：

```python
def get_total_travel_time(flow_vector, links):
    """计算所有出行者的总行程时间TTT。
       TTT = Σ(q_a * t_a)
    """
    total_travel_time = 0.0
    for i in range(len(links)):
        q = flow_vector[i]
        if q <= 0: continue
        t = get_link_travel_time(flow_vector, i, links)
        total_travel_time += q * t
    return total_travel_time
```

此外，为支持 Frank-Wolfe 算法的理论验证与收敛性分析，本模块还实现了 #strong[Beckmann 函数]（Beckmann Function）。该函数由 Beckmann 等人（1956）提出，是用户均衡（UE）问题的等价优化目标：当且仅当流量分配使 Beckmann 函数取最小值时，Wardrop 第一原理成立。

对于本项目的路阻函数 $t = t_0 (1 + q/C)^2$ ，其 Beckmann 函数可通过积分得到：
$
  Z(q)
  = sum_a t_0_a ( q_a + q_a^2/C_a + q_a^3/(3 C_a^2) )
$
其中：

$t_0_a$ ：路段  $a$  的自由流行程时间（单位：h）

$q_a$ ：路段  $a$  上的交通流量（单位：veh/h），即分配给该路段的出行量；

$C_a$ ：路段  $a$ 的通行能力（单位：veh/h）
```python
def Beckmann_function(flow_vector, links):
    """ Beckmann函数Z(x)，是对路阻函数的积分"""
    total = 0.0
    for i, q in enumerate(flow_vector):
        C = links[i]['capacity']
        t0 = links[i]['t0']
        # 积分 ∫₀^q t0*(1 + (x/C))² dx = t0*(q + q²/C + q³/(3*C²))
        total += t0 * (q + (q ** 2) / C + (q ** 3) / (3 * C ** 2))
    return total
```
#strong[3.最优步长求解：Newton-Raphson 精确搜索（`calculate.py`）]

在 Frank-Wolfe 算法中，每一步迭代需确定最优步长 $lambda in [0,1]$，以最小化 Beckmann 函数 $Z(q)$ 沿当前搜索方向（即从当前解 $x$ 指向 AON 解 $y$ 的方向）的值。

教材中常采用#strong[二分法]（Bisection Method）求解该一维优化问题，而本项目则采用#strong[牛顿-拉弗森法]（Newton-Raphson Method）实现高精度线搜索，后者在本场景下具有显著优势，主要体现在以下三方面：

#strong[收敛速度：牛顿法 vs 二分法]

- 牛顿法：利用目标函数的一阶导数 $phi'(lambda)$ 与二阶导数 $phi''(lambda)$ 构造局部二次近似，在极小值点附近具有#emph[二次收敛性]（quadratic convergence）。这意味着误差平方级下降——例如，若当前误差为 $10^(-2)$，下一步可能降至 $10^(-4)$。
- 二分法：仅依赖一阶导数的符号变化进行区间缩放，收敛速度为#emph[线性]（linear），每次迭代仅将误差减半。要达到 $10^(-6)$ 精度，通常需约 20 次迭代,而牛顿法往往只需 3-5 次迭代即可达到同等精度。

在交通分配中，Beckmann 函数是严格凸且无限可微的，完全满足牛顿法快速收敛的前提条件。

下面省略繁琐的数学公式推导，直接给出牛顿法的代码：

```python
def line_search_newton(x, y, links, max_iter=10, tol=1e-8):
    """
    使用 Newton-Raphson 方法精确求解最优步长 alpha ∈ [0, 1]
    利用 phi'(alpha) = sum( (y_i - x_i) * t(q_i(alpha)) )
    """
    n = len(x)
    d = [y[i] - x[i] for i in range(n)]  # direction

    # 特殊情况：初始零解
    if all(v == 0 for v in x):
        return 1.0

    alpha = 0.5  # 初始猜测

    for _ in range(max_iter):
        alpha = max(0.0, min(1.0, alpha))  # 投影到可行域
        q = [(1 - alpha) * x[i] + alpha * y[i] for i in range(n)]

        # 计算一阶导数 phi'
        phi_prime = sum(
            d[i] * get_link_travel_time(q, i, links)
            for i in range(n) if abs(d[i]) > 1e-12
        )

        if abs(phi_prime) < tol:
            break

        # 计算二阶导数 phi''
        phi_double_prime = 0.0
        for i in range(n):
            if abs(d[i]) < 1e-12:
                continue
            C, t0 = links[i]['capacity'], links[i]['t0']
            if C <= 0: continue
            dt_dq = 2 * t0 / C * (1 + q[i] / C)
            phi_double_prime += d[i]**2 * dt_dq

        if phi_double_prime <= 0:
            phi_double_prime = 1e-12  # 防止除零或非凸情况

        alpha_new = alpha - phi_prime / phi_double_prime
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return max(0.0, min(1.0, alpha))
```

#strong[4.路网数据加载与图结构构建的关键实现（`data_load.py`）]

本模块的核心任务是将 JSON 数据文件转化为可处理的图结构。其关键设计体现在以下三个方面：

#strong[4.1自由流行程时间 $t_0$ 的推导]

路段自由流阻抗并非人为设定，而是基于几何与最大限速来计算：
$ t_0 = l/v_max $
```python
def euclidean_distance(u, v):
    # 计算两点之间的欧式距离
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
...

```

#strong[4.2 双向有向边建模]

现实道路通常支持双向通行，因此每条无向路段被显式拆分为两条方向相反的有向边：
```python
links.append({
    'from': u,'to': v,
    'length': length,
    'capacity': capacity,
    'speedmax': speedmax,
    't0': t0
})

links.append({
    'from': v,'to': u,
    'length': length,
    'capacity': capacity,
    'speedmax': speedmax,
    't0': t0
})

```

#strong[4.3 图存储结构设计——邻接表]

为兼顾空间利用率与访问效率，用邻接表来构建图`graph`：

```python
# 构建邻接表
graph = defaultdict(list)
for idx, link in enumerate(links):
    graph[link['from']].append((link['to'], idx))
```

#strong[5.全有全无（All-or-Nothing, AON）交通分配（`AON.py`）]

全有全无（All-or-Nothing, AON）分配是交通流分配中最基础的模型，其核心假设为：所有出行者均基于自由流行程时间 $t_0$ 选择最短路径，并将OD对的全部需求一次性加载至该路径上。本模块不仅实现了这一经典方法，还为增量分配（IA）与Frank-Wolfe（FW）算法提供了可复用的基础组件。

调用`assignment_utils.py`中的`all_or_nothing_assignment()`，完成一次全有全无分配
```python
def All_or_Nothing_Traffic_Assignment(links, graph, pos, node_names, od_demand):
    """执行基于自由流时间的全有全无交通分配"""
    free_flow_tt = [link['t0'] for link in links]
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
```

#strong[6.增量分配（Incremental Assignment, IA）（`IA.py`）]

增量分配（Incremental Assignment, IA）是介于全有全无（AON）与用户均衡（UE）之间的一种启发式方法。其核心思想是：将总OD需求划分为 $K$ 个等份（或者不等份），逐次执行AON分配，并在每次分配前根据当前累积流量更新路段阻抗，从而部分反映交通流与行程时间之间的动态反馈。本模块实现了这一策略，并通过参数 $K$ 控制逼近均衡的精细程度。

```python
def Incremental_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand, K=1000):
    # 将总OD需求划分为 K 个等份
    step_demand = {od: amt / K for od, amt in od_demand.items()}

    # 初始化零流量
    x = [0.0] * n_links

    for k in range(1, K + 1):
        # 基于当前流量 x 计算实时行程时间
        t_current = [get_link_travel_time(x, i, links) for i in range(len(links))]

        # 执行一次 AON 分配当前小份需求
        y_k = all_or_nothing_assignment(graph, links, step_demand, t_current)

        # 累加流量
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
```

#strong[分析：增量分配（IA）算法特性与局限性]

优点：相比 全有全无（AON），增量分配（IA）能部分捕捉拥堵效应，结果更接近现实；实现简单，仅需在 AON 基础上增加外层循环。

当 K→∞ 时，增量分配（IA）理论上趋近于用户均衡解。在本项目中，取 K=30 其实就已能获得较均衡的解，随着K的增大，结果会更加收敛于用户均衡(UE)解。

局限：仍为启发式方法，不保证满足 Wardrop 第一原理；最终解依赖于 K 的取值和加载顺序（本实现为均匀顺序加载）。



#strong[7.基于Frank-Wolfe算法的用户均衡分配（`FW.py`）]

Frank-Wolfe（FW）算法是求解 Wardrop 用户均衡（User Equilibrium, UE）问题的经典方法。其理论基础是：UE 状态等价于 Beckmann 函数 $Z(q)$ 的全局最小值。

本模块严格遵循教材所述迭代流程，结合 Newton-Raphson 精确线搜索，实现了高精度、高效率的均衡分配。

```python
def Frank_Wolfe_Traffic_Assignment(links, graph, pos, node_names, n_links, od_demand, max_iter=500, epsilon=1e-6, verbose=False):
    # 步骤1: 初始化 —— 自由流下的 AON 分配
    free_flow_tt = [link['t0'] for link in links]
    x = all_or_nothing_assignment(graph, links, od_demand, free_flow_tt)

    # 主迭代循环
    for iteration in range(1, max_iter + 1):
        # 步骤2: 更新路段阻抗
        t_current = [get_link_travel_time(x, i, links) for i in range(n_links)]

        # 步骤3: 求解线性主问题（AON方向）
        y = all_or_nothing_assignment(graph, links, od_demand, t_current)

        # 步骤4: 精确线搜索（Newton法求最优步长 λ）
        lambda_val = line_search_newton(x, y, links)
        lambda_val = max(0.0, min(1.0, lambda_val))  # 确保 lambda_val 在 [0,1] 上

        # 步骤5: 确定新的迭代起点
        x_new = [x[i] + lambda_val * (y[i] - x[i]) for i in range(n_links)]

        # 步骤6: 收敛性检验：convergence_metric < epsilon
        numerator = sqrt(sum((x_new[i] - x[i])**2 for i in range(n_links)))
        denominator = sum(x)
        convergence_metric = numerator / denominator if denominator > 1e-12 else float('inf')

        if convergence_metric < epsilon:
            break

        # 更新迭代起点
        x = x_new

    # 主循环结束，计算结果
    final_TTT = get_total_travel_time(x, links)
    final_Beckmann_value = Beckmann_function(x, links)

    return {
            'flow': x,
            'total_travel_time': final_TTT,
            'Beckmann_value': final_Beckmann_value,
            'iterations': iteration,
            'converged': converged,
            'graph': graph,
            'links': links,
            'pos': pos,
            'node_names': node_names
        }
```

#strong[算法优势与理论保证]

- 理论完备性：项目指定的阻抗函数所对应的 Beckmann 函数严格凸且光滑，FW 算法全局可收敛至用户均衡(UE)解；
- 计算高效性：使用Newton 法搜索最优步长，适用于大规模网络；
- 精度可控：通过迭代精度 $epsilon$ 判断收敛，平衡计算成本与解的质量。

= 测试结果与分析
为测试软件，本项目包含一个测试场景，其路网结构`network.json`和交通需求`demand.json`

#problem[不考虑拥堵，任意两点间的最快的路径是什么？]
这个问题其实是为了验证我们的Dijkstra算法是否正确。

运行main.py后，可以看到控制台打印了任意两点间的最快路径：（注：如果有多条路径时间相等，仅打印其中一条）
```python
A to B: A → B
A to C: A → B → C
A to D: A → B → D
A to E: A → B → C → E
A to F: A → B → C → E → F
A to G: A → B → D → G
B to A: B → A
B to C: B → C
B to D: B → D
B to E: B → C → E
B to F: B → C → E → F
B to G: B → D → G
C to A: C → B → A
C to B: C → B
C to D: C → B → D
C to E: C → E
C to F: C → E → F
C to G: C → B → D → G
D to A: D → B → A
D to B: D → B
D to C: D → B → C
D to E: D → E
D to F: D → E → F
D to G: D → G
E to A: E → C → B → A
E to B: E → C → B
E to C: E → C
E to D: E → D
E to F: E → F
E to G: E → D → G
F to A: F → E → C → B → A
F to B: F → E → C → B
F to C: F → E → C
F to D: F → E → D
F to E: F → E
F to G: F → E → D → G
G to A: G → D → B → A
G to B: G → D → B
G to C: G → D → B → C
G to D: G → D
G to E: G → D → E
G to F: G → D → E → F
```

由于不考虑拥堵，所以流量对行程时间没有影响，所有路段上的行程时间均为自由流行程时间t0。下面是路网可视化结果。

#figure(
  image("../images/不考虑拥堵时，每个路段上的行程时间恒为自由流时间_t0.png", width: 90%),
  caption: [
    不考虑拥堵时，仅展示各路段上的自由流行程时间t0
  ],
)

值得注意的是：明明BE之间有直接连接的道路，但是BE之间的最快路径却是B→C→E。

这是因为BC与CE道路上的限速为60，而BE上的限速为30。

实际计算后发现:
- B→C→E 的行程时间为0.17+0.17=0.34小时，
- B→E的行程时间为0.47小时
换言之：BC与CE是“快车道”！

#problem[假设各路段流量已知，考虑拥堵效应，任意两点之间的最快路径是什么？]
问题2在问题1的基础上，进一步验证我们的行程时间函数函数是否正确实现。为了回答这个问题，我们使用 FW 分配后的流量作为“已知流量”。

#strong[2.1 控制台打印结果]

运行main.py后，可以看到控制台打印了任意两点间最快的路径：

```python
A to B: A → B
A to C: A → B → C
A to D: A → B → D
A to E: A → B → C → E
A to F: A → B → C → E → F
A to G: A → B → D → G
B to A: B → A
B to C: B → C
B to D: B → D
B to E: B → C → E
B to F: B → C → E → F
B to G: B → D → G
C to A: C → B → A
C to B: C → B
C to D: C → B → D
C to E: C → E
C to F: C → E → F
C to G: C → B → D → G
D to A: D → B → A
D to B: D → B
D to C: D → E → C
D to E: D → E
D to F: D → E → F
D to G: D → G
E to A: E → C → B → A
E to B: E → C → B
E to C: E → C
E to D: E → D
E to F: E → F
E to G: E → D → G
F to A: F → E → C → B → A
F to B: F → E → C → B
F to C: F → E → C
F to D: F → E → D
F to E: F → E
F to G: F → E → D → G
G to A: G → D → B → A
G to B: G → D → B
G to C: G → D → E → C
G to D: G → D
G to E: G → D → E
G to F: G → D → E → F
```

#strong[2.2 路网可视化结果]

#figure(
  image("../images/考虑所有_OD_对，Frank-Wolfe_算法分配结果.png", width: 90%),
  caption: [
    考虑拥堵，且使用 FW 分配后的流量作为“已知流量”，展示各路段的行程时间t
  ],
)

值得注意的是：在问题1中，D到C之间的最快路径是D → B → C 和 D → E → C，行程时间均为0.33+0.17= 0.5小时。

而在问题2中，程序打印的D到C之间最速的路径只剩下了D → E → C。

因为此时D → B → C上的行程时间 = 0.81 + 0.35 = 1.16小时，

而D → E → C上的行程时间 = 0.54 + 0.26 = 0.8小时

这个细节验证了我们的程序正确考虑了拥堵效应，实现了对行程时间函数函数的正确计算，并能够正确回答问题2。

#problem[只考虑一个起迄点对的交通需求，例如A到F,各路段上的流量是多少？有多少被使用的路径？这些路径上的行程时间是否相等？]

为了回答这个问题，我们构造了仅含A到F的OD需求，并分别使用AON、IA（K=3、K=1000）与FW算法进行分配。


#strong[3.1 全有全无（AON）算法分析]

为深入理解不同交通分配模型在单一 OD 对下的行为差异，我们以 A→F 的 2000 辆车需求为例，应用全有全无（All-or-Nothing, AON）并通过可视化结果揭示其路径选择特性与系统均衡状态。

#figure(
  image("../images/仅考虑_A→F_时，全有全无_AON_分配结果.png", width: 90%),
  caption: [
    只考虑A到F的交通需求，使用全有全无 AON 算法的分配结果
  ],
)

在 AON 分配结果中，所有出行者均基于自由流行程时间 $t_0$ 选择最短路径。结果显示：
- 唯一被使用的路径为：A → B → C → E → F；
- 该路径上流量高达 2000 辆，而其他替代路径（如 A → B → E → F）流量为零；
- 各路段行程时间分别为：
  - A→B：1.49 h，
  - B→C：0.40 h，
  - C→E：0.40 h，
  - E→F：1.49 h，
  - 总行程时间：$1.49 + 0.40 + 0.40 + 1.49 = 4.04$ 小时。

然而，此时路网处于*严重不平衡状态*。

例如，若个别出行者选择路径 A → B → E → F，则其行程时间为：
$
  1.49 h + 0.47 h + 1.49 h = 3.45 h
$
显著低于当前路径的 4.04 小时。这表明存在“单方面改变路径可缩短行程”的机会，违反了 Wardrop 第一原理，即用户均衡条件。

#summary[
  全有全无（All-or-Nothing, AON）分配方法基于静态自由流行程时间 $t_0$ 进行路径选择，导致所有出行者集中于单一最短路径。出行者无法根据实时拥堵调整路径选择，从而违反 Wardrop 第一原理。

  尽管AON算法简单易行，但其在模拟现实复杂路网行为方面存在明显不足，主要用于初步评估或作为复杂算法的基础步骤。
]

#strong[3.2 增量分配（Incremental Assignment, IA）算法分析]

为探究增量分配（Incremental Assignment, IA）算法在逼近用户均衡（UE）过程中的行为特征，我们分别设置了两种参数：$K=3$ 和 $K=1000$，并观察其对 A→F 单一 OD 对的分配结果。

#figure(
  image("../images/仅考虑_A→F_时，增量分配_IA_分配结果(K_=_3).png", width: 90%),
  caption: [
    只考虑A到F的交通需求，使用增量分配 IA 的分配结果 ($K=3$)
  ],
)

在 $K=3$ 的情况下：
- 总需求 2000 辆车被分为三批，每批约 666.7 辆；
- 第一批加载时，路网处于自由流状态，所有出行者选择最短路径 A → B → C → E → F；
- 随着后续批次进入，B→C 与 C→E 段因流量上升而阻抗增加，部分出行者开始转向替代路径；
- 分配结果中：
  - 路径1：A → B → C → E → F，流量为 1333.33 辆；
  - 路径2：A → B → E → F，流量为 666.67 辆；
  - 路径3：A → B → D → E → F，未被使用（流量为 0）；

尽管已出现路径分流，但各路径行程时间仍存在明显差异：
- A → B → C → E → F：$1.49 + 0.31 + 0.31 + 1.49 = 4.00$ 小时；
- A → B → E → F：$1.49 + 0.89 + 1.49 = 3.87$ 小时；
- A → B → D → E → F：$1.49 + 0.33 + 0.33 + 1.49 = 4.33$ 小时；

这表明系统*尚未达到平衡状态*，出行者仍有动机从第一条路径转向第二条路径以节省时间。

#figure(
  image("../images/仅考虑_A→F_时，增量分配_IA_分配结果(K_=_1000).png", width: 90%),
  caption: [
    只考虑A到F的交通需求，使用增量分配 IA 的分配结果 ($K=1000$)
  ],
)

当 $K=1000$ 时：
- 需求被细分为 1000 批，每批仅 2 辆车；
- 每次加载前均根据当前累积流量更新路段阻抗，模拟了“连续感知”过程；
- 最终结果中，三条路径均被启用，此时各路径行程时间趋于一致：
  - A → B → C → E → F：$1.49 + 0.35 + 0.35 + 1.49 = 4.04$ 小时；
  - A → B → E → F：$1.49 + 0.69 + 1.49 = 4.03$ 小时；
  - A → B → D → E → F：$1.49 + 0.35 + 0.35 + 1.49 = 4.04$ 小时；

在可接受的计算误差范围内，三条路径的行程时间基本相等，且总出行时间 TTT = 7325.76 veh·h，与后面的 FW 算法结果（7325.86 veh·h）高度接近。

#summary[
  由此可见，增量分配算法通过将总需求分批加载并动态更新阻抗，实现了对用户均衡的逐步逼近。当 $K$ 足够大时，IA 结果能有效捕捉出行者的路径选择行为，使各被使用路径的行程时间趋于相等，从而满足 Wardrop 第一原理。该过程直观展示了“拥堵反馈如何引导路径均衡”，是理解交通分配模型演化的关键案例。
]

#strong[3.3 Frank-Wolfe 算法分析]

Frank-Wolfe 算法通过迭代优化 Beckmann 函数，最终收敛至用户均衡解。

#figure(
  image("../images/仅考虑_A→F_时，Frank-Wolfe_算法分配结果.png", width: 90%),
  caption: [
    只考虑A到F的交通需求，使用FW算法的分配结果
  ],
)

结果表明，A到F之间有三条路径被使用，分别是：
- 路径1：A → B → C → E → F，行程时间为1.49+0.35+0.35+1.49=4.04小时
- 路径2：A → B → E → F，行程时间为1.49+0.69+1.49=4.03小时
- 路径3：A → B → D → E → F，行程时间为1.49+0.35+0.35+1.49=4.04小时
- 在可接受的计算误差范围内，3条路径的行程时间都*相等*
- 此时路网处于*平衡状态*，出行者无法通过单方面改变路径来缩短行程时间，符合Wardrop第一原理！

在 FW 算法下，出行者根据实时拥堵反馈动态调整路径选择，最终达到“任意被使用路径的行程时间相等，且未被使用路径的行程时间不小于该值”的均衡状态。

#summary[
  Frank-Wolfe 算法通过迭代优化 Beckmann 函数，精确逼近用户均衡（User Equilibrium, UE）。

  FW 方法提供了一种从理论上严格保证收敛至全局最优解的途径，是研究和实践交通分配的重要工具，尤其适用于大规模网络中探索出行者的理性行为模式。
]

#problem[考虑所有起迄点对的交通需求，各路段的流量是多少，所有出行者的总出行时间是多少？]

在实际城市路网中，交通需求通常来自多个起讫点对（OD pairs）。为全面评估不同分配算法在复杂网络环境下的表现，我们引入完整的 OD 需求矩阵，并分别应用全有全无（AON）、增量分配（IA）与 Frank-Wolfe（FW）算法进行交通分配。通过对比各算法结果，探究其在系统总出行时间（Total Travel Time, TTT）和流量分布上的差异。

#strong[4.1 全有全无（AON）算法结果分析]

#figure(
  image("../images/考虑所有_OD_对，全有全无_AON_分配结果.png", width: 90%),
  caption: [
    考虑所有OD对，全有全无 AON 分配结果
  ],
)

在 AON 分配下，所有出行者基于自由流行程时间 $t_0$ 选择最短路径，导致路径高度集中：
- *关键瓶颈*：B→C 和 C→E 段流量均达 2000 辆，B→C→E 行程时间为 0.40+0.40 = 0.80 h，高于B→E的 0.47 h；而 B→E、E→B 路径流量为零，说明系统未实现路径分流；
- *系统性能*：总出行时间（TTT）为  23129.63 veh·h 。

由于 AON 忽略拥堵反馈，部分路径仍存在“可节省时间”的机会，例如从 A→B→E→F 可比当前路径更快，因此该解处于非均衡状态。

#strong[4.2 增量分配（IA）算法结果分析]

IA 算法通过将总需求分批加载并动态更新阻抗，逐步逼近用户均衡。

#figure(
  image("../images/考虑所有_OD_对，增量分配_IA_分配结果(K_=_3).png", width: 90%),
  caption: [
    考虑所有OD对，增量分配 IA 分配结果(K=3)
  ],
)

*当 $K=3$ 时：*
- 流量开始向替代路径转移，如 B→E 流量增至 666.67 辆；
- B→C 与 C→E 段流量下降至约 1333 辆，阻抗降低；
- TTT 下降至 22941.26，较 AON 优化约 0.8%。


#figure(
  image("../images/考虑所有_OD_对，增量分配_IA_分配结果(K_=_1000).png", width: 90%),
  caption: [
    考虑所有OD对，增量分配 IA 分配结果(K=1000)
  ],
)

*当 $K=1000$ 时：*
- 所有路径均被启用，包括 B→E、C→B、D→E 等；
- 流量进一步分散，如 B→C 流量降至 1610 辆，C→E 为 1610 辆；
- 各路径行程时间趋于平衡，TTT 进一步降至 22890.43 veh·h。

#strong[4.3 探究分割数 $K$ 对解的收敛性的影响]

为量化 $K$ 对 IA 结果的影响，我们进行了敏感性分析，测试了 $K = 1$ 到 $60$ 的 TTT 变化趋势：

#figure(
  image("../images/IA K-TTT.png", width: 90%),
  caption: [
    探究分割数K对解的收敛性的影响
  ],
)

- 当 $K=1$ 时，TTT 最高（23129.63 veh·h），等价于 AON；
- 随着 $K$ 增加，TTT 快速下降，在 $K=4$ 附近达到最小值（22878.41 veh·h）；
- 当 $K > 10$ 后，TTT 波动趋于平缓；

该图清晰展示了 IA 算法的“收敛特性”：*并非 $K$ 越大越好*，而是在某个临界点后收益递减。这提示在实际应用中，可选取适中的 $K$（如 50–100）以平衡精度与计算成本。

#strong[4.4 Frank-Wolfe 算法结果分析]

FW 算法作为用户均衡的严格求解器，其结果代表了理论最优解：

#figure(
  image("../images/考虑所有_OD_对，Frank-Wolfe_算法分配结果.png", width: 90%),
  caption: [
    考虑所有OD对，Frank-Wolfe 算法分配结果
  ],
)
- *流量分布*：所有可能路径均有合理分配
- *行程时间均衡*：同一OD对的各路径行程时间趋于平衡，符合 Wardrop 第一原理；
- *系统性能*：TTT = 22890.45 veh·h，与 IA（$K=1000$）结果几乎一致。

这验证了 FW 算法在大规模网络中仍能高效收敛至均衡解，且其 TTT 与 IA 的极限值高度吻合，说明两者在理想条件下趋同。

#figure(
  image("../images/FW收敛过程.png", width: 90%),
  caption: [
    FW 收敛过程($epsilon$=$10^(-6)$)
  ],
)

FW 迭代过程，$epsilon$ 为迭代精度，可以看到算法在前两轮相对间隙 convergence 迅速下降迅速，此后稳定下降，直至在第169轮收敛至$epsilon$=$10^(-6)$，验证了算法的高效性。
