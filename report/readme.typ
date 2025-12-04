#import "./lib.typ": *
#show: project.with(
  title: "《交通规划原理》课程报告",
  author: "倪子煊2023091203002",
  date: auto,
  // abstract: [
  //   摘要内容...
  // ],
  // keywords: ("关键词1", "关键词2")
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
全有全无分配（AON）：基于自由流阻抗 $t_0$ 执行一次AON分配，忽略流量对阻抗的反馈；
- 增量分配（IA）：将总OD需求等分为 $K$ 份，逐次执行AON分配并累加流量，每次分配前更新路段阻抗，逐步逼近均衡状态；
- Frank-Wolfe用户均衡分配（FW）：以AON结果为初始解，迭代执行“阻抗更新 → AON方向搜索 → 步长优化 → 解更新 → 收敛检验”五步流程，直至满足预设精度 $epsilon$，理论上可收敛至Wardrop用户均衡解。

#strong[（5）可视化模块（`visualize_network.py`）]

该模块将抽象的分配结果转化为直观的空间图形：
- 接收NetworkX有向图 $G$（边属性含流量 $Q$ 与行程时间 $T$）、节点坐标字典 `pos_dict` 及总出行时间 `TTT`；
- 采用对数缩放策略映射流量至线宽，Viridis色谱映射流量至颜色，增强视觉辨识度；
- 在每条边上叠加三行标签（路段标识、流量 $q$、行程时间 $t$），并通过垂直偏移避免双向边标签重叠；
- 绘制带色标的路网图，并在图中嵌入TTT数值框，实现结果的一站式呈现。

#strong[（6）主程序（`main.py`）]

作为系统入口，该模块整合前述功能，系统性验证课程要求的各项能力：
- 分别在自由流与拥堵状态下查询任意OD对的最短路径；
- 针对单一OD对（如A→F）执行FW均衡分配，分析多路径使用情况；
- 对全网OD需求，对比AON、IA与FW三种算法的路段流量分布与TTT指标；
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

可通过以下命令快速部署并运行本软件：
```bash
git clone https://github.com/nzxQAQ/uestc-traffic-assignment-2025.git
cd uestc-traffic-assignment-2025
pip install -r requirements.txt
python main.py
```

= 关键算法代码片段
为确保算法实现的透明性与可复现性，本节选取若干核心函数进行展示与说明。所有代码均采用Python语言编写，注重可读性与模块化设计。但受限于篇幅，部分函数会以伪代码或者省略号替代，完整代码请访问GitHub仓库。


#strong[1. 最短路径搜索与全有全无分配（`assignment_utils.py`）]

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

`all_or_nothing_assignment` 函数则遍历所有OD对，调用上述最短路径函数，完成一次完整的AON分配：

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

#strong[2. 路阻函数t、总出行时间TTT 以及 Beckmann函数（`calculate.py`）]

本项目采用老师指定的路阻函数，其形式为 $ t = t_0 (1 + q/C)^2 $

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
#strong[3. 最优步长求解：Newton-Raphson 精确线搜索（`calculate.py`）]

在 Frank-Wolfe 算法中，每一步迭代需确定最优步长 $lambda in [0,1]$，以最小化 Beckmann 函数 $Z(q)$ 沿当前搜索方向（即从当前解 $x$ 指向 AON 解 $y$ 的方向）的值。

教材中常采用#strong[二分法]（Bisection Method）求解该一维优化问题，而本项目则采用#strong[牛顿-拉弗森法]（Newton-Raphson Method）实现高精度线搜索，后者在本场景下具有显著优势，主要体现在以下三方面：

#strong[收敛速度：牛顿法 vs. 二分法]

- 牛顿法：利用目标函数的一阶导数 $phi'(lambda)$ 与二阶导数 $phi''(lambda)$ 构造局部二次近似，在极小值点附近具有#emph[二次收敛性]（quadratic convergence）。这意味着误差平方级下降——例如，若当前误差为 $10^(-2)$，下一步可能降至 $10^(-4)$。
- 二分法：仅依赖一阶导数的符号变化进行区间缩放，收敛速度为#emph[线性]（linear），每次迭代仅将误差减半。要达到 $10^{-6}$ 精度，通常需约 20 次迭代,而牛顿法往往只需 3-5 次迭代即可达到同等精度。

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

#strong[自由流行程时间 $t_0$ 的推导]

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

#strong[双向有向边建模]

现实道路通常支持双向通行，因此每条无向路段被显式拆分为两条方向相反的有向边：
```python
links.append({
    'from': u,
    'to': v,
    'length': length,
    'capacity': capacity,
    'speedmax': speedmax,
    't0': t0
})

links.append({
    'from': v,
    'to': u,
    'length': length,
    'capacity': capacity,
    'speedmax': speedmax,
    't0': t0
})

```

#strong[邻接表设计]

为兼顾算法效率与结果可解释性，用邻接表来构建图`graph`：

```python
# 构建邻接表
graph = defaultdict(list)
for idx, link in enumerate(links):
    graph[link['from']].append((link['to'], idx))
```


