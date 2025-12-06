# UESTC-Traffic-Assignment-2025

## 1.项目介绍

本项目是电子科技大学2025年交通规划原理的课程项目，聚焦于编程实现三类典型交通分配算法——全有全无（AON）、增量分配（Incremental Assignment, IA）与基于Frank-Wolfe算法的用户均衡分配——并在标准测试路网上进行对比分析

## 2.使用方法

- 输入下面命令完成环境配置
```bash
conda create -n traffic_assignment python=3.8
conda activate traffic_assignment
pip install -r requirements.txt
```
将准备好的交通道路网络文件 netowrk.json 与交通需求文件 demand.json 放置于data目录下。
- 输入下面的命令，分别执行全有全无分配（AON），增量分配（IA）以及基于Frank-Wolfe算法的用户均衡分配
```bash
python AON.py 
python IA.py
python FW.py
```
- 也可直接运行进行测试
```bash
python main.py
```