# 基于自适应分组策略的PSO-GA混合优化算法

## 项目概述

本项目实现了一种创新的PSO（粒子群优化）和GA（遗传算法）混合策略，通过自适应分组控制和动态权重调整机制，有效解决传统算法容易陷入局部极值的问题。

## 创新点

1. **自适应分组控制策略**：根据适应度值将种群分为优解组和劣解组，分别采用不同的更新策略
2. **动态权重调整**：根据迭代进程和收敛状态动态调整PSO和GA的权重比例
3. **精英保留机制**：结合遗传算法的交叉变异和PSO的速度位置更新
4. **多目标优化支持**：支持单目标和多目标优化问题

## 项目结构

```
├── algorithms/           # 算法实现
│   ├── pso.py           # PSO算法
│   ├── ga.py            # 遗传算法
│   └── hybrid_pso_ga.py # 混合算法
├── benchmark/           # 测试函数
│   └── test_functions.py
├── experiments/         # 实验脚本
│   ├── comparison.py    # 对比实验
│   └── visualization.py # 结果可视化
├── data/               # 实验数据
└── report/             # 报告相关
    └── template.md     # 报告模板
```

## 环境要求

- Python 3.8+
- numpy
- matplotlib
- scipy

## 运行说明

1. 安装依赖：`pip install -r requirements.txt`
2. 运行对比实验：`python experiments/comparison.py`
3. 生成可视化结果：`python experiments/visualization.py`

## 联系方式

如有问题，请联系项目维护者。
