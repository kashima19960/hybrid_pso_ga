# 基于自适应分组策略的PSO-GA混合优化算法

## 项目概述

本项目实现了一种创新的PSO（粒子群优化）和GA（遗传算法）混合策略，通过自适应分组控制和动态权重调整机制，有效解决传统算法容易陷入局部极值的问题。该算法在多个标准测试函数上表现出色，收敛精度和稳定性显著优于传统PSO和GA算法。

## 算法特点

1. **自适应分组控制策略**
   - 根据适应度排名动态将种群分为优解组和劣解组
   - 优解组采用PSO进行快速局部搜索
   - 劣解组采用GA进行全局探索
   - 分组比例根据种群多样性自适应调整

2. **三重动态权重调整**
   - 多样性因子：根据种群聚集度调整
   - 收敛速度因子：监控适应度改善速率
   - 进程因子：考虑优化阶段特征
   - 综合三因子实现精细化权重控制

3. **精英迁移机制**
   - PSO→GA：全局最优位置指导GA变异
   - GA→PSO：优秀个体替换PSO中的劣质粒子
   - 双向信息交换促进算法协同进化

4. **自适应重新分组**
   - 基于性能反馈的重新分组触发机制
   - 防止算法陷入停滞状态
   - 维持种群多样性

## 项目结构

```
hybrid_pso_ga/
├── algorithms/              # 核心算法实现
│   ├── __init__.py         # 模块初始化
│   ├── pso.py              # 粒子群优化算法
│   ├── ga.py               # 遗传算法
│   └── hybrid_pso_ga.py    # 自适应混合PSO-GA算法
├── benchmark/              # 测试函数集
│   ├── __init__.py         # 模块初始化  
│   └── test_functions.py   # 标准测试函数（Sphere、Rastrigin等）
├── experiments/            # 实验与分析
│   ├── __init__.py         # 模块初始化
│   ├── comparison.py       # 算法对比实验
│   └── visualization.py    # 结果可视化生成
├── data/                   # 实验数据与结果
│   ├── experiment_results.json      # 实验结果数据
│   ├── experiment_report.csv        # 统计报告
│   ├── performance_comparison.png   # 性能对比图
│   ├── convergence_curves.png       # 收敛曲线图
│   ├── algorithm_ranking.png        # 算法排名热力图
│   ├── average_ranking.png          # 平均排名图
│   ├── statistical_analysis.png     # 统计分析图
│   └── 2d_optimization.png          # 2D优化轨迹图
├── requirements.txt        # Python依赖包
├── main.py                # 主程序入口
├── README.md              # 项目说明文档
└── .gitignore             # Git忽略文件配置
```

## 测试函数集

本项目包含7个经典优化测试函数，涵盖不同的优化挑战：

| 函数名 | 类型 | 特点 | 维度 | 搜索域 |
|--------|------|------|------|--------|
| Sphere | 单峰 | 简单凸函数 | 2D | [-5.12, 5.12] |
| Rosenbrock | 窄谷 | 收敛困难 | 2D | [-2.048, 2.048] |
| Rastrigin | 多峰 | 高度多模态 | 2D | [-5.12, 5.12] |
| Griewank | 多峰 | 多局部极值 | 2D | [-600, 600] |
| Ackley | 指数 | 指数特性 | 2D | [-32, 32] |
| Schwefel | 欺骗性 | 远距离全局最优 | 2D | [-500, 500] |
| Levy | 多峰 | 复杂地形 | 2D | [-10, 10] |

## 环境要求
- Python 3.8 或更高版本

## 安装与运行

### 1. 环境准备
```bash
# 克隆项目（如果使用Git）
git clone https://github.com/kashima19960/hybrid_pso_ga
cd hybrid_pso_ga

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速开始
```bash
# 运行完整实验流程
python main.py

# 或分步执行：

# 1. 运行算法对比实验
python experiments/comparison.py

# 2. 生成可视化结果
python experiments/visualization.py
```


## 实验结果
参照`data/`目录

## 常见问题

1. **字体显示问题**
   ```python
   # 如果中文显示为方块，确保系统安装了微软雅黑字体
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
   ```

2. **内存不足**
   ```python
   # 减少种群大小或迭代次数
   optimizer = AdaptiveHybridPSOGA(population_size=30, max_iterations=100)
   ```

3. **收敛过慢**
   ```python
   # 调整参数设置
   optimizer = AdaptiveHybridPSOGA(
       alpha=0.7,              # 增加优解组比例
       exchange_interval=5     # 减少交换间隔
   )
   ```