"""
author: 木人舟
brief: 实验结果可视化脚本
生成收敛曲线、性能对比图表等可视化结果
contact:CodingCV@outlook.com
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import sys
from typing import Dict, List

# 设置中文字体
def setup_chinese_fonts():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 初始化字体设置
setup_chinese_fonts()

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import PSO, GA, AdaptiveHybridPSOGA
from benchmark import get_simple_problems


class VisualizationGenerator:
    """可视化生成器类"""
    
    def __init__(self):
        """初始化可视化生成器"""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.results = self.load_results()
        
        # 重新设置字体（确保在类中生效）
        setup_chinese_fonts()
        
        # 设置图表样式
        sns.set_style("whitegrid")
        try:
            plt.style.use('seaborn-v0_8')
        except:
            # 如果seaborn-v0_8不可用，使用默认样式
            sns.set_style("whitegrid")
        
        # 算法颜色映射
        self.colors = {
            'PSO': '#1f77b4',
            'GA': '#ff7f0e', 
            'Hybrid_PSO_GA': '#2ca02c'
        }
        
        # 算法名称映射
        self.alg_names = {
            'PSO': 'PSO',
            'GA': 'GA',
            'Hybrid_PSO_GA': '混合PSO-GA'
        }
    
    def load_results(self) -> Dict:
        """加载实验结果"""
        results_file = os.path.join(self.data_dir, 'experiment_results.json')
        if not os.path.exists(results_file):
            print(f"结果文件不存在: {results_file}")
            print("请先运行 comparison.py 进行实验")
            return {}
        
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    def _ensure_chinese_display(self):
        """确保中文显示正常"""
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为微软雅黑
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    def generate_performance_comparison(self):
        """生成性能对比图表"""
        if not self.results:
            return
        
        # 确保中文显示正常
        self._ensure_chinese_display()
        
        print("生成性能对比图表...")
        
        # 准备数据
        data = []
        for problem_name, problem_results in self.results.items():
            for alg_name, alg_results in problem_results.items():
                stats = alg_results['statistics']
                data.append({
                    '测试函数': problem_name,
                    '算法': self.alg_names[alg_name],
                    '平均适应度': stats['mean_fitness'],
                    '标准差': stats['std_fitness'],
                    '最优适应度': stats['min_fitness'],
                    '成功率': stats['success_rate'],
                    '平均时间': stats['mean_time']
                })
        
        df = pd.DataFrame(data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('算法性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 平均适应度对比（对数尺度）
        pivot_fitness = df.pivot(index='测试函数', columns='算法', values='平均适应度')
        ax1 = axes[0, 0]
        pivot_fitness.plot(kind='bar', ax=ax1, color=[self.colors[k] for k in ['PSO', 'GA', 'Hybrid_PSO_GA']])
        ax1.set_yscale('log')
        ax1.set_title('平均适应度对比 (对数尺度)')
        ax1.set_ylabel('平均适应度')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        
        # 2. 成功率对比
        pivot_success = df.pivot(index='测试函数', columns='算法', values='成功率')
        ax2 = axes[0, 1]
        pivot_success.plot(kind='bar', ax=ax2, color=[self.colors[k] for k in ['PSO', 'GA', 'Hybrid_PSO_GA']])
        ax2.set_title('成功率对比')
        ax2.set_ylabel('成功率')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # 3. 执行时间对比
        pivot_time = df.pivot(index='测试函数', columns='算法', values='平均时间')
        ax3 = axes[1, 0]
        pivot_time.plot(kind='bar', ax=ax3, color=[self.colors[k] for k in ['PSO', 'GA', 'Hybrid_PSO_GA']])
        ax3.set_title('平均执行时间对比')
        ax3.set_ylabel('执行时间 (秒)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # 4. 算法稳定性对比（标准差）
        pivot_std = df.pivot(index='测试函数', columns='算法', values='标准差')
        ax4 = axes[1, 1]
        pivot_std.plot(kind='bar', ax=ax4, color=[self.colors[k] for k in ['PSO', 'GA', 'Hybrid_PSO_GA']])
        ax4.set_yscale('log')
        ax4.set_title('算法稳定性对比 (标准差)')
        ax4.set_ylabel('标准差')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_convergence_curves(self):
        """生成收敛曲线图"""
        # 确保中文显示正常
        self._ensure_chinese_display()
        
        print("生成收敛曲线图...")
        
        # 运行简单的2D问题来获取收敛曲线
        problems = get_simple_problems()
        
        fig, axes = plt.subplots(1, len(problems), figsize=(5*len(problems), 4))
        if len(problems) == 1:
            axes = [axes]
        
        for i, problem in enumerate(problems):
            print(f"  处理 {problem.name}...")
            
            # 运行算法获取收敛曲线
            algorithms = {
                'PSO': PSO(problem.evaluate, problem.dimension, problem.bounds, 
                          n_particles=30, max_iterations=100),
                'GA': GA(problem.evaluate, problem.dimension, problem.bounds,
                        population_size=30, max_generations=100),
                'Hybrid_PSO_GA': AdaptiveHybridPSOGA(problem.evaluate, problem.dimension, problem.bounds,
                                                    population_size=30, max_iterations=100)
            }
            
            ax = axes[i]
            
            for alg_name, algorithm in algorithms.items():
                _, _, convergence_curve = algorithm.optimize()
                ax.plot(convergence_curve, label=self.alg_names[alg_name], 
                       color=self.colors[alg_name], linewidth=2)
            
            ax.set_yscale('log')
            ax.set_title(f'{problem.name} 收敛曲线')
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('最优适应度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'convergence_curves.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_algorithm_ranking(self):
        """生成算法排名图"""
        if not self.results:
            return
        
        # 确保中文显示正常
        self._ensure_chinese_display()
        
        print("生成算法排名图...")
        
        # 计算每个问题上的排名
        rankings = []
        for problem_name, problem_results in self.results.items():
            # 按平均适应度排序
            sorted_results = sorted(
                problem_results.items(),
                key=lambda x: x[1]['statistics']['mean_fitness']
            )
            
            for rank, (alg_name, _) in enumerate(sorted_results, 1):
                rankings.append({
                    '测试函数': problem_name,
                    '算法': self.alg_names[alg_name],
                    '排名': rank
                })
        
        df_rankings = pd.DataFrame(rankings)
        
        # 创建排名热力图
        pivot_rankings = df_rankings.pivot(index='算法', columns='测试函数', values='排名')
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_rankings, annot=True, cmap='RdYlGn_r', center=2,
                   cbar_kws={'label': '排名'}, fmt='d')
        plt.title('算法在各测试函数上的排名', fontsize=14, fontweight='bold')
        plt.xlabel('测试函数')
        plt.ylabel('算法')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'algorithm_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算平均排名
        avg_rankings = df_rankings.groupby('算法')['排名'].mean().sort_values()
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(avg_rankings.index, avg_rankings.values, 
                      color=[self.colors[k] for k in ['PSO', 'GA', 'Hybrid_PSO_GA']])
        plt.title('算法平均排名', fontsize=14, fontweight='bold')
        plt.xlabel('算法')
        plt.ylabel('平均排名')
        plt.ylim(0, 4)
        
        # 添加数值标签
        for bar, value in zip(bars, avg_rankings.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'average_ranking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_statistical_analysis(self):
        """生成统计分析图表"""
        if not self.results:
            return
        
        # 确保中文显示正常
        self._ensure_chinese_display()
        
        print("生成统计分析图表...")
        
        # 准备数据
        all_fitness = []
        for problem_name, problem_results in self.results.items():
            for alg_name, alg_results in problem_results.items():
                fitness_values = alg_results['best_fitness']
                for fitness in fitness_values:
                    all_fitness.append({
                        '测试函数': problem_name,
                        '算法': self.alg_names[alg_name],
                        '适应度': fitness
                    })
        
        df_fitness = pd.DataFrame(all_fitness)
        
        # 创建箱线图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        problems = list(self.results.keys())
        for i, problem in enumerate(problems[:6]):  # 最多显示6个问题
            if i >= len(axes):
                break
                
            problem_data = df_fitness[df_fitness['测试函数'] == problem]
            
            sns.boxplot(data=problem_data, x='算法', y='适应度', ax=axes[i])
            axes[i].set_yscale('log')
            axes[i].set_title(f'{problem} 适应度分布')
            axes[i].tick_params(axis='x', rotation=45)
        
        # 隐藏多余的子图
        for i in range(len(problems), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'statistical_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_2d_optimization_visualization(self):
        """生成2D优化过程可视化"""
        print("生成2D优化过程可视化...")
        
        # 使用简单的2D函数进行可视化
        from benchmark.test_functions import TestFunctions
        
        def rastrigin_2d(x):
            return TestFunctions.rastrigin(x)
        
        # 创建自定义PSO类来记录路径
        class PSO_WithPath(PSO):
            def __init__(self, *args, **kwargs):
                # 先初始化路径列表，避免在父类初始化过程中出现AttributeError
                self.path = []
                super().__init__(*args, **kwargs)
            
            def optimize(self, verbose=False):
                # 清空路径并记录初始位置
                self.path = []
                # 运行优化
                result = super().optimize(verbose)
                return result
            
            def _update_global_best(self):
                super()._update_global_best()
                # 安全地记录路径
                if hasattr(self, 'path') and hasattr(self, 'global_best_position') and self.global_best_position is not None:
                    self.path.append(self.global_best_position.copy())
        
        # 创建网格
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rastrigin_2d(np.array([X[i, j], Y[i, j]]))
        
        # 运行算法并记录路径
        bounds = [(-5, 5), (-5, 5)]
        
        algorithms = {
            'PSO': PSO_WithPath(rastrigin_2d, 2, bounds, n_particles=20, max_iterations=50),
            'GA': GA(rastrigin_2d, 2, bounds, population_size=20, max_generations=50),
            'Hybrid_PSO_GA': AdaptiveHybridPSOGA(rastrigin_2d, 2, bounds, 
                                               population_size=20, max_iterations=50)
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (alg_name, algorithm) in enumerate(algorithms.items()):
            ax = axes[i]
            
            # 绘制等高线
            contour = ax.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
            ax.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
            
            # 运行算法
            best_pos, best_fit, _ = algorithm.optimize()
            
            # 绘制最优解
            ax.plot(best_pos[0], best_pos[1], 'r*', markersize=15, 
                   label=f'最优解 ({best_fit:.4f})')
            
            # 绘制真实最优解
            ax.plot(0, 0, 'w*', markersize=15, label='全局最优')
            
            # 如果是PSO，绘制路径
            if hasattr(algorithm, 'path') and len(algorithm.path) > 1:
                path = np.array(algorithm.path)
                ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.7, label='搜索路径')
            
            ax.set_title(f'{self.alg_names[alg_name]} 优化过程')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, '2d_optimization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")
        print("="*50)
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            # 生成各种图表
            self.generate_performance_comparison()
            self.generate_convergence_curves()
            self.generate_algorithm_ranking()
            self.generate_statistical_analysis()
            self.generate_2d_optimization_visualization()
            
            print("\n所有可视化图表已生成完成！")
            print(f"图表保存在: {self.data_dir}")
            
        except Exception as e:
            print(f"生成可视化时出错: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    generator = VisualizationGenerator()
    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()