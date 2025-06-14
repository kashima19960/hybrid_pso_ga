"""
author: 木人舟
brief: 算法对比实验脚本,比较PSO、GA和混合算法在标准测试函数上的性能
contact:CodingCV@outlook.com
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import os
from typing import Dict, List, Tuple
import sys

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import PSO, GA, AdaptiveHybridPSOGA
from benchmark import get_benchmark_problems, get_simple_problems


class ExperimentRunner:
    """实验运行器类"""
    
    def __init__(self, 
                 dimension: int = 10,
                 n_runs: int = 30,
                 max_iterations: int = 200,
                 population_size: int = 50):
        """
        初始化实验参数
        
        Args:
            dimension: 问题维度
            n_runs: 独立运行次数
            max_iterations: 最大迭代次数
            population_size: 种群大小
        """
        self.dimension = dimension
        self.n_runs = n_runs
        self.max_iterations = max_iterations
        self.population_size = population_size
        
        # 获取测试问题
        self.problems = get_benchmark_problems(dimension)
        
        # 结果存储
        self.results = {}
        
        # 创建数据目录
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def run_algorithm(self, algorithm_class, algorithm_name: str, problem, **kwargs) -> Dict:
        """
        运行单个算法
        
        Args:
            algorithm_class: 算法类
            algorithm_name: 算法名称
            problem: 测试问题
            **kwargs: 算法参数
            
        Returns:
            实验结果字典
        """
        print(f"运行 {algorithm_name} 在 {problem.name} 上...")
        
        results = {
            'best_fitness': [],
            'final_solutions': [],
            'convergence_curves': [],
            'execution_times': [],
            'statistics': {}
        }
        
        for run in range(self.n_runs):
            print(f"  运行 {run + 1}/{self.n_runs}", end='\r')
            
            # 创建算法实例
            if algorithm_name == 'PSO':
                algorithm = algorithm_class(
                    fitness_func=problem.evaluate,
                    dimension=problem.dimension,
                    bounds=problem.bounds,
                    n_particles=self.population_size,
                    max_iterations=self.max_iterations,
                    **kwargs
                )
            elif algorithm_name == 'GA':
                algorithm = algorithm_class(
                    fitness_func=problem.evaluate,
                    dimension=problem.dimension,
                    bounds=problem.bounds,
                    population_size=self.population_size,
                    max_generations=self.max_iterations,
                    **kwargs
                )
            else:  # 混合算法
                algorithm = algorithm_class(
                    fitness_func=problem.evaluate,
                    dimension=problem.dimension,
                    bounds=problem.bounds,
                    population_size=self.population_size,
                    max_iterations=self.max_iterations,
                    **kwargs
                )
            
            # 运行算法
            start_time = time.time()
            best_solution, best_fitness, convergence_curve = algorithm.optimize()
            end_time = time.time()
            
            # 记录结果
            results['best_fitness'].append(best_fitness)
            results['final_solutions'].append(best_solution)
            results['convergence_curves'].append(convergence_curve)
            results['execution_times'].append(end_time - start_time)
        
        print()  # 换行
        
        # 计算统计信息
        best_fitness_array = np.array(results['best_fitness'])
        results['statistics'] = {
            'mean_fitness': np.mean(best_fitness_array),
            'std_fitness': np.std(best_fitness_array),
            'min_fitness': np.min(best_fitness_array),
            'max_fitness': np.max(best_fitness_array),
            'median_fitness': np.median(best_fitness_array),
            'success_rate': np.sum(best_fitness_array < (problem.global_optimum + 1e-6)) / self.n_runs,
            'mean_time': np.mean(results['execution_times']),
            'std_time': np.std(results['execution_times'])
        }
        
        return results
    
    def run_comparison_experiment(self):
        """运行完整的对比实验"""
        print("开始运行算法对比实验...")
        print(f"测试问题数量: {len(self.problems)}")
        print(f"问题维度: {self.dimension}")
        print(f"独立运行次数: {self.n_runs}")
        print(f"最大迭代次数: {self.max_iterations}")
        print("="*50)
        
        # 算法配置
        algorithms = {
            'PSO': {
                'class': PSO,
                'params': {
                    'w': 0.9,
                    'c1': 2.0,
                    'c2': 2.0,
                    'w_decay': True
                }
            },
            'GA': {
                'class': GA,
                'params': {
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.1,
                    'elite_size': 2,
                    'selection_method': 'tournament'
                }
            },
            'Hybrid_PSO_GA': {
                'class': AdaptiveHybridPSOGA,
                'params': {
                    'w': 0.9,
                    'c1': 2.0,
                    'c2': 2.0,
                    'crossover_rate': 0.8,
                    'mutation_rate': 0.1,
                    'group_ratio': 0.6,
                    'exchange_interval': 10
                }
            }
        }
        
        # 运行实验
        for problem in self.problems:
            print(f"\n测试问题: {problem.name}")
            print("-" * 30)
            
            self.results[problem.name] = {}
            
            for alg_name, alg_config in algorithms.items():
                result = self.run_algorithm(
                    alg_config['class'],
                    alg_name,
                    problem,
                    **alg_config['params']
                )
                self.results[problem.name][alg_name] = result
        
        # 保存结果
        self.save_results()
        
        # 生成报告
        self.generate_report()
    
    def save_results(self):
        """保存实验结果"""
        # 保存完整结果（不包含收敛曲线，太大）
        simplified_results = {}
        for problem_name, problem_results in self.results.items():
            simplified_results[problem_name] = {}
            for alg_name, alg_results in problem_results.items():
                simplified_results[problem_name][alg_name] = {
                    'best_fitness': alg_results['best_fitness'],
                    'execution_times': alg_results['execution_times'],
                    'statistics': alg_results['statistics']
                }
        
        # 保存为JSON
        results_file = os.path.join(self.data_dir, 'experiment_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n实验结果已保存到: {results_file}")
    
    def generate_report(self):
        """生成实验报告"""
        print("\n生成实验报告...")
        
        # 创建统计表格
        report_data = []
        for problem_name, problem_results in self.results.items():
            for alg_name, alg_results in problem_results.items():
                stats = alg_results['statistics']
                report_data.append({
                    '问题': problem_name,
                    '算法': alg_name,
                    '平均适应度': f"{stats['mean_fitness']:.6e}",
                    '标准差': f"{stats['std_fitness']:.6e}",
                    '最优适应度': f"{stats['min_fitness']:.6e}",
                    '成功率': f"{stats['success_rate']:.2%}",
                    '平均时间(s)': f"{stats['mean_time']:.3f}"
                })
        
        # 创建DataFrame
        df = pd.DataFrame(report_data)
        
        # 保存CSV报告
        report_file = os.path.join(self.data_dir, 'experiment_report.csv')
        df.to_csv(report_file, index=False, encoding='utf-8-sig')
        
        # 打印摘要
        print("\n实验结果摘要:")
        print("="*80)
        for problem_name in self.results.keys():
            print(f"\n{problem_name}:")
            problem_df = df[df['问题'] == problem_name]
            print(problem_df[['算法', '平均适应度', '成功率', '平均时间(s)']].to_string(index=False))
        
        print(f"\n详细报告已保存到: {report_file}")
    
    def calculate_algorithm_rankings(self) -> pd.DataFrame:
        """计算算法排名"""
        rankings = []
        
        for problem_name, problem_results in self.results.items():
            # 按平均适应度排名
            fitness_ranking = sorted(
                problem_results.items(),
                key=lambda x: x[1]['statistics']['mean_fitness']
            )
            
            for rank, (alg_name, _) in enumerate(fitness_ranking, 1):
                rankings.append({
                    '问题': problem_name,
                    '算法': alg_name,
                    '适应度排名': rank
                })
        
        df = pd.DataFrame(rankings)
        
        # 计算平均排名
        avg_rankings = df.groupby('算法')['适应度排名'].mean().sort_values()
        
        print("\n算法平均排名:")
        print("-" * 20)
        for alg, rank in avg_rankings.items():
            print(f"{alg}: {rank:.2f}")
        
        return df


def main():
    """主函数"""
    # 创建实验运行器
    runner = ExperimentRunner(
        dimension=10,
        n_runs=30,
        max_iterations=200,
        population_size=50
    )
    
    # 运行对比实验
    runner.run_comparison_experiment()
    
    # 计算排名
    runner.calculate_algorithm_rankings()
    
    print("\n实验完成！")
    print("可以运行 visualization.py 来生成可视化结果")


if __name__ == "__main__":
    main()
