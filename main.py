"""
author: 木人舟
brief: 基于自适应分组策略的PSO-GA混合优化算法
主运行脚本
"""
import os
import sys
import time

def print_banner():
    print("-"*60)
    print(" 基于自适应分组策略的PSO-GA混合优化算法")
    print("-"*60)
    print(" 算法特点:")
    print(" ✓ 自适应分组控制策略")
    print(" ✓ 动态权重调整机制") 
    print(" ✓ 精英迁移与信息交换")
    print(" ✓ 多目标性能评估")
    print(" ✓ 完整的可视化分析")
    print("-"*60)

def check_dependencies():
    """检查依赖包"""
    print("\n检查依赖包...")
    required_packages = ['numpy', 'matplotlib', 'pandas', 'seaborn', 'scipy']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ 所有依赖包已安装")
    return True

def run_experiments():
    """运行对比实验"""
    print("\n" + "-"*50)
    print("运行算法对比实验")
    print("-"*50)
    print("PSO、GA和混合算法在7个标准测试函数上的对比实验")
    
    try:
        # 运行对比实验
        from experiments.comparison import main as run_comparison
        print("开始运行对比实验...")
        start_time = time.time()
        run_comparison()
        end_time = time.time()
        print(f"对比实验耗时: {end_time - start_time:.1f} 秒")
        return True
    except Exception as e:
        print(f"出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_visualization():
    """运行可视化"""
    print("\n" + "="*50)
    print("生成可视化结果")
    print("="*50)
    print("生成以下可视化图表:")
    print("- 性能对比图表")
    print("- 收敛曲线分析")
    print("- 算法排名热力图")
    print("- 统计分析箱线图")
    print("- 2D优化过程可视化")
    
    try:
        # 运行可视化
        from experiments.visualization import main as run_visualization_func
        print("开始生成可视化...")
        start_time = time.time()
        run_visualization_func()
        end_time = time.time()
        print(f"可视化耗时: {end_time - start_time:.1f} 秒")
        return True
    except Exception as e:
        print(f"生成可视化时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_demo():
    """运行算法演示"""
    print("\n" + "="*50)
    print("算法演示")
    print("="*50)
    print("运行演示，展示混合算法的优化过程")
    
    try:
        from algorithms import PSO, GA, AdaptiveHybridPSOGA
        from benchmark import get_simple_problems
        import numpy as np
        
        # 选择一个简单的2D问题进行演示
        problems = get_simple_problems()
        problem = problems[1]  # Rastrigin 2D
        
        print(f"\n演示问题: {problem.name}")
        print(f"维度: {problem.dimension}")
        print(f"搜索范围: {problem.bounds}")
        print(f"全局最优: {problem.global_optimum}")
        
        algorithms = {
            'PSO': PSO(problem.evaluate, problem.dimension, problem.bounds, 
                      n_particles=20, max_iterations=50),
            'GA': GA(problem.evaluate, problem.dimension, problem.bounds,
                    population_size=20, max_generations=50),
            'Hybrid PSO-GA': AdaptiveHybridPSOGA(problem.evaluate, problem.dimension, problem.bounds,
                                                population_size=20, max_iterations=50)
        }
        
        print("\n运行算法...")
        results = {}
        for name, algorithm in algorithms.items():
            print(f"  {name}...", end="")
            start_time = time.time()
            best_pos, best_fit, convergence = algorithm.optimize()
            end_time = time.time()
            results[name] = {
                'best_position': best_pos,
                'best_fitness': best_fit,
                'time': end_time - start_time,
                'convergence': convergence
            }
            print(f" 完成 (耗时: {end_time - start_time:.3f}s)")
        
        print("\n结果对比:")
        print("-" * 60)
        print(f"{'算法':<15} {'最优适应度':<15} {'最优位置':<25} {'耗时(s)':<10}")
        print("-" * 60)
        for name, result in results.items():
            pos_str = f"[{result['best_position'][0]:.3f}, {result['best_position'][1]:.3f}]"
            print(f"{name:<15} {result['best_fitness']:<15.6e} {pos_str:<25} {result['time']:<10.3f}")
        
        # 找出最优算法
        best_algorithm = min(results.items(), key=lambda x: x[1]['best_fitness'])
        print(f"\n最优算法: {best_algorithm[0]} (适应度: {best_algorithm[1]['best_fitness']:.6e})")
        
    except Exception as e:
        print(f"运行演示时出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("程序退出")
        return
    
    print("\n开始")
    
    # 运行实验
    exp_success = run_experiments()
    
    # 生成可视化
    if exp_success:
        run_visualization()
    
    # 运行演示
    run_demo()
    
    print("\n结束！")
    print("结果已保存到 data/ 目录")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
