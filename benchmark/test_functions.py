"""
author: 木人舟
brief: 标准测试函数集包含经典的优化问题测试函数，用于验证算法性能
contact:CodingCV@outlook.com
"""

import numpy as np
from typing import Callable, Tuple, List
import math


class TestFunctions:
    """标准测试函数类"""
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        球面函数（Sphere Function）
        全局最优: f(0,...,0) = 0
        搜索范围: [-5.12, 5.12]^n
        特点: 单峰，凸函数，简单
        """
        return np.sum(x**2)
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """
        Rosenbrock函数（香蕉函数）
        全局最优: f(1,...,1) = 0
        搜索范围: [-2.048, 2.048]^n
        特点: 窄valley，收敛困难
        """
        result = 0
        for i in range(len(x) - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return result
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Rastrigin函数
        全局最优: f(0,...,0) = 0
        搜索范围: [-5.12, 5.12]^n
        特点: 多峰，高度多模态
        """
        n = len(x)
        A = 10
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """
        Griewank函数
        全局最优: f(0,...,0) = 0
        搜索范围: [-600, 600]^n
        特点: 多峰，随维数增加难度增加
        """
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """
        Ackley函数
        全局最优: f(0,...,0) = 0
        搜索范围: [-32, 32]^n
        特点: 多峰，指数函数特性
        """
        n = len(x)
        a, b, c = 20, 0.2, 2 * np.pi
        
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        
        return term1 + term2 + a + np.exp(1)
    
    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """
        Schwefel函数
        全局最优: f(420.9687,...,420.9687) ≈ 0
        搜索范围: [-500, 500]^n
        特点: 欺骗性强，全局最优远离局部最优
        """
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def levy(x: np.ndarray) -> float:
        """
        Levy函数
        全局最优: f(1,...,1) = 0
        搜索范围: [-10, 10]^n
        特点: 多峰，具有挑战性
        """
        def w(xi):
            return 1 + (xi - 1) / 4
        
        w_vec = np.array([w(xi) for xi in x])
        
        term1 = np.sin(np.pi * w_vec[0])**2
        
        term2 = 0
        for i in range(len(x) - 1):
            term2 += (w_vec[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w_vec[i] + 1)**2)
        
        term3 = (w_vec[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w_vec[-1])**2)
        
        return term1 + term2 + term3
    
    @staticmethod
    def dixon_price(x: np.ndarray) -> float:
        """
        Dixon & Price函数
        全局最优: f(x*) = 0，其中 x_i = 2^(-(2^i-2)/(2^i))
        搜索范围: [-10, 10]^n
        特点: 单峰，但最优解不在原点
        """
        term1 = (x[0] - 1)**2
        term2 = 0
        for i in range(1, len(x)):
            term2 += (i + 1) * (2 * x[i]**2 - x[i-1])**2
        return term1 + term2
    
    @staticmethod
    def zakharov(x: np.ndarray) -> float:
        """
        Zakharov函数
        全局最优: f(0,...,0) = 0
        搜索范围: [-5, 10]^n
        特点: 单峰，具有噪声特性
        """
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum1 + sum2**2 + sum2**4


class BenchmarkProblem:
    """基准测试问题类"""
    
    def __init__(self, name: str, func: Callable, bounds: List[Tuple[float, float]], 
                 global_optimum: float, optimal_solution: np.ndarray = None):
        """
        初始化基准问题
        
        Args:
            name: 函数名称
            func: 目标函数
            bounds: 搜索边界
            global_optimum: 全局最优值
            optimal_solution: 最优解（可选）
        """
        self.name = name
        self.func = func
        self.bounds = bounds
        self.global_optimum = global_optimum
        self.optimal_solution = optimal_solution
        self.dimension = len(bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """评估函数值"""
        return self.func(x)
    
    def get_random_solution(self) -> np.ndarray:
        """生成随机解"""
        return np.array([
            np.random.uniform(bound[0], bound[1]) 
            for bound in self.bounds
        ])


def get_benchmark_problems(dimension: int = 10) -> List[BenchmarkProblem]:
    """
    获取标准基准测试问题集
    
    Args:
        dimension: 问题维度
        
    Returns:
        基准问题列表
    """
    problems = []
    
    # Sphere函数
    problems.append(BenchmarkProblem(
        name="Sphere",
        func=TestFunctions.sphere,
        bounds=[(-5.12, 5.12)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.zeros(dimension)
    ))
    
    # Rosenbrock函数
    problems.append(BenchmarkProblem(
        name="Rosenbrock",
        func=TestFunctions.rosenbrock,
        bounds=[(-2.048, 2.048)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.ones(dimension)
    ))
    
    # Rastrigin函数
    problems.append(BenchmarkProblem(
        name="Rastrigin",
        func=TestFunctions.rastrigin,
        bounds=[(-5.12, 5.12)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.zeros(dimension)
    ))
    
    # Griewank函数
    problems.append(BenchmarkProblem(
        name="Griewank",
        func=TestFunctions.griewank,
        bounds=[(-600, 600)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.zeros(dimension)
    ))
    
    # Ackley函数
    problems.append(BenchmarkProblem(
        name="Ackley",
        func=TestFunctions.ackley,
        bounds=[(-32, 32)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.zeros(dimension)
    ))
    
    # Schwefel函数
    problems.append(BenchmarkProblem(
        name="Schwefel",
        func=TestFunctions.schwefel,
        bounds=[(-500, 500)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.full(dimension, 420.9687)
    ))
    
    # Levy函数
    problems.append(BenchmarkProblem(
        name="Levy",
        func=TestFunctions.levy,
        bounds=[(-10, 10)] * dimension,
        global_optimum=0.0,
        optimal_solution=np.ones(dimension)
    ))
    
    return problems


def get_simple_problems(dimension: int = 2) -> List[BenchmarkProblem]:
    """获取简单的2D测试问题，用于可视化"""
    problems = []
    
    # 简单的2D问题
    problems.append(BenchmarkProblem(
        name="Sphere_2D",
        func=TestFunctions.sphere,
        bounds=[(-5, 5), (-5, 5)],
        global_optimum=0.0,
        optimal_solution=np.zeros(2)
    ))
    
    problems.append(BenchmarkProblem(
        name="Rastrigin_2D",
        func=TestFunctions.rastrigin,
        bounds=[(-5, 5), (-5, 5)],
        global_optimum=0.0,
        optimal_solution=np.zeros(2)
    ))
    
    problems.append(BenchmarkProblem(
        name="Ackley_2D",
        func=TestFunctions.ackley,
        bounds=[(-5, 5), (-5, 5)],
        global_optimum=0.0,
        optimal_solution=np.zeros(2)
    ))
    
    return problems
