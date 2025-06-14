"""
author: 木人舟
brief: 粒子群优化算法（PSO）实现
contact:CodingCV@outlook.com
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import copy


class Particle:
    """粒子类，表示PSO算法中的单个粒子"""
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        """
        初始化粒子
        
        Args:
            position: 粒子位置
            velocity: 粒子速度
        """
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.best_position = position.copy()  # 个体最优位置
        self.best_fitness = float('inf')      # 个体最优适应度
        self.fitness = float('inf')           # 当前适应度
    
    def update_best(self):
        """更新个体最优位置和适应度"""
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()


class PSO:
    """粒子群优化算法类"""
    
    def __init__(self, 
                 fitness_func: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: List[Tuple[float, float]],
                 n_particles: int = 30,
                 max_iterations: int = 100,
                 w: float = 0.9,           # 惯性权重
                 c1: float = 2.0,          # 个体学习因子
                 c2: float = 2.0,          # 社会学习因子
                 w_decay: bool = True):    # 是否使用权重衰减
        """
        初始化PSO算法
        
        Args:
            fitness_func: 适应度函数
            dimension: 搜索空间维度
            bounds: 每个维度的边界 [(min1, max1), (min2, max2), ...]
            n_particles: 粒子数量
            max_iterations: 最大迭代次数
            w: 惯性权重
            c1: 个体学习因子
            c2: 社会学习因子
            w_decay: 是否使用权重衰减
        """
        self.fitness_func = fitness_func
        self.dimension = dimension
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.w_min = 0.1              # 最小惯性权重
        self.w_max = w                # 最大惯性权重
        self.c1 = c1
        self.c2 = c2
        self.w_decay = w_decay
        
        # 算法状态
        self.particles: List[Particle] = []
        self.global_best_position = np.zeros(dimension)
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        self.current_iteration = 0
        
        # 初始化粒子群
        self._initialize_particles()
    
    def _initialize_particles(self):
        """初始化粒子群"""
        self.particles = []
        
        for _ in range(self.n_particles):
            # 随机初始化位置和速度
            position = np.array([
                np.random.uniform(bound[0], bound[1]) 
                for bound in self.bounds
            ])
            
            # 速度初始化为位置范围的一定比例
            velocity = np.array([
                np.random.uniform(-abs(bound[1] - bound[0]) * 0.1, 
                                 abs(bound[1] - bound[0]) * 0.1)
                for bound in self.bounds
            ])
            
            particle = Particle(position, velocity)
            self.particles.append(particle)
        
        # 评估初始适应度
        self._evaluate_fitness()
        self._update_global_best()
    
    def _evaluate_fitness(self):
        """评估所有粒子的适应度"""
        for particle in self.particles:
            particle.fitness = self.fitness_func(particle.position)
            particle.update_best()
    
    def _update_global_best(self):
        """更新全局最优位置"""
        for particle in self.particles:
            if particle.best_fitness < self.global_best_fitness:
                self.global_best_fitness = particle.best_fitness
                self.global_best_position = particle.best_position.copy()
    
    def _update_particles(self):
        """更新粒子位置和速度"""
        # 动态调整惯性权重
        if self.w_decay:
            current_w = self.w_max - (self.w_max - self.w_min) * (
                self.current_iteration / self.max_iterations
            )
        else:
            current_w = self.w
        
        for particle in self.particles:
            # 生成随机数
            r1 = np.random.random(self.dimension)
            r2 = np.random.random(self.dimension)
            
            # 更新速度
            cognitive_component = self.c1 * r1 * (
                particle.best_position - particle.position
            )
            social_component = self.c2 * r2 * (
                self.global_best_position - particle.position
            )
            
            particle.velocity = (current_w * particle.velocity + 
                               cognitive_component + social_component)
            
            # 限制速度
            for i in range(self.dimension):
                max_velocity = abs(self.bounds[i][1] - self.bounds[i][0]) * 0.2
                particle.velocity[i] = np.clip(particle.velocity[i], 
                                             -max_velocity, max_velocity)
            
            # 更新位置
            particle.position += particle.velocity
            
            # 边界约束
            for i in range(self.dimension):
                particle.position[i] = np.clip(particle.position[i],
                                             self.bounds[i][0], 
                                             self.bounds[i][1])
    
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行PSO优化
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            最优位置, 最优适应度, 适应度历史
        """
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            
            # 更新粒子
            self._update_particles()
            
            # 评估适应度
            self._evaluate_fitness()
            
            # 更新全局最优
            self._update_global_best()
            
            # 记录历史
            self.fitness_history.append(self.global_best_fitness)
            
            if verbose and iteration % 10 == 0:
                print(f"PSO迭代 {iteration}: 最优适应度 = {self.global_best_fitness:.6f}")
        
        return self.global_best_position, self.global_best_fitness, self.fitness_history
    
    def get_particles_info(self) -> List[dict]:
        """获取所有粒子的信息，用于混合算法"""
        particles_info = []
        for i, particle in enumerate(self.particles):
            particles_info.append({
                'index': i,
                'position': particle.position.copy(),
                'velocity': particle.velocity.copy(),
                'fitness': particle.fitness,
                'best_position': particle.best_position.copy(),
                'best_fitness': particle.best_fitness
            })
        return particles_info
    
    def set_particles_from_info(self, particles_info: List[dict]):
        """从信息设置粒子状态，用于混合算法"""
        for info in particles_info:
            idx = info['index']
            if idx < len(self.particles):
                self.particles[idx].position = info['position'].copy()
                self.particles[idx].velocity = info['velocity'].copy()
                self.particles[idx].fitness = info['fitness']
                self.particles[idx].best_position = info['best_position'].copy()
                self.particles[idx].best_fitness = info['best_fitness']
