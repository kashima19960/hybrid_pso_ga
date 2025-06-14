"""
author: 木人舟
brief: 自适应分组PSO-GA混合优化算法
contact:CodingCV@outlook.com
创新点：
1. 自适应分组控制策略：根据适应度将种群分为优解组和劣解组
2. 动态权重调整：根据收敛状态动态调整PSO和GA的权重
3. 精英迁移机制：在两个算法间进行信息交换
4. 多阶段优化：前期重视全局探索，后期重视局部开发
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import copy
from .pso import PSO, Particle
from .ga import GA, Individual


class AdaptiveHybridPSOGA:
    """自适应分组PSO-GA混合优化算法"""
    
    def __init__(self,
                 fitness_func: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 50,
                 max_iterations: int = 200,
                 # PSO参数
                 w: float = 0.9,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 # GA参数
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 # 混合算法参数
                 group_ratio: float = 0.6,        # 优解组比例
                 exchange_interval: int = 10,     # 信息交换间隔
                 convergence_threshold: float = 1e-6,  # 收敛阈值
                 diversity_threshold: float = 0.1):    # 多样性阈值
        """
        初始化混合算法
        
        Args:
            fitness_func: 适应度函数
            dimension: 搜索空间维度
            bounds: 搜索边界
            population_size: 总种群大小
            max_iterations: 最大迭代次数
            w, c1, c2: PSO参数
            crossover_rate, mutation_rate: GA参数
            group_ratio: 优解组占总种群的比例
            exchange_interval: 算法间信息交换的间隔
            convergence_threshold: 判断收敛的阈值
            diversity_threshold: 判断多样性的阈值
        """
        self.fitness_func = fitness_func
        self.dimension = dimension
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.group_ratio = group_ratio
        self.exchange_interval = exchange_interval
        self.convergence_threshold = convergence_threshold
        self.diversity_threshold = diversity_threshold
        
        # 计算子种群大小
        self.elite_size = int(population_size * group_ratio)
        self.regular_size = population_size - self.elite_size
        
        # 初始化PSO和GA
        self.pso = PSO(
            fitness_func=fitness_func,
            dimension=dimension,
            bounds=bounds,
            n_particles=self.elite_size,
            max_iterations=max_iterations,
            w=w, c1=c1, c2=c2
        )
        
        self.ga = GA(
            fitness_func=fitness_func,
            dimension=dimension,
            bounds=bounds,
            population_size=self.regular_size,
            max_generations=max_iterations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )
        
        # 算法状态
        self.current_iteration = 0
        self.global_best_position = np.zeros(dimension)
        self.global_best_fitness = float('inf')
        self.fitness_history = []
        self.convergence_history = []
        self.diversity_history = []
        
        # 权重调整参数
        self.pso_weight = 0.5
        self.ga_weight = 0.5
        self.weight_history = []
        
        # 初始化统一种群并分组
        self._initialize_unified_population()
    
    def _initialize_unified_population(self):
        """初始化统一种群并进行初始分组"""
        # 创建统一的初始种群
        unified_population = []
        for _ in range(self.population_size):
            position = np.array([
                np.random.uniform(bound[0], bound[1])
                for bound in self.bounds
            ])
            fitness = self.fitness_func(position)
            unified_population.append({
                'position': position,
                'fitness': fitness
            })
        
        # 按适应度排序
        unified_population.sort(key=lambda x: x['fitness'])
        
        # 分配给PSO（优解组）
        pso_particles = []
        for i in range(self.elite_size):
            individual = unified_population[i]
            velocity = np.array([
                np.random.uniform(-abs(bound[1] - bound[0]) * 0.1,
                                 abs(bound[1] - bound[0]) * 0.1)
                for bound in self.bounds
            ])
            particle = Particle(individual['position'], velocity)
            particle.fitness = individual['fitness']
            particle.update_best()
            pso_particles.append(particle)
        
        self.pso.particles = pso_particles
        self.pso._update_global_best()
        
        # 分配给GA（劣解组）
        ga_individuals = []
        for i in range(self.elite_size, self.population_size):
            individual_data = unified_population[i]
            individual = Individual(individual_data['position'])
            individual.fitness = individual_data['fitness']
            ga_individuals.append(individual)
        
        self.ga.population = ga_individuals
        self.ga._update_best()
        
        # 更新全局最优
        self._update_global_best()
    
    def _update_global_best(self):
        """更新全局最优解"""
        # 比较PSO和GA的最优解
        pso_best_fitness = self.pso.global_best_fitness
        ga_best_fitness = self.ga.best_individual.fitness
        
        if pso_best_fitness < ga_best_fitness:
            if pso_best_fitness < self.global_best_fitness:
                self.global_best_fitness = pso_best_fitness
                self.global_best_position = self.pso.global_best_position.copy()
        else:
            if ga_best_fitness < self.global_best_fitness:
                self.global_best_fitness = ga_best_fitness
                self.global_best_position = self.ga.best_individual.genes.copy()
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        all_positions = []
        
        # 收集所有位置
        for particle in self.pso.particles:
            all_positions.append(particle.position)
        for individual in self.ga.population:
            all_positions.append(individual.genes)
        
        if len(all_positions) < 2:
            return 1.0
        
        # 计算平均距离作为多样性指标
        total_distance = 0
        count = 0
        for i in range(len(all_positions)):
            for j in range(i + 1, len(all_positions)):
                distance = np.linalg.norm(all_positions[i] - all_positions[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def _calculate_convergence_rate(self) -> float:
        """计算收敛速度"""
        if len(self.fitness_history) < 10:
            return 0
        
        recent_fitness = self.fitness_history[-10:]
        improvement = recent_fitness[0] - recent_fitness[-1]
        return improvement / recent_fitness[0] if recent_fitness[0] != 0 else 0
    
    def _adaptive_weight_adjustment(self):
        """自适应权重调整"""
        diversity = self._calculate_diversity()
        convergence_rate = self._calculate_convergence_rate()
        progress = self.current_iteration / self.max_iterations
        
        # 基于多样性调整权重
        if diversity < self.diversity_threshold:
            # 多样性低，增加GA权重（增强全局搜索）
            diversity_factor = 0.3
        else:
            # 多样性高，增加PSO权重（加快收敛）
            diversity_factor = 0.7
        
        # 基于收敛速度调整权重
        if convergence_rate < self.convergence_threshold:
            # 收敛慢，增加PSO权重
            convergence_factor = 0.7
        else:
            # 收敛快，保持GA权重
            convergence_factor = 0.3
        
        # 基于进化进程调整权重
        if progress < 0.3:
            # 前期：重视全局搜索（GA）
            progress_factor = 0.3
        elif progress < 0.7:
            # 中期：平衡
            progress_factor = 0.5
        else:
            # 后期：重视局部开发（PSO）
            progress_factor = 0.8
        
        # 综合权重
        self.pso_weight = (diversity_factor + convergence_factor + progress_factor) / 3
        self.ga_weight = 1 - self.pso_weight
        
        # 记录权重历史
        self.weight_history.append((self.pso_weight, self.ga_weight))
    
    def _exchange_information(self):
        """算法间信息交换"""
        # 获取PSO中最差的粒子
        pso_particles = self.pso.particles
        worst_pso_idx = max(range(len(pso_particles)), 
                           key=lambda i: pso_particles[i].fitness)
        
        # 获取GA中最优的个体
        self.ga.population.sort()
        best_ga_individual = self.ga.population[0]
        
        # 用GA最优个体替换PSO最差粒子
        worst_particle = pso_particles[worst_pso_idx]
        worst_particle.position = best_ga_individual.genes.copy()
        worst_particle.fitness = best_ga_individual.fitness
        worst_particle.update_best()
        
        # 用PSO全局最优替换GA中的随机个体
        if len(self.ga.population) > 1:
            random_idx = np.random.randint(1, len(self.ga.population))
            random_individual = self.ga.population[random_idx]
            random_individual.genes = self.pso.global_best_position.copy()
            random_individual.fitness = self.pso.global_best_fitness
    
    def _adaptive_regrouping(self):
        """自适应重新分组"""
        # 收集所有个体信息
        all_individuals = []
        
        # PSO粒子
        for particle in self.pso.particles:
            all_individuals.append({
                'position': particle.position.copy(),
                'fitness': particle.fitness,
                'type': 'pso'
            })
        
        # GA个体
        for individual in self.ga.population:
            all_individuals.append({
                'position': individual.genes.copy(),
                'fitness': individual.fitness,
                'type': 'ga'
            })
        
        # 按适应度排序
        all_individuals.sort(key=lambda x: x['fitness'])
        
        # 重新分组：优解组给PSO，劣解组给GA
        new_pso_particles = []
        new_ga_individuals = []
        
        for i, ind_info in enumerate(all_individuals):
            if i < self.elite_size:
                # 优解组 -> PSO
                if ind_info['type'] == 'pso':
                    # 保持原有速度
                    for particle in self.pso.particles:
                        if np.allclose(particle.position, ind_info['position']):
                            new_pso_particles.append(particle)
                            break
                else:
                    # GA个体转为PSO粒子
                    velocity = np.random.uniform(-0.1, 0.1, self.dimension)
                    particle = Particle(ind_info['position'], velocity)
                    particle.fitness = ind_info['fitness']
                    particle.update_best()
                    new_pso_particles.append(particle)
            else:
                # 劣解组 -> GA
                individual = Individual(ind_info['position'])
                individual.fitness = ind_info['fitness']
                new_ga_individuals.append(individual)
        
        # 更新种群
        self.pso.particles = new_pso_particles
        self.ga.population = new_ga_individuals
        
        # 更新最优解
        self.pso._update_global_best()
        self.ga._update_best()
    
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行混合优化算法
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            最优位置, 最优适应度, 适应度历史
        """
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            
            # 自适应权重调整
            self._adaptive_weight_adjustment()
            
            # 执行PSO更新（加权）
            if self.pso_weight > 0:
                self.pso.current_iteration = iteration
                self.pso._update_particles()
                self.pso._evaluate_fitness()
                self.pso._update_global_best()
            
            # 执行GA更新（加权）
            if self.ga_weight > 0:
                self.ga.current_generation = iteration
                self.ga._create_next_generation()
                self.ga._evaluate_fitness()
                self.ga._update_best()
            
            # 信息交换
            if iteration % self.exchange_interval == 0 and iteration > 0:
                self._exchange_information()
            
            # 自适应重新分组
            if iteration % (self.exchange_interval * 2) == 0 and iteration > 0:
                self._adaptive_regrouping()
            
            # 更新全局最优
            self._update_global_best()
            
            # 记录历史
            self.fitness_history.append(self.global_best_fitness)
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            convergence_rate = self._calculate_convergence_rate()
            self.convergence_history.append(convergence_rate)
            
            if verbose and iteration % 20 == 0:
                print(f"混合算法迭代 {iteration}: "
                      f"最优适应度 = {self.global_best_fitness:.6f}, "
                      f"PSO权重 = {self.pso_weight:.3f}, "
                      f"多样性 = {diversity:.6f}")
        
        return self.global_best_position, self.global_best_fitness, self.fitness_history
    
    def get_algorithm_statistics(self) -> dict:
        """获取算法统计信息"""
        return {
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'convergence_history': self.convergence_history,
            'weight_history': self.weight_history,
            'final_pso_weight': self.pso_weight,
            'final_ga_weight': self.ga_weight,
            'elite_size': self.elite_size,
            'regular_size': self.regular_size
        }
