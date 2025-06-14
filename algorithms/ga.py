"""
author: 木人舟
brief: 遗传算法（GA）实现
contact:CodingCV@outlook.com
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
import copy


class Individual:
    """个体类，表示GA算法中的单个个体"""
    
    def __init__(self, genes: np.ndarray):
        """
        初始化个体
        
        Args:
            genes: 个体基因（实数编码）
        """
        self.genes = genes.copy()
        self.fitness = float('inf')
    
    def __lt__(self, other):
        """用于排序比较"""
        return self.fitness < other.fitness


class GA:
    """遗传算法类"""
    
    def __init__(self,
                 fitness_func: Callable[[np.ndarray], float],
                 dimension: int,
                 bounds: List[Tuple[float, float]],
                 population_size: int = 30,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2,
                 selection_method: str = 'tournament',
                 tournament_size: int = 3):
        """
        初始化遗传算法
        
        Args:
            fitness_func: 适应度函数
            dimension: 搜索空间维度
            bounds: 每个维度的边界
            population_size: 种群大小
            max_generations: 最大进化代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英个体数量
            selection_method: 选择方法 ('tournament', 'roulette', 'rank')
            tournament_size: 锦标赛选择的比赛规模
        """
        self.fitness_func = fitness_func
        self.dimension = dimension
        self.bounds = bounds
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        
        # 算法状态
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.fitness_history = []
        self.current_generation = 0
        
        # 初始化种群
        self._initialize_population()
    
    def _initialize_population(self):
        """初始化种群"""
        self.population = []
        
        for _ in range(self.population_size):
            # 随机生成个体基因
            genes = np.array([
                np.random.uniform(bound[0], bound[1])
                for bound in self.bounds
            ])
            individual = Individual(genes)
            self.population.append(individual)
        
        # 评估初始适应度
        self._evaluate_fitness()
        self._update_best()
    
    def _evaluate_fitness(self):
        """评估种群中所有个体的适应度"""
        for individual in self.population:
            individual.fitness = self.fitness_func(individual.genes)
    
    def _update_best(self):
        """更新最优个体"""
        self.population.sort()  # 按适应度排序
        if self.best_individual is None or self.population[0].fitness < self.best_individual.fitness:
            self.best_individual = copy.deepcopy(self.population[0])
    
    def _tournament_selection(self) -> Individual:
        """锦标赛选择"""
        tournament = np.random.choice(self.population, self.tournament_size, replace=False)
        return min(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self) -> Individual:
        """轮盘赌选择"""
        # 转换为最大化问题（适应度越大越好）
        max_fitness = max(ind.fitness for ind in self.population)
        fitness_values = [max_fitness - ind.fitness + 1e-10 for ind in self.population]
        total_fitness = sum(fitness_values)
        
        # 轮盘赌选择
        pick = np.random.uniform(0, total_fitness)
        current = 0
        for i, individual in enumerate(self.population):
            current += fitness_values[i]
            if current >= pick:
                return individual
        return self.population[-1]
    
    def _rank_selection(self) -> Individual:
        """基于排名的选择"""
        # 按适应度排序（最优在前）
        sorted_pop = sorted(self.population)
        # 排名权重（排名越高权重越大）
        ranks = np.arange(len(sorted_pop), 0, -1)
        probabilities = ranks / np.sum(ranks)
        
        # 根据概率选择
        chosen_idx = np.random.choice(len(sorted_pop), p=probabilities)
        return sorted_pop[chosen_idx]
    
    def _selection(self) -> Individual:
        """选择操作"""
        if self.selection_method == 'tournament':
            return self._tournament_selection()
        elif self.selection_method == 'roulette':
            return self._roulette_selection()
        elif self.selection_method == 'rank':
            return self._rank_selection()
        else:
            raise ValueError(f"未知的选择方法: {self.selection_method}")
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作（混合交叉）"""
        if np.random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # 混合交叉（BLX-α）
        alpha = 0.5
        child1_genes = np.zeros(self.dimension)
        child2_genes = np.zeros(self.dimension)
        
        for i in range(self.dimension):
            # 计算交叉范围
            min_val = min(parent1.genes[i], parent2.genes[i])
            max_val = max(parent1.genes[i], parent2.genes[i])
            diff = max_val - min_val
            
            # 扩展范围
            low = min_val - alpha * diff
            high = max_val + alpha * diff
            
            # 边界约束
            low = max(low, self.bounds[i][0])
            high = min(high, self.bounds[i][1])
            
            # 生成子代
            child1_genes[i] = np.random.uniform(low, high)
            child2_genes[i] = np.random.uniform(low, high)
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _mutation(self, individual: Individual) -> Individual:
        """变异操作（高斯变异）"""
        mutated = copy.deepcopy(individual)
        
        for i in range(self.dimension):
            if np.random.random() < self.mutation_rate:
                # 高斯变异
                sigma = (self.bounds[i][1] - self.bounds[i][0]) * 0.1
                mutation_value = np.random.normal(0, sigma)
                mutated.genes[i] += mutation_value
                
                # 边界约束
                mutated.genes[i] = np.clip(mutated.genes[i],
                                         self.bounds[i][0],
                                         self.bounds[i][1])
        
        return mutated
    
    def _create_next_generation(self):
        """创建下一代种群"""
        next_population = []
        
        # 精英保留
        self.population.sort()
        for i in range(self.elite_size):
            next_population.append(copy.deepcopy(self.population[i]))
        
        # 生成剩余个体
        while len(next_population) < self.population_size:
            # 选择父代
            parent1 = self._selection()
            parent2 = self._selection()
            
            # 交叉产生子代
            child1, child2 = self._crossover(parent1, parent2)
            
            # 变异
            child1 = self._mutation(child1)
            child2 = self._mutation(child2)
            
            next_population.extend([child1, child2])
        
        # 确保种群大小不超过设定值
        self.population = next_population[:self.population_size]
    
    def optimize(self, verbose: bool = False) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行遗传算法优化
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            最优位置, 最优适应度, 适应度历史
        """
        for generation in range(self.max_generations):
            self.current_generation = generation
            
            # 创建下一代
            self._create_next_generation()
            
            # 评估适应度
            self._evaluate_fitness()
            
            # 更新最优个体
            self._update_best()
            
            # 记录历史
            self.fitness_history.append(self.best_individual.fitness)
            
            if verbose and generation % 10 == 0:
                print(f"GA代数 {generation}: 最优适应度 = {self.best_individual.fitness:.6f}")
        
        return self.best_individual.genes, self.best_individual.fitness, self.fitness_history
    
    def get_population_info(self) -> List[dict]:
        """获取种群信息，用于混合算法"""
        population_info = []
        for i, individual in enumerate(self.population):
            population_info.append({
                'index': i,
                'genes': individual.genes.copy(),
                'fitness': individual.fitness
            })
        return population_info
    
    def set_population_from_info(self, population_info: List[dict]):
        """从信息设置种群状态，用于混合算法"""
        self.population = []
        for info in population_info:
            individual = Individual(info['genes'])
            individual.fitness = info['fitness']
            self.population.append(individual)
    
    def get_elite_individuals(self, n: int) -> List[Individual]:
        """获取精英个体"""
        self.population.sort()
        return [copy.deepcopy(ind) for ind in self.population[:n]]
