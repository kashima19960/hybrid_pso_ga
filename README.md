# Hybrid PSO-GA Algorithm with Adaptive Grouping Strategy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/kashima19960/hybrid_pso_ga)

📖 **[中文文档 / Chinese Documentation](README_ZH.md)**

## Overview

This project implements an innovative hybrid optimization algorithm that combines Particle Swarm Optimization (PSO) and Genetic Algorithm (GA) with an adaptive grouping strategy. The algorithm effectively addresses the local optima problem common in traditional optimization algorithms through dynamic weight adjustment mechanisms and population grouping control.

## 🚀 Key Features

### 1. Adaptive Grouping Control Strategy
- **Dynamic Population Division**: Splits population into elite and inferior groups based on fitness ranking
- **Elite Group**: Uses PSO for fast local search
- **Inferior Group**: Uses GA for global exploration
- **Adaptive Ratio**: Group proportions adjust based on population diversity

### 2. Triple Dynamic Weight Adjustment
- **Diversity Factor**: Adjusts based on population convergence degree
- **Convergence Speed Factor**: Monitors fitness improvement rate
- **Progress Factor**: Considers optimization phase characteristics
- **Integrated Control**: Fine-grained weight management through three-factor synthesis

### 3. Elite Migration Mechanism
- **PSO→GA**: Global best position guides GA mutation
- **GA→PSO**: Elite individuals replace poor particles in PSO
- **Bidirectional Exchange**: Promotes collaborative evolution

### 4. Adaptive Regrouping
- **Performance-based Trigger**: Regrouping based on performance feedback
- **Stagnation Prevention**: Prevents algorithm from getting stuck
- **Diversity Maintenance**: Preserves population diversity

## 📁 Project Structure

```
hybrid_pso_ga/
├── algorithms/              # Core algorithm implementations
│   ├── __init__.py         # Module initialization
│   ├── pso.py              # Particle Swarm Optimization
│   ├── ga.py               # Genetic Algorithm
│   └── hybrid_pso_ga.py    # Adaptive Hybrid PSO-GA Algorithm
├── benchmark/              # Test function suite
│   ├── __init__.py         # Module initialization  
│   └── test_functions.py   # Standard test functions (Sphere, Rastrigin, etc.)
├── experiments/            # Experiments and analysis
│   ├── __init__.py         # Module initialization
│   ├── comparison.py       # Algorithm comparison experiments
│   └── visualization.py    # Results visualization
├── data/                   # Experimental data and results
│   ├── experiment_results.json      # Experiment result data
│   ├── experiment_report.csv        # Statistical report
│   ├── performance_comparison.png   # Performance comparison chart
│   ├── convergence_curves.png       # Convergence curves
│   ├── algorithm_ranking.png        # Algorithm ranking heatmap
│   ├── average_ranking.png          # Average ranking chart
│   ├── statistical_analysis.png     # Statistical analysis
│   └── 2d_optimization.png          # 2D optimization trajectory
├── requirements.txt        # Python dependencies
├── main.py                # Main program entry
├── README.md              # Project documentation (English)
├── README_ZH.md           # Project documentation (Chinese)
└── .gitignore             # Git ignore configuration
```

## 🧪 Test Function Suite

The project includes 7 classical optimization test functions covering different optimization challenges:

| Function | Type | Characteristics | Dimensions | Search Domain |
|----------|------|----------------|------------|---------------|
| Sphere | Unimodal | Simple convex function | 2D | [-5.12, 5.12] |
| Rosenbrock | Valley | Difficult convergence | 2D | [-2.048, 2.048] |
| Rastrigin | Multimodal | Highly multimodal | 2D | [-5.12, 5.12] |
| Griewank | Multimodal | Multiple local optima | 2D | [-600, 600] |
| Ackley | Exponential | Exponential characteristics | 2D | [-32, 32] |
| Schwefel | Deceptive | Distant global optimum | 2D | [-500, 500] |
| Levy | Multimodal | Complex landscape | 2D | [-10, 10] |

## 📋 Requirements

- Python 3.8 or higher

## 🛠️ Installation & Usage

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/kashima19960/hybrid_pso_ga
cd hybrid_pso_ga

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start

```bash
# Run complete experimental workflow
python main.py

# Or execute step by step:
# 1. Run algorithm comparison
python experiments/comparison.py

# 2. Generate visualizations
python experiments/visualization.py
```

## 📊 Experimental Results
reference the `data/` directory

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ FAQ
1. **Memory Issues**
   ```python
   # If you encounter memory issues, try reducing the population size or the number of iterations
   optimizer = AdaptiveHybridPSOGA(population_size=30, max_iterations=100)
   ```
2. **Slow Convergence**
   ```python
   # If convergence is slow, consider adjusting the parameter settings
   optimizer = AdaptiveHybridPSOGA(
       alpha=0.7,              # Increase elite group ratio
       exchange_interval=5     # Decrease exchange interval
   )
   ```