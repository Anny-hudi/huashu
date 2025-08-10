# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**华数杯数学建模竞赛解决方案** - A comprehensive 5-question mathematical modeling competition solution focusing on 5G network optimization and energy efficiency. The project demonstrates progressive complexity from single-base station resource allocation to intelligent multi-agent systems for green communication.

## Architecture Structure

### Question Evolution Architecture

```
q1/        # Single-base station optimal allocation
├── q1_solution.py       # Core implementation
├── q1_optimized.py      # Improved solutions
├── q1_plot.py          # Visualization
└── 第一题解题思路.md      # Methodology explanation

q2/        # Multi-base station collaborative optimization
├── q2_solution.py       # Core multi-BS implementation
├── q2_hybrid_solution.py # Hybrid heuristic approach
├── q2_plot.py          # Multi-BS visualization
└── 第二题解题思路和算法总结.md

q3/        # Multi-agent deep reinforcement learning
├── multi_bs_env.py     # Gym environment for 3 BS scenario
├── maddqn_agents.py    # MADDPG algorithm implementation
├── q3_train.py         # Training scripts
├── q3_eval.py          # Evaluation framework
└── q3_plot.py          # Result visualization

q4/        # Large-scale heterogeneous network optimization
├── q4_solution.py      # Genetic algorithm + DRL hybrid
├── q4_plot.py          # Complex network visualization
└── 第4题解题思路.md      # Advanced optimization strategy

q5/        # PPO-based energy efficiency optimization
├── q5_solution.py      # PPO implementation for energy efficiency
├── q5_trained_model.pth # Saved trained model
└── 第五题解题思路.md      # Green communication methodology
```

### Data Structure

```
data_1/    # Small scale data (Q1)
data_2/    # Medium scale data (Q2)  
data_3/    # Three base stations data (Q3)
data_4/    # Heterogeneous network data (Q4-Q5)
data_5/    # Energy efficiency scenarios

├── MBS_1大规模衰减.csv      # Large-scale fading for MBS
├── SBS_1/2/3小规模瑞丽衰减.csv # Small-scale Rayleigh fading
├── 用户位置4.csv           # User position data
└── 用户任务流4.csv         # Task flow patterns
```

## Key Dependencies & Setup

### Core Requirements
```bash
pip install -r requirements.txt
# Includes: numpy, matplotlib, pandas, scipy, joblib
# DRL dependencies: gymnasium, torch (implied in q3/q5)
```

### Optional Enhanced Dependencies
```bash
pip install torch torchvision gymnasium tensorboard  # For q3/q5 RL
pip install scikit-learn plotly seaborn             # Advanced analytics
```

## Development Commands

### Running Individual Questions

#### Q1: Single Base Station
```bash
python q1_solution.py              # Basic implementation
python q1_optimized.py            # Enhanced version
# Generates: joint_plots, user_analysis charts
```

#### Q2: Multi-Base Station
```bash
python q2_solution.py             # Base solution
python q2_hybrid_solution.py      # Advanced hybrid approach
# Results: qos_comparison.png, resource_allocation.png
```

#### Q3: MADDPG Training
```bash
cd q3/
python q3_train.py               # Start MADDPG training
python q3_eval.py                # Evaluate trained model
python q3_plot.py                # Generate plots
# Produces: BS*_power_levels.png, reward_over_time.png
```

#### Q4: Hybrid Optimization
```bash
python q4/q4_solution.py         # GA + DRL hybrid
python q4/q4_plot.py             # Complex visualizations
```

#### Q5: PPO Energy Optimization
```bash
python q5/q5_solution.py         # PPO training
# Saves: q5_trained_model.pth, training_metrics.png
```

### Comprehensive Analysis
```bash
# Generate all results and plots
python all.py                     # Custom script for full pipeline
# Or run individually for debugging
```

## Algorithm Complexity Map

| Question | Algorithm Type | Complexity | Key Metrics |
|----------|----------------|------------|-------------|
| Q1 | Greedy + Iterative | O(n²) | Throughput maximization |
| Q2 | Hybrid Heuristic | O(n³) | QoS balancing |
| Q3 | MADDPG | O(n²) + Training | Adaptive learning |
| Q4 | GA + DRL | O(n log n) | Global optimization |
| Q5 | PPO Energy | O(n²) + Training | Energy efficiency |

## Key Implementation Patterns

### Data Loading Pattern
```python
# Standard data loading across all questions
import pandas as pd
import numpy as np

def load_question_data(question_num):
    base_path = f"data_{question_num}"
    return {
        'user_positions': pd.read_csv(f"{base_path}/用户位置{question_num}.csv"),
        'task_flow': pd.read_csv(f"{base_path}/用户任务流{question_num}.csv"),
        'fading_large': pd.read_csv(f"{base_path}/*大规模衰减.csv"),
        'fading_small': pd.read_csv(f"{base_path}/*小规模瑞丽衰减.csv")
    }
```

### Common Calculations
```python
def calculate_sinr(rx_power, interference, noise_dbm):
    """Standard SINR calculation pattern used across questions"""
    noise_mw = 10**((noise_dbm - 174 - 10*np.log10(bandwidth_hz))/10)
    return rx_power / (interference + noise_mw)

def calculate_rate(sinr, bandwidth_hz):
    """Shannon capacity calculation"""
    return bandwidth_hz * np.log2(1 + sinr)
```

## Visualization Standards

### Plot Generation
- Location plots: Scatter plot for user/base station positions
- Performance plots: Line chart for rewards/convergence  
- Resource allocation: Heatmap for RB/energy usage
- Network topology: Graph visualization with edges colored by throughput

### Output Directory Structure
```
q*/outputs/
├── *.png                   # Auto-generated plots
├── *.csv                   # Performance metrics
└── logs/                   # Training logs
```

## RL Training Guidelines

### Model Saving/Loading Pattern
```python
# Used in q3/q5
import torch

CONFIG = {
    "model_path": "q5_trained_model.pth",
    "buffer_size": 100000,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "max_episodes": 3000
}

def save_model(agent, path):
    torch.save(agent.state_dict(), path)

def load_model(agent, path):
    agent.load_state_dict(torch.load(path))
    agent.eval()
```

### Hyperparameter Tuning
- **Environment**: Gymnasium-based custom environments
- **Agents**: Shared critic vs independent critics
- **Reward**: Weight balancing (QoS vs energy vs fairness)

## Debugging Strategies

### Common Issues
1. **Memory Issues**: Use `joblib` for large arrays in q3/q4
2. **Training Instability**: Reduce learning rate in q5 
3. **Convergence**: Increase episodes from 1000→3000 for q5
4. **Data Loading**: Ensure column names match expected format

### Debug Tools
```python
debug_info.py          # System configuration check
# Enable debug mode for any question
export DEBUG=true && python q*/q*_solution.py
```

## File Organization Best Practices

### Naming Convention
- `q[num]_solution.py`: Core problem implementation
- `q[num]_plot.py`: Visualization utilities  
- `q[num]_train.py`: Training scripts (if RL)
- `*.md`: Chinese documentation for competition requirements

### Development Workflow
1. **Start Simple**: Begin with basic implementations
2. **Iterate**: Use `q[num]_optimized.py` for improvements
3. **Validate**: Cross-reference with documented results
4. **Visualize**: Always include plotting for verification

## Competition-Specific Notes

### Mathematical Modeling Requirements
- All solutions include detailed Chinese methodology documents
- Formulas use LaTeX notation per competition standards
- Results are reproducible with provided datasets
- Implementation follows mathematical rigor expected

### Evaluation Metrics
- **Q1**: System throughput, user satisfaction
- **Q2**: QoS balancing across slices
- **Q3**: RL convergence and adaptability  
- **Q4**: Global optimization performance
- **Q5**: Energy efficiency vs QoS trade-off

## Advanced Features

### Q5 Energy Model Details
- **Base Consumption**: MBS 130W, SBS 6.8W
- **RF Efficiency**: 28% per base station
- **Modes**: Direct(1x), Relay(1.3x), D2D(0.7x) energy multipliers
- **Optimization**: α=0.7(QoS) + β=0.25(Energy) + γ=0.05(Fairness)

### Multi-Agent Coordination
- **Architecture**: Centralized critic, decentralized actors
- **Communication**: Shared experiences without direct coordination  
- **Scalability**: 1+3 base station design extensible to arbitrary networks