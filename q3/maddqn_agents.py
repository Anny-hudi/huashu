import numpy as np
from typing import Dict, Tuple, List


class LinearQAgent:
    """
    线性函数近似的 Q-learning Agent：Q(s,a) = w_a^T s
    - 动作空间离散（索引集合）
    - 使用 epsilon-greedy 探索
    """

    def __init__(self, name: str, obs_dim: int, num_actions: int,
                 learning_rate: float = 0.05, gamma: float = 0.95,
                 epsilon: float = 0.2, epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 seed: int = 42):
        self.name = name
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(num_actions, obs_dim) * 0.01

        # 临时缓存上一状态-动作
        self.prev_obs = None
        self.prev_action = None

    def select_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        q_values = self.weights.dot(obs)
        return int(np.argmax(q_values))

    def update(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        # 目标：r + gamma * max_a' Q(next, a')
        q_values_next = self.weights.dot(next_obs)
        target = reward if done else (reward + self.gamma * float(np.max(q_values_next)))

        # 当前 Q
        q_values = self.weights.dot(obs)
        td_error = target - q_values[action]
        # 梯度 = obs
        self.weights[action] += self.learning_rate * td_error * obs

        # 衰减 epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class MultiAgentController:
    """
    多智能体控制器：每个基站一个 Agent。

    每个 Agent 的动作索引映射到 (RB三元组索引, Pwr三元组索引)。联合动作空间 = RB动作集合 x Pwr动作集合。
    """

    def __init__(self, agent_names: List[str], obs_dim: int,
                 rb_actions: List[Tuple[int, int, int]],
                 pwr_actions: List[Tuple[int, int, int]],
                 lr: float = 0.05, gamma: float = 0.95, eps: float = 0.3):
        self.agent_names = agent_names
        self.rb_actions = rb_actions
        self.pwr_actions = pwr_actions
        self.joint_actions = [(i, j) for i in range(len(rb_actions)) for j in range(len(pwr_actions))]
        self.num_actions = len(self.joint_actions)

        self.agents: Dict[str, LinearQAgent] = {
            name: LinearQAgent(name, obs_dim, self.num_actions, learning_rate=lr, gamma=gamma, epsilon=eps)
            for name in agent_names
        }

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        actions = {}
        for name in self.agent_names:
            action_idx = self.agents[name].select_action(obs_dict[name])
            rb_idx, pwer_idx = self.joint_actions[action_idx]
            actions[name] = (self.rb_actions[rb_idx], self.pwr_actions[pwer_idx])
        return actions

    def learn(self, obs_dict: Dict[str, np.ndarray], action_dict: Dict[str, Tuple[int, int]],
              reward: float, next_obs_dict: Dict[str, np.ndarray], done: bool):
        # 为简单：使用全局奖励更新每个 Agent
        for name in self.agent_names:
            agent = self.agents[name]
            # 将 (rb_idx, pwr_idx) 映射回联合动作索引
            joint_idx = action_dict[name]
            action_idx = self.joint_actions.index(joint_idx)
            agent.update(obs_dict[name], action_idx, reward, next_obs_dict[name], done)

    def save_weights(self, file_path: str):
        data = {}
        for name, agent in self.agents.items():
            data[f"weights_{name}"] = agent.weights
            data[f"epsilon_{name}"] = np.array([agent.epsilon], dtype=np.float32)
        np.savez(file_path, **data)

    def load_weights(self, file_path: str):
        d = np.load(file_path)
        for name, agent in self.agents.items():
            key_w = f"weights_{name}"
            key_e = f"epsilon_{name}"
            if key_w in d:
                agent.weights = d[key_w]
            if key_e in d:
                agent.epsilon = float(d[key_e][0])

