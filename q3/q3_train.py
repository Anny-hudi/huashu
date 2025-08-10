import numpy as np
from typing import Dict, Tuple

# 兼容直接脚本运行与作为包运行
if __package__ is None or __package__ == "":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from q3.multi_bs_env import MultiBSInterferenceEnv, enumerate_rb_actions, enumerate_power_actions
    from q3.maddqn_agents import MultiAgentController
else:
    from .multi_bs_env import MultiBSInterferenceEnv, enumerate_rb_actions, enumerate_power_actions
    from .maddqn_agents import MultiAgentController


def build_actions(env: MultiBSInterferenceEnv):
    rb_actions = enumerate_rb_actions(env.rb_total_per_bs, {
        "URLLC": env.slice_params["URLLC"]["rb_unit"],
        "eMBB": env.slice_params["eMBB"]["rb_unit"],
        "mMTC": env.slice_params["mMTC"]["rb_unit"],
    })
    # 降低动作维度：功率离散为 {10, 20, 30} dBm
    pwr_actions = enumerate_power_actions([10, 20, 30])
    return rb_actions, pwr_actions


def train(num_episodes: int = 5, seed: int = 42, save_path: str = "q3_weights.npz"):
    env = MultiBSInterferenceEnv(seed=seed)
    rb_actions, pwr_actions = build_actions(env)

    # 构造一次观测以获知维度
    obs = env.reset()
    obs_dim = obs[env.base_stations[0]].shape[0]

    mac = MultiAgentController(agent_names=env.base_stations, obs_dim=obs_dim,
                               rb_actions=rb_actions, pwr_actions=pwr_actions,
                               lr=0.05, gamma=0.95, eps=0.3)

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # 选行动
            action_tuples = {}
            # 为学习记录 (rb_idx, pwr_idx)
            chosen_indices = {}
            for bs in env.base_stations:
                # agent 选联合动作
                agent = mac.agents[bs]
                action_idx = agent.select_action(obs[bs])
                rb_idx, pwr_idx = mac.joint_actions[action_idx]
                action_tuples[bs] = (rb_actions[rb_idx], pwr_actions[pwr_idx])
                chosen_indices[bs] = (rb_idx, pwr_idx)

            next_obs, reward, done, info = env.step(action_tuples)
            ep_reward += reward

            # 学习（全局奖励）
            for bs in env.base_stations:
                mac.agents[bs].update(obs[bs], mac.joint_actions.index(chosen_indices[bs]), reward, next_obs[bs], done)

            obs = next_obs

        print(f"Episode {ep+1}/{num_episodes}: reward={ep_reward:.4f}")

    # 保存权重
    try:
        mac.save_weights(save_path)
        print(f"Saved weights to {save_path}")
    except Exception as e:
        print(f"Warning: failed to save weights due to {e}")


if __name__ == "__main__":
    train(num_episodes=3, save_path="q3_weights.npz")

