import csv
import numpy as np

# 兼容导入
if __package__ is None or __package__ == "":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from q3.multi_bs_env import MultiBSInterferenceEnv, enumerate_rb_actions, enumerate_power_actions
    from q3.maddqn_agents import MultiAgentController
else:
    from .multi_bs_env import MultiBSInterferenceEnv, enumerate_rb_actions, enumerate_power_actions
    from .maddqn_agents import MultiAgentController


def evaluate(weights_path: str = "q3_weights.npz", out_csv: str = "q3_eval_decisions.csv"):
    env = MultiBSInterferenceEnv(seed=123)
    rb_actions = enumerate_rb_actions(env.rb_total_per_bs, {
        "URLLC": env.slice_params["URLLC"]["rb_unit"],
        "eMBB": env.slice_params["eMBB"]["rb_unit"],
        "mMTC": env.slice_params["mMTC"]["rb_unit"],
    })
    pwr_actions = enumerate_power_actions([10, 20, 30])

    obs = env.reset()
    obs_dim = obs[env.base_stations[0]].shape[0]
    mac = MultiAgentController(agent_names=env.base_stations, obs_dim=obs_dim,
                               rb_actions=rb_actions, pwr_actions=pwr_actions,
                               lr=0.0, gamma=0.0, eps=0.0)
    try:
        mac.load_weights(weights_path)
        # 评估使用贪心：将epsilon置0
        for a in mac.agents.values():
            a.epsilon = 0.0
        print(f"Loaded weights from {weights_path}")
    except Exception as e:
        print(f"Warning: cannot load weights: {e}. Using random policy.")

    # 导出每次决策的 (BS, URLLC_RB, eMBB_RB, mMTC_RB, URLLC_P, eMBB_P, mMTC_P, reward)
    rows = []
    done = False
    total_reward = 0.0
    while not done:
        # 构造动作（贪心）
        action_tuples = {}
        action_indices = {}
        for bs in env.base_stations:
            agent = mac.agents[bs]
            action_idx = agent.select_action(obs[bs])
            rb_idx, pwr_idx = mac.joint_actions[action_idx]
            rb = rb_actions[rb_idx]
            pw = pwr_actions[pwr_idx]
            action_tuples[bs] = (rb, pw)
            action_indices[bs] = (rb_idx, pwr_idx)

        next_obs, reward, done, info = env.step(action_tuples)
        total_reward += reward

        # 记录
        row = {"reward": reward}
        for bs in env.base_stations:
            rb, pw = action_tuples[bs]
            row[f"{bs}_URLLC_RB"] = rb[0]
            row[f"{bs}_eMBB_RB"] = rb[1]
            row[f"{bs}_mMTC_RB"] = rb[2]
            row[f"{bs}_URLLC_P"] = pw[0]
            row[f"{bs}_eMBB_P"] = pw[1]
            row[f"{bs}_mMTC_P"] = pw[2]
        rows.append(row)

        obs = next_obs

    # 写CSV
    fieldnames = ["reward"]
    for bs in env.base_stations:
        fieldnames += [f"{bs}_URLLC_RB", f"{bs}_eMBB_RB", f"{bs}_mMTC_RB",
                       f"{bs}_URLLC_P", f"{bs}_eMBB_P", f"{bs}_mMTC_P"]
    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Evaluation decisions saved to {out_csv}. Total reward={total_reward:.4f}")


if __name__ == "__main__":
    evaluate(weights_path="q3_weights.npz", out_csv="q3_eval_decisions.csv")

