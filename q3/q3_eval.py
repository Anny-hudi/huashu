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

    # 导出每次决策的详细信息
    rows = []
    done = False
    total_reward = 0.0
    decision_history = []  # 记录马尔可夫决策历史
    
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

        # 记录当前状态（用于马尔可夫分析）
        current_state = {
            'decision_idx': env.decision_index,
            'queue_lengths': {bs: {slice_name: len(env.queues[bs][slice_name]) 
                                  for slice_name in env.slice_names} 
                             for bs in env.base_stations},
            'actions': action_tuples.copy()
        }
        decision_history.append(current_state)

        next_obs, reward, done, info = env.step(action_tuples)
        total_reward += reward

        # 记录详细信息
        row = {
            "decision_idx": env.decision_index - 1,  # 当前决策索引
            "reward": reward,
            "total_reward": total_reward,
            "time_ms": (env.decision_index - 1) * env.decision_interval_ms
        }
        
        # 记录每个基站的资源分配和功率
        for bs in env.base_stations:
            rb, pw = action_tuples[bs]
            row[f"{bs}_URLLC_RB"] = rb[0]
            row[f"{bs}_eMBB_RB"] = rb[1]
            row[f"{bs}_mMTC_RB"] = rb[2]
            row[f"{bs}_URLLC_P"] = pw[0]
            row[f"{bs}_eMBB_P"] = pw[1]
            row[f"{bs}_mMTC_P"] = pw[2]
            
            # 记录队列状态
            for slice_name in env.slice_names:
                queue_len = len(env.queues[bs][slice_name])
                row[f"{bs}_{slice_name}_queue_len"] = queue_len
                
                # 计算平均排队时延
                if queue_len > 0:
                    avg_queue_delay = sum(t["queue_ms"] for t in env.queues[bs][slice_name]) / queue_len
                    row[f"{bs}_{slice_name}_avg_queue_delay"] = avg_queue_delay
                else:
                    row[f"{bs}_{slice_name}_avg_queue_delay"] = 0.0
        
        # 记录超时惩罚信息
        timeout_penalties = {slice_name: 0 for slice_name in env.slice_names}
        total_delays = {slice_name: [] for slice_name in env.slice_names}
        
        # 从环境信息中提取超时惩罚（如果可用）
        if hasattr(env, 'last_step_info') and env.last_step_info:
            for slice_name in env.slice_names:
                if 'timeout_counts' in env.last_step_info:
                    timeout_penalties[slice_name] = env.last_step_info['timeout_counts'].get(slice_name, 0)
                if 'delays' in env.last_step_info:
                    total_delays[slice_name] = env.last_step_info['delays'].get(slice_name, [])
        
        for slice_name in env.slice_names:
            row[f"{slice_name}_timeout_penalty"] = timeout_penalties[slice_name]
            if total_delays[slice_name]:
                row[f"{slice_name}_avg_delay"] = np.mean(total_delays[slice_name])
                row[f"{slice_name}_max_delay"] = np.max(total_delays[slice_name])
                row[f"{slice_name}_delay_variance"] = np.var(total_delays[slice_name])
            else:
                row[f"{slice_name}_avg_delay"] = 0.0
                row[f"{slice_name}_max_delay"] = 0.0
                row[f"{slice_name}_delay_variance"] = 0.0
        
        rows.append(row)
        obs = next_obs

    # 写CSV - 扩展字段名
    fieldnames = ["decision_idx", "reward", "total_reward", "time_ms"]
    
    # 资源分配和功率字段
    for bs in env.base_stations:
        fieldnames += [f"{bs}_URLLC_RB", f"{bs}_eMBB_RB", f"{bs}_mMTC_RB",
                       f"{bs}_URLLC_P", f"{bs}_eMBB_P", f"{bs}_mMTC_P"]
    
    # 队列状态字段
    for bs in env.base_stations:
        for slice_name in env.slice_names:
            fieldnames += [f"{bs}_{slice_name}_queue_len", f"{bs}_{slice_name}_avg_queue_delay"]
    
    # 超时惩罚和时延字段
    for slice_name in env.slice_names:
        fieldnames += [f"{slice_name}_timeout_penalty", f"{slice_name}_avg_delay", 
                       f"{slice_name}_max_delay", f"{slice_name}_delay_variance"]
    
    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    
    print(f"Evaluation decisions saved to {out_csv}. Total reward={total_reward:.4f}")
    print(f"Recorded {len(rows)} decision steps with detailed metrics")


if __name__ == "__main__":
    evaluate(weights_path="q3_weights.npz", out_csv="q3_eval_decisions.csv")

