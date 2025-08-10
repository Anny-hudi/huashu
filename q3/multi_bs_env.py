import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class MultiBSInterferenceEnv:
    """
    多基站+干扰+功率控制 环境（问题三数据 data_3）。

    - 基站集合: BS1, BS2, BS3
    - 切片: URLLC, eMBB, mMTC
    - 每个基站每次决策输出: (URLLC_RB, eMBB_RB, mMTC_RB), (URLLC_P, eMBB_P, mMTC_P)
      其中功率单位为 dBm，范围 [10, 30]，按离散等级选择
    - 干扰: 其他基站同切片功率对目标用户构成干扰，按附录(1)(5)公式计算
    - 决策周期: 100 ms，总时长 1000 ms，共 10 次决策
    - 数据: data_3/ 目录

    简化与建模假设（保证可跑且结构清晰）：
    1) 用户接入：每次决策前，将每个用户分配给当前“等效增益”最大的基站（基于 φ+|h|^2 的线性值）。
    2) 任务生成：按照 任务流3.csv 在该决策时刻对应概率触发一次（布尔）；数据量按表1范围均匀采样。
    3) 切片调度：同一切片的 RB 在该基站内由该切片用户均分带宽（近似）。
    4) 时延：总时延 L = Q(排队) + T(传输)，若超过 SLA 记惩罚值。
    5) 噪声功率：N0 = -174 + 10*log10(i*b) + NF，b=360kHz，NF=7dB。
    6) 速率：r = i*b*log2(1 + SINR) （bps），输出转为 Mbps。
    7) 干扰：同一切片，其他基站对该用户的干扰为 ∑ p_rx_other。

    输出：用于多智能体（每基站一个 agent）的观测、全局奖励。
    """

    slice_names: List[str] = ["URLLC", "eMBB", "mMTC"]

    def __init__(self, data_dir: str = "data_3", decision_interval_ms: int = 100,
                 total_time_ms: int = 1000,
                 rb_total_per_bs: int = 50,
                 power_min_dbm: int = 10,
                 power_max_dbm: int = 30,
                 seed: int = 42):
        np.random.seed(seed)

        self.data_dir = data_dir
        self.decision_interval_ms = decision_interval_ms
        self.total_time_ms = total_time_ms
        self.num_decisions = total_time_ms // decision_interval_ms
        self.rb_total_per_bs = rb_total_per_bs
        self.power_min_dbm = power_min_dbm
        self.power_max_dbm = power_max_dbm

        # 常量
        self.bandwidth_per_rb_hz = 360e3
        self.noise_spectral_density_dbm_per_hz = -174
        self.noise_figure_db = 7

        # 切片 SLA 与数据量范围（表1）
        self.slice_params = {
            "URLLC": {
                "rb_unit": 10,
                "rate_sla_mbps": 10,
                "delay_sla_ms": 5,
                "penalty": 5,
                "data_range_mbit": (0.01, 0.012),
                "alpha": 0.95,
            },
            "eMBB": {
                "rb_unit": 5,
                "rate_sla_mbps": 50,
                "delay_sla_ms": 100,
                "penalty": 3,
                "data_range_mbit": (0.10, 0.12),
            },
            "mMTC": {
                "rb_unit": 2,
                "rate_sla_mbps": 1,
                "delay_sla_ms": 500,
                "penalty": 1,
                "data_range_mbit": (0.012, 0.014),
            },
        }

        # 用户集合（依据数据列名）
        self.urllc_users = [f"U{i}" for i in range(1, 7)]
        self.embb_users = [f"e{i}" for i in range(1, 13)]
        self.mmtc_users = [f"m{i}" for i in range(1, 31)]
        self.all_users = self.urllc_users + self.embb_users + self.mmtc_users

        # 基站集合
        self.base_stations = ["BS1", "BS2", "BS3"]

        # 加载数据
        self.task_flow = pd.read_csv(f"{self.data_dir}/任务流3.csv")
        self.user_positions = pd.read_csv(f"{self.data_dir}/用户位置3.csv")
        self.large_scale = {
            bs: pd.read_csv(f"{self.data_dir}/{bs}_大规模衰减.csv")
            for bs in self.base_stations
        }
        # 加载每个基站的小规模瑞丽衰减；若某个文件缺失则回退为 BS1
        self.small_scale = {}
        try:
            self.small_scale["BS1"] = pd.read_csv(f"{self.data_dir}/BS1_小规模瑞丽衰减.csv")
        except Exception as e:
            raise RuntimeError(f"缺少 {self.data_dir}/BS1_小规模瑞丽衰减.csv") from e
        for bs in ["BS2", "BS3"]:
            try:
                self.small_scale[bs] = pd.read_csv(f"{self.data_dir}/{bs}_小规模瑞丽衰减.csv")
            except Exception:
                self.small_scale[bs] = self.small_scale["BS1"]

        # 队列：每个基站、每个切片一个队列（队列元素: dict(task)）
        self.queues: Dict[str, Dict[str, List[dict]]] = {
            bs: {slice_name: [] for slice_name in self.slice_names}
            for bs in self.base_stations
        }

        # 用户归属：每个用户当前接入的基站
        self.user_bs_map: Dict[str, str] = {}

        # 时间索引
        self.decision_index = 0

    # ============== 环境接口 ==============
    def reset(self):
        self.queues = {
            bs: {slice_name: [] for slice_name in self.slice_names}
            for bs in self.base_stations
        }
        self.user_bs_map = {}
        self.decision_index = 0

        # 初始用户接入分配
        self._assign_users_to_bs(self._row_index_from_decision(self.decision_index))
        # 生成初始任务
        self._spawn_tasks(self._row_index_from_decision(self.decision_index))
        return self._build_observations(self._row_index_from_decision(self.decision_index))

    def step(self, joint_actions: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]):
        """
        joint_actions: { bs_name: ((u_rb, e_rb, m_rb), (u_p, e_p, m_p)) }
        返回: obs_dict, global_reward, done, info
        """
        row_idx = self._row_index_from_decision(self.decision_index)

        # 执行调度与传输，计算奖励
        total_qos = 0.0
        total_tasks = 0

        # 预先构造每个 BS 的切片功率，供干扰使用
        slice_powers_dbm = {
            bs: {"URLLC": joint_actions[bs][1][0],
                 "eMBB": joint_actions[bs][1][1],
                 "mMTC": joint_actions[bs][1][2]}
            for bs in self.base_stations
        }

        # 每个 BS 处理其队列（按每用户RB单位，优先级调度）
        for bs in self.base_stations:
            (u_rb, e_rb, m_rb), (u_p, e_p, m_p) = joint_actions[bs]
            alloc = {"URLLC": u_rb, "eMBB": e_rb, "mMTC": m_rb}
            pwrs = {"URLLC": u_p, "eMBB": e_p, "mMTC": m_p}

            for slice_name in self.slice_names:
                rb = alloc[slice_name]
                tasks = self.queues[bs][slice_name]
                if rb <= 0 or not tasks:
                    continue

                unit = self.slice_params[slice_name]["rb_unit"]
                max_users = max(0, rb // unit)
                if max_users == 0:
                    continue

                # 选择要服务的任务
                if slice_name == "URLLC":
                    # 紧急优先（等待长者优先）
                    selected = sorted(tasks, key=lambda t: t["queue_ms"], reverse=True)[:max_users]
                else:
                    # eMBB/mMTC：选择信道更好的（估计增益）
                    def est_gain(t):
                        user = t["user"]
                        phi_db = float(self.large_scale[bs].iloc[row_idx][user])
                        h = float(self.small_scale[bs].iloc[row_idx][user])
                        return (10 ** (phi_db / 10.0)) + (abs(h) ** 2)

                    selected = sorted(tasks, key=est_gain, reverse=True)[:max_users]

                # 为选中的用户计算传输与QoS
                completed = []
                for t in selected:
                    user = t["user"]
                    sinr_linear = self._calc_sinr_with_interference(
                        row_idx=row_idx,
                        user=user,
                        serving_bs=bs,
                        slice_name=slice_name,
                        rb_num=unit,
                        slice_powers_dbm=slice_powers_dbm,
                    )
                    rate_mbps = self._calc_rate_mbps(sinr_linear, unit)
                    data_mbit = t["data_mbit"]
                    transmission_ms = (data_mbit / rate_mbps) * 1000.0 if rate_mbps > 1e-9 else 1e9
                    total_delay_ms = max(0.0, t["queue_ms"] + transmission_ms)

                    qos = self._calc_qos(slice_name, rate_mbps, total_delay_ms, unit)
                    total_qos += qos
                    total_tasks += 1

                    if total_delay_ms <= self.slice_params[slice_name]["delay_sla_ms"]:
                        completed.append(t)

                # 完成出队
                for c in completed:
                    if c in tasks:
                        tasks.remove(c)

        # 时间推进到下一个决策点：更新队列等待时间，重新接入、产生命令
        self.decision_index += 1
        done = self.decision_index >= self.num_decisions
        next_row_idx = self._row_index_from_decision(self.decision_index)

        # 每个切片队列中的任务增加 100ms 排队时间
        for bs in self.base_stations:
            for slice_name in self.slice_names:
                for t in self.queues[bs][slice_name]:
                    t["queue_ms"] += self.decision_interval_ms

        # 重新分配接入（用户移动，信道变化，基于新时刻）
        if not done:
            self._assign_users_to_bs(next_row_idx)
            # 产生新任务
            self._spawn_tasks(next_row_idx)

        # 观测
        obs = self._build_observations(next_row_idx)

        # 奖励重塑：切片权重 + 队列超时惩罚
        # 基于平均QoS（若无任务则为0）
        base_reward = (total_qos / max(1, total_tasks)) if total_tasks > 0 else 0.0

        # 额外惩罚：队列中超过各自SLA的任务
        overdue_penalty = 0.0
        for bs in self.base_stations:
            for slice_name in self.slice_names:
                sla = self.slice_params[slice_name]["delay_sla_ms"]
                over = sum(1 for t in self.queues[bs][slice_name] if t["queue_ms"] > sla)
                if over > 0:
                    if slice_name == "URLLC":
                        overdue_penalty -= 0.1 * over
                    elif slice_name == "eMBB":
                        overdue_penalty -= 0.05 * over
                    else:
                        overdue_penalty -= 0.02 * over

        global_reward = base_reward + overdue_penalty

        return obs, global_reward, done, {"num_tasks": total_tasks}

    # ============== 工具函数 ==============
    def _row_index_from_decision(self, decision_idx: int) -> int:
        """将第k次决策映射到CSV中的行号（100ms -> 行索引+100）"""
        # 文件时间列步长是 0.001，对应 1ms；100ms -> +100 行
        return min(len(self.task_flow) - 1, decision_idx * 100)

    def _assign_users_to_bs(self, row_idx: int):
        """将每个用户分配给等效增益最大的基站（线性合并 φ 与 |h|^2）。"""
        for user in self.all_users:
            best_bs = None
            best_gain_linear = -1.0
            for bs in self.base_stations:
                phi_db = float(self.large_scale[bs].iloc[row_idx][user])
                h = float(self.small_scale[bs].iloc[row_idx][user])
                # 合并：10^(phi/10) + |h|^2
                gain_linear = (10 ** (phi_db / 10.0)) + (abs(h) ** 2)
                if gain_linear > best_gain_linear:
                    best_gain_linear = gain_linear
                    best_bs = bs
            self.user_bs_map[user] = best_bs

    def _spawn_tasks(self, row_idx: int):
        """按任务流概率生成任务，加入其所接入基站的对应切片队列。"""
        row = self.task_flow.iloc[row_idx]
        for user in self.all_users:
            prob = float(row[user])
            if np.random.random() < prob:
                slice_name = self._user_slice(user)
                data_lo, data_hi = self.slice_params[slice_name]["data_range_mbit"]
                task = {
                    "user": user,
                    "slice": slice_name,
                    "data_mbit": float(np.random.uniform(data_lo, data_hi)),
                    "queue_ms": 0.0,
                    "arrival_row": row_idx,
                }
                bs = self.user_bs_map[user]
                self.queues[bs][slice_name].append(task)

        # 不再强制注入任务，完全依赖数据概率

    def _build_observations(self, row_idx: int) -> Dict[str, np.ndarray]:
        """为每个基站构造观测：切片队列统计 + 代表性用户增益 + 归一化时间。"""
        obs = {}
        norm_time = (row_idx % 1000) / 1000.0
        for bs in self.base_stations:
            features: List[float] = []
            # 切片队列统计: [队列长度, 平均等待, 紧急比]
            for slice_name in self.slice_names:
                q = self.queues[bs][slice_name]
                q_len = len(q)
                features.append(min(q_len / 20.0, 1.0))
                if q_len > 0:
                    avg_wait = float(np.mean([t["queue_ms"] for t in q]))
                    sla = self.slice_params[slice_name]["delay_sla_ms"]
                    urgent_ratio = float(np.mean([1.0 if t["queue_ms"] >= 0.8 * sla else 0.0 for t in q]))
                    features.append(min(avg_wait / (2.0 * sla), 1.0))
                    features.append(urgent_ratio)
                else:
                    features.extend([0.0, 0.0])

            # 代表性用户（各切片第一个）增益，归一化到 [0,1]
            repr_users = [self.urllc_users[0], self.embb_users[0], self.mmtc_users[0]]
            for user in repr_users:
                bs_sel = self.user_bs_map.get(user, self.base_stations[0])
                phi_db = float(self.large_scale[bs_sel].iloc[row_idx][user])
                h = float(self.small_scale[bs_sel].iloc[row_idx][user])
                gain_db = 10.0 * np.log10((10 ** (phi_db / 10.0)) + (abs(h) ** 2) + 1e-12)
                # 范围 -100~+100 近似归一化
                features.append((gain_db + 100.0) / 200.0)

            # 时间
            features.append(norm_time)
            obs[bs] = np.array(features, dtype=np.float32)
        return obs

    def _user_slice(self, user: str) -> str:
        if user.startswith("U"):
            return "URLLC"
        if user.startswith("e"):
            return "eMBB"
        return "mMTC"

    def _calc_sinr_with_interference(self, row_idx: int, user: str, serving_bs: str,
                                      slice_name: str, rb_num: int,
                                      slice_powers_dbm: Dict[str, Dict[str, float]]) -> float:
        """
        计算指定用户在 serving_bs/切片 下的 SINR（线性）。
        信号功率：p_rx_signal = 10^((p_dbm - φ_db)/10) * |h|^2
        干扰功率：sum_{u!=serving} 10^((p_u_dbm - φ_u_db)/10) * |h_u|^2
        噪声功率：N0 = -174 + 10log10(i*b) + NF (dBm) -> 线性
        """
        # 提取信号增益
        phi_db_sig = float(self.large_scale[serving_bs].iloc[row_idx][user])
        h_sig = float(self.small_scale[serving_bs].iloc[row_idx][user])
        p_dbm_sig = float(slice_powers_dbm[serving_bs][slice_name])
        p_rx_sig_mw = (10 ** ((p_dbm_sig - phi_db_sig) / 10.0)) * (abs(h_sig) ** 2)

        # 干扰
        interf_mw = 0.0
        for bs in self.base_stations:
            if bs == serving_bs:
                continue
            p_dbm_i = float(slice_powers_dbm[bs][slice_name])
            phi_db_i = float(self.large_scale[bs].iloc[row_idx][user])
            h_i = float(self.small_scale[bs].iloc[row_idx][user])
            interf_mw += (10 ** ((p_dbm_i - phi_db_i) / 10.0)) * (abs(h_i) ** 2)

        # 噪声功率（mW）
        total_bw_hz = max(1.0, rb_num * self.bandwidth_per_rb_hz)
        noise_dbm = self.noise_spectral_density_dbm_per_hz + 10.0 * np.log10(total_bw_hz) + self.noise_figure_db
        noise_mw = 10 ** ((noise_dbm - 30.0) / 10.0)

        denom = interf_mw + noise_mw
        if denom <= 0.0:
            return 1e6
        return p_rx_sig_mw / denom

    def _calc_rate_mbps(self, sinr_linear: float, rb_num: int) -> float:
        total_bw_hz = max(1.0, rb_num * self.bandwidth_per_rb_hz)
        rate_bps = total_bw_hz * np.log2(1.0 + sinr_linear)
        return float(rate_bps / 1e6)

    def _calc_qos(self, slice_name: str, rate_mbps: float, delay_ms: float, rb_num: int) -> float:
        p = self.slice_params[slice_name]
        if delay_ms > p["delay_sla_ms"]:
            return -float(p["penalty"])
        if slice_name == "URLLC":
            # y = alpha^L
            alpha = p["alpha"]
            return float(alpha ** max(0.0, delay_ms))
        if slice_name == "eMBB":
            if rate_mbps >= p["rate_sla_mbps"]:
                return 1.0
            return max(0.0, float(rate_mbps / p["rate_sla_mbps"]))
        # mMTC: 接入比例近似：若该用户速率>=1Mbps 则计为接入成功（单用户粒度）
        return 1.0 if rate_mbps >= p["rate_sla_mbps"] else 0.0


def enumerate_rb_actions(rb_total: int, units: Dict[str, int]) -> List[Tuple[int, int, int]]:
    """生成 (URLLC, eMBB, mMTC) 的 RB 分配三元组（满足单位倍数与和<=rb_total）。"""
    urllc_unit = units["URLLC"]
    embb_unit = units["eMBB"]
    mmtc_unit = units["mMTC"]
    actions: List[Tuple[int, int, int]] = []
    for u in range(0, rb_total + 1, urllc_unit):
        for e in range(0, rb_total - u + 1, embb_unit):
            rem = rb_total - u - e
            # m 不必填满
            for m in range(0, rem + 1, mmtc_unit):
                actions.append((u, e, m))
    return actions


def enumerate_power_actions(levels: List[int]) -> List[Tuple[int, int, int]]:
    """生成功率离散等级三元组 (URLLC, eMBB, mMTC)。"""
    actions: List[Tuple[int, int, int]] = []
    for pu in levels:
        for pe in levels:
            for pm in levels:
                actions.append((pu, pe, pm))
    return actions

