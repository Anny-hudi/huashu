// 初始化系统参数
系统总资源块 R_total
URLLC/eMBB/mMTC 资源倍数倍数 URLLC_MULTIPLE=10, eMBB_MULTIPLE=5, mMTC_MULTIPLE=2
SLA最大延迟 D_u_max, D_e_max, D_m_max
eMBB最小速率 R_e_min=50Mbps
惩罚值 P_u, P_e, P_m (均为负值)

// 初始化任务队列与资源分配
任务队列 task_queues = {
    URLLC: [任务1, 任务2, ...],  // 每个任务含ID、所需资源、时延要求等属性
    eMBB: [任务1, 任务2, ...],
    mMTC: [任务1, 任务2, ...]
}
当前资源分配 current_alloc = {
    URLLC: 0,  // 初始为0，后续动态分配
    eMBB: 0,
    mMTC: 0
}

// 主流程：周期性资源分配与任务处理
循环 按时间间隔 T 执行：
    // 1. 动态资源分配
    新资源分配 new_alloc = 计算资源分配(
        任务队列 task_queues,
        系统总资源 R_total,
        倍数约束 [URLLC_MULTIPLE, eMBB_MULTIPLE, mMTC_MULTIPLE]
    )
    
    // 验证资源约束
    if 新资源分配总和 > R_total 或 不满足倍数约束:
        调整 new_alloc 使其满足约束  // 如按优先级削减低优先级切片资源
    current_alloc = new_alloc
    
    // 2. 任务处理（按队列顺序）
    对 每个切片类型 slice_type in [URLLC, eMBB, mMTC]:
        任务队列 queue = task_queues[slice_type]
        分配资源 rb = current_alloc[slice_type]
        
        while 队列非空 且 资源未耗尽:
            任务 = 队列头部任务
            处理任务(任务, rb)  // 每个时间片处理部分传输
            if 任务完成:
                移除队列头部任务
                释放资源 rb  // 供其他任务复用
            else:
                计算当前时延 D = 任务已用时间
                检查QoS约束:
                    if slice_type == URLLC 且 D > D_u_max:
                        服务质量 Q_u = P_u
                    if slice_type == eMBB:
                        if D > D_e_max:
                            服务质量 Q_e = P_e
                        if 任务速率 < R_e_min:
                            服务质量 Q_e = P_e  // 速率不达标也触发惩罚
                    if slice_type == mMTC 且 D > D_m_max:
                        服务质量 Q_m = P_m  // 单个任务超时则整体惩罚
    
    // 3. 更新任务队列（加入新到达任务）
    task_queues = 加入新任务(task_queues, 新到达任务列表)

// 资源分配计算函数（核心逻辑）
函数 计算资源分配(任务队列, R_total, 倍数约束):
    // 输入：各切片任务队列、总资源、倍数规则
    // 输出：满足约束的资源分配方案
    初步分配 = 根据任务优先级/需求估算初始值
    修正分配 = 确保每个切片资源为对应倍数
    返回 修正分配