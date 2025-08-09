# 任务流程分析报告

## 1. user.md 中的任务流程定义

### 核心流程：
1. **任务队列管理**：每个切片类型维护独立的任务队列
2. **周期性资源分配**：按时间间隔T动态分配资源
3. **任务处理**：按队列顺序处理任务，每个时间片处理部分传输
4. **QoS约束检查**：实时检查延迟和速率约束
5. **任务完成判断**：任务完成后从队列移除并释放资源

### 关键约束：
- **资源倍数约束**：URLLC=10, eMBB=5, mMTC=2
- **SLA延迟约束**：URLLC≤5ms, eMBB≤100ms, mMTC≤500ms
- **速率约束**：eMBB≥50Mbps
- **惩罚机制**：超时或速率不达标给予负分

## 2. 当前代码的任务流程考虑

### ✅ 已实现的部分：

#### 2.1 任务数据量使用
```python
# 使用data_1/任务流.csv中的实际任务数据量
task_size = user_data['task_flow'][user_key]  # 实际任务数据量
```
- **URLLC用户**：U1=0.010305Mbit, U2=0.011077Mbit
- **eMBB用户**：e1=0.109381Mbit, e2=0.112798Mbit, e3=0.100556Mbit, e4=0.198076Mbit
- **mMTC用户**：m1-m10=0.012169-0.013739Mbit

#### 2.2 延迟计算
```python
# 使用实际任务数据量计算延迟
delay = task_size / rate * 1000  # 转换为ms
```

#### 2.3 QoS约束检查
```python
def calculate_urllc_qos(rate, delay):
    if delay <= URLLC_SLA_delay:  # 5ms
        return alpha ** delay
    else:
        return -M_URLLC  # -5

def calculate_embb_qos(rate, delay):
    if delay <= eMBB_SLA_delay:  # 100ms
        if rate >= eMBB_SLA_rate:  # 50Mbps
            return 1.0
        else:
            return rate / eMBB_SLA_rate
    else:
        return -M_eMBB  # -3
```

#### 2.4 资源倍数约束
```python
# 考虑倍数约束的分配方案
urllc_possible = [0, 10, 20, 30, 40, 50]  # 10的倍数
embb_possible = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # 5的倍数
mmtc_possible = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]  # 2的倍数
```

### ❌ 未完全实现的部分：

#### 3.1 任务队列管理
**当前代码问题**：
- 没有维护动态任务队列
- 假设所有用户同时有任务到达
- 没有考虑任务的到达时间和处理顺序

**user.md要求**：
```python
任务队列 task_queues = {
    URLLC: [任务1, 任务2, ...],  // 每个任务含ID、所需资源、时延要求等属性
    eMBB: [任务1, 任务2, ...],
    mMTC: [任务1, 任务2, ...]
}
```

#### 3.2 周期性资源分配
**当前代码问题**：
- 只进行一次静态资源分配
- 没有按时间间隔T进行动态调整

**user.md要求**：
```python
循环 按时间间隔 T 执行：
    新资源分配 new_alloc = 计算资源分配(任务队列, R_total, 倍数约束)
```

#### 3.3 任务处理流程
**当前代码问题**：
- 没有按队列顺序处理任务
- 没有"每个时间片处理部分传输"的概念
- 没有任务完成判断和资源释放

**user.md要求**：
```python
while 队列非空 且 资源未耗尽:
    任务 = 队列头部任务
    处理任务(任务, rb)  // 每个时间片处理部分传输
    if 任务完成:
        移除队列头部任务
        释放资源 rb  // 供其他任务复用
```

#### 3.4 实时QoS监控
**当前代码问题**：
- 只计算最终QoS，没有实时监控
- 没有在任务处理过程中检查约束

**user.md要求**：
```python
计算当前时延 D = 任务已用时间
检查QoS约束:
    if slice_type == URLLC 且 D > D_u_max:
        服务质量 Q_u = P_u
```

## 4. 改进建议

### 4.1 实现任务队列管理
```python
class Task:
    def __init__(self, task_id, user_id, task_size, arrival_time, slice_type):
        self.task_id = task_id
        self.user_id = user_id
        self.task_size = task_size
        self.arrival_time = arrival_time
        self.slice_type = slice_type
        self.progress = 0  # 已传输的数据量
        self.start_time = None

class TaskQueue:
    def __init__(self):
        self.urllc_queue = []
        self.embb_queue = []
        self.mmtc_queue = []
    
    def add_task(self, task):
        if task.slice_type == 'URLLC':
            self.urllc_queue.append(task)
        elif task.slice_type == 'eMBB':
            self.embb_queue.append(task)
        elif task.slice_type == 'mMTC':
            self.mmtc_queue.append(task)
```

### 4.2 实现周期性资源分配
```python
def periodic_resource_allocation(task_queues, current_time, T):
    """按时间间隔T进行周期性资源分配"""
    # 根据当前任务队列状态计算最优分配
    new_allocation = calculate_optimal_allocation(task_queues)
    return new_allocation
```

### 4.3 实现任务处理流程
```python
def process_tasks(task_queues, current_allocation, time_slot):
    """处理任务队列中的任务"""
    for slice_type in ['URLLC', 'eMBB', 'mMTC']:
        queue = task_queues.get_queue(slice_type)
        allocated_rbs = current_allocation[slice_type]
        
        while queue and allocated_rbs > 0:
            task = queue[0]  # 队列头部任务
            if process_task(task, allocated_rbs, time_slot):
                queue.pop(0)  # 任务完成，移除
            else:
                break  # 资源耗尽
```

### 4.4 实现实时QoS监控
```python
def check_qos_constraints(task, current_time):
    """实时检查QoS约束"""
    elapsed_time = current_time - task.start_time
    
    if task.slice_type == 'URLLC':
        if elapsed_time > URLLC_SLA_delay:
            return -M_URLLC
    elif task.slice_type == 'eMBB':
        if elapsed_time > eMBB_SLA_delay:
            return -M_eMBB
        if task.current_rate < eMBB_SLA_rate:
            return -M_eMBB
    elif task.slice_type == 'mMTC':
        if elapsed_time > mMTC_SLA_delay:
            return -M_mMTC
```

## 5. 总结

### 当前代码的优势：
1. ✅ 正确使用了实际任务数据量
2. ✅ 实现了SLA约束检查
3. ✅ 考虑了资源倍数约束
4. ✅ 提供了详细的性能分析

### 需要改进的方面：
1. ❌ 实现动态任务队列管理
2. ❌ 实现周期性资源分配
3. ❌ 实现任务处理流程
4. ❌ 实现实时QoS监控

### 建议：
当前代码更适合第一题的静态优化场景，但对于后续的动态场景（问题二、三、四），需要实现完整的任务流程管理机制。 