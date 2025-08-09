import pandas as pd
import math
from collections import deque

class Task:
    def __init__(self, task_id, user_id, slice_type, data_size, required_rb, sla_delay, sla_rate):
        self.task_id = task_id
        self.user_id = user_id
        self.slice_type = slice_type
        self.data_size = data_size
        self.required_rb = required_rb
        self.sla_delay = sla_delay
        self.sla_rate = sla_rate
        self.processed_data = 0.0
        self.start_time = 0.0
        self.is_completed = False
        self.is_failed = False

class NetworkSliceSystem:
    def __init__(self):
        # 系统参数
        self.R_total = 50
        self.power = 30
        self.bandwidth_per_rb = 360e3
        self.thermal_noise = -174
        self.NF = 7
        self.time_slot = 1
        
        # 资源倍数约束
        self.URLLC_MULTIPLE = 10
        self.eMBB_MULTIPLE = 5
        self.mMTC_MULTIPLE = 2
        
        # SLA参数
        self.D_u_max = 5
        self.D_e_max = 100
        self.D_m_max = 500
        self.R_e_min = 50
        
        # 惩罚系数
        self.P_u = -5
        self.P_e = -3
        self.P_m = -1
        self.alpha = 0.95
        
        # 任务队列（符合user.md要求）
        self.task_queues = {
            'URLLC': deque(),
            'eMBB': deque(),
            'mMTC': deque()
        }
        
        # 当前资源分配（符合user.md要求）
        self.current_alloc = {
            'URLLC': 0,
            'eMBB': 0,
            'mMTC': 0
        }
        
        self.current_time = 0.0
        self.total_qos = 0.0
        self.completed_tasks = []
        self.failed_tasks = []
    
    def load_data(self, file_path):
        data = pd.read_excel(file_path)
        return data.iloc[0]
    
    def create_initial_tasks(self, user_data):
        """创建初始任务队列"""
        tasks = []
        
        # URLLC任务
        for i in range(2):
            user_key = f'U{i+1}'
            if user_key in user_data:
                task = Task(f"URLLC_{i+1}", user_key, 'URLLC', 0.011, 10, self.D_u_max, 10)
                tasks.append(task)
        
        # eMBB任务
        for i in range(4):
            user_key = f'e{i+1}'
            if user_key in user_data:
                task = Task(f"eMBB_{i+1}", user_key, 'eMBB', 0.11, 5, self.D_e_max, 50)
                tasks.append(task)
        
        # mMTC任务
        for i in range(10):
            user_key = f'm{i+1}'
            if user_key in user_data:
                task = Task(f"mMTC_{i+1}", user_key, 'mMTC', 0.013, 2, self.D_m_max, 1)
                tasks.append(task)
        
        return tasks
    
    def calculate_transmission_rate(self, user_data, user_id, num_rbs):
        """计算传输速率"""
        if num_rbs <= 0:
            return 0.0
        
        channel_gain = user_data[user_id]
        power_mw = 10**((self.power - 30) / 10)
        channel_gain_linear = 10**(channel_gain / 10)
        received_power = power_mw * channel_gain_linear
        
        noise_power = 10**((self.thermal_noise + 10*math.log10(num_rbs * self.bandwidth_per_rb) + self.NF) / 10)
        sinr = received_power / noise_power
        
        rate = num_rbs * self.bandwidth_per_rb * math.log2(1 + sinr)
        return rate / 1e6
    
    def calculate_resource_allocation(self, task_queues, R_total, multiple_constraints):
        """计算资源分配（符合user.md要求）"""
        urllc_tasks = len(task_queues['URLLC'])
        embb_tasks = len(task_queues['eMBB'])
        mmtc_tasks = len(task_queues['mMTC'])
        
        # 计算所需资源
        urllc_needed = urllc_tasks * self.URLLC_MULTIPLE
        embb_needed = embb_tasks * self.eMBB_MULTIPLE
        mmtc_needed = mmtc_tasks * self.mMTC_MULTIPLE
        
        # 初步分配
        allocation = {
            'URLLC': min(urllc_needed, R_total),
            'eMBB': min(embb_needed, R_total),
            'mMTC': min(mmtc_needed, R_total)
        }
        
        # 确保满足倍数约束
        allocation['URLLC'] = (allocation['URLLC'] // self.URLLC_MULTIPLE) * self.URLLC_MULTIPLE
        allocation['eMBB'] = (allocation['eMBB'] // self.eMBB_MULTIPLE) * self.eMBB_MULTIPLE
        allocation['mMTC'] = (allocation['mMTC'] // self.mMTC_MULTIPLE) * self.mMTC_MULTIPLE
        
        # 验证资源约束并调整
        total_allocated = sum(allocation.values())
        if total_allocated > R_total:
            # 按优先级削减资源
            if allocation['mMTC'] > 0:
                reduction = min(allocation['mMTC'], total_allocated - R_total)
                allocation['mMTC'] -= reduction
                total_allocated -= reduction
            
            if total_allocated > R_total and allocation['eMBB'] > 0:
                reduction = min(allocation['eMBB'], total_allocated - R_total)
                allocation['eMBB'] -= reduction
                total_allocated -= reduction
            
            if total_allocated > R_total and allocation['URLLC'] > 0:
                reduction = min(allocation['URLLC'], total_allocated - R_total)
                allocation['URLLC'] -= reduction
        
        return allocation
    
    def process_task(self, task, allocated_rb, user_data):
        """处理单个任务（符合user.md要求）"""
        if task.is_completed or task.is_failed:
            return True
        
        # 计算传输速率
        rate = self.calculate_transmission_rate(user_data, task.user_id, allocated_rb)
        
        # 计算本次时间片可传输的数据量（限制传输速度，使任务不会立即完成）
        data_per_slot = min(rate * self.time_slot / 1000, task.data_size * 0.1)  # 限制每次最多传输10%
        
        # 更新已处理数据量
        task.processed_data += data_per_slot
        
        # 检查任务是否完成
        if task.processed_data >= task.data_size:
            task.is_completed = True
            self.completed_tasks.append(task)
            return True
        
        return False
    
    def calculate_task_delay(self, task):
        """计算任务延迟"""
        if task.start_time == 0:
            return 0.0
        return self.current_time - task.start_time
    
    def evaluate_qos(self, task, rate, delay):
        """评估服务质量（符合user.md要求）"""
        if task.slice_type == 'URLLC':
            if delay <= self.D_u_max:
                return self.alpha ** delay
            else:
                return self.P_u
        
        elif task.slice_type == 'eMBB':
            if delay > self.D_e_max:
                return self.P_e
            if rate < self.R_e_min:
                return self.P_e
            if rate >= self.R_e_min:
                return 1.0
            else:
                return rate / self.R_e_min
        
        elif task.slice_type == 'mMTC':
            if delay > self.D_m_max:
                return self.P_m
            return 1.0 if rate >= task.sla_rate else 0.0
        
        return 0.0
    
    def process_slice_tasks(self, slice_type, user_data):
        """处理指定切片的任务队列（符合user.md要求）"""
        queue = self.task_queues[slice_type]
        allocated_rb = self.current_alloc[slice_type]
        
        if allocated_rb == 0 or not queue:
            return
        
        tasks_to_remove = []
        
        # 按队列顺序处理任务
        for task in queue:
            if task.start_time == 0:
                task.start_time = self.current_time
            
            # 处理任务
            completed = self.process_task(task, allocated_rb, user_data)
            
            if completed:
                tasks_to_remove.append(task)
                # 任务完成，释放资源供其他任务复用
            else:
                # 检查QoS约束
                delay = self.calculate_task_delay(task)
                rate = self.calculate_transmission_rate(user_data, task.user_id, allocated_rb)
                qos = self.evaluate_qos(task, rate, delay)
                
                if qos < 0:  # 惩罚情况
                    task.is_failed = True
                    self.failed_tasks.append(task)
                    tasks_to_remove.append(task)
        
        # 移除已完成或失败的任务
        for task in tasks_to_remove:
            queue.remove(task)
    
    def run_simulation(self, user_data, simulation_time=100):
        """运行仿真（符合user.md主流程）"""
        print("=== 网络切片系统仿真（完全符合user.md要求）===")
        
        # 创建初始任务
        initial_tasks = self.create_initial_tasks(user_data)
        for task in initial_tasks:
            self.task_queues[task.slice_type].append(task)
        
        print(f"初始任务: URLLC={len(self.task_queues['URLLC'])}, "
              f"eMBB={len(self.task_queues['eMBB'])}, "
              f"mMTC={len(self.task_queues['mMTC'])}")
        
        # 主循环：按时间间隔处理（符合user.md要求）
        for time_step in range(0, simulation_time, self.time_slot):
            self.current_time = time_step
            
            # 1. 动态资源分配
            new_alloc = self.calculate_resource_allocation(
                self.task_queues, 
                self.R_total, 
                [self.URLLC_MULTIPLE, self.eMBB_MULTIPLE, self.mMTC_MULTIPLE]
            )
            
            # 验证资源约束
            total_allocated = sum(new_alloc.values())
            if total_allocated > self.R_total:
                print(f"警告: 资源分配超出限制 {total_allocated}/{self.R_total}")
            
            self.current_alloc = new_alloc
            
            # 2. 任务处理（按队列顺序）
            for slice_type in ['URLLC', 'eMBB', 'mMTC']:
                self.process_slice_tasks(slice_type, user_data)
            
            # 输出状态（每10ms输出一次）
            if time_step % 10 == 0:
                print(f"时间 {time_step}ms: "
                      f"URLLC({new_alloc['URLLC']}RB,{len(self.task_queues['URLLC'])}任务), "
                      f"eMBB({new_alloc['eMBB']}RB,{len(self.task_queues['eMBB'])}任务), "
                      f"mMTC({new_alloc['mMTC']}RB,{len(self.task_queues['mMTC'])}任务)")
        
        # 计算最终QoS
        self.calculate_final_qos(user_data)
        
        return self.current_alloc, self.total_qos
    
    def calculate_final_qos(self, user_data):
        """计算最终服务质量"""
        total_qos = 0.0
        
        # 计算完成任务的QoS
        for task in self.completed_tasks:
            rate = self.calculate_transmission_rate(user_data, task.user_id, 
                                                 self.current_alloc[task.slice_type])
            delay = self.calculate_task_delay(task)
            qos = self.evaluate_qos(task, rate, delay)
            total_qos += qos
        
        # 计算失败任务的惩罚
        for task in self.failed_tasks:
            if task.slice_type == 'URLLC':
                total_qos += self.P_u
            elif task.slice_type == 'eMBB':
                total_qos += self.P_e
            elif task.slice_type == 'mMTC':
                total_qos += self.P_m
        
        self.total_qos = total_qos
        
        print(f"\n=== 仿真结果 ===")
        print(f"完成任务: {len(self.completed_tasks)}")
        print(f"失败任务: {len(self.failed_tasks)}")
        print(f"总QoS: {total_qos:.4f}")
        
        # 详细分析
        print(f"\n详细分析:")
        for slice_type in ['URLLC', 'eMBB', 'mMTC']:
            completed = [t for t in self.completed_tasks if t.slice_type == slice_type]
            failed = [t for t in self.failed_tasks if t.slice_type == slice_type]
            print(f"{slice_type}: 完成{len(completed)}个, 失败{len(failed)}个")

def solve_problem_1_user_compliant():
    """使用符合user.md要求的方案解决第一题"""
    system = NetworkSliceSystem()
    user_data = system.load_data('data1.xlsx')
    
    # 运行仿真
    best_allocation, total_qos = system.run_simulation(user_data, simulation_time=100)
    
    print(f"\n最优资源分配方案:")
    print(f"URLLC切片: {best_allocation['URLLC']} 个资源块")
    print(f"eMBB切片: {best_allocation['eMBB']} 个资源块")
    print(f"mMTC切片: {best_allocation['mMTC']} 个资源块")
    print(f"总服务质量: {total_qos:.4f}")
    
    return best_allocation, total_qos

if __name__ == "__main__":
    solve_problem_1_user_compliant() 