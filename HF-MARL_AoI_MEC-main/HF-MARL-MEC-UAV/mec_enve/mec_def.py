# -*- coding: UTF-8 -*-
import numpy as np
import random
import logging
import math
logging.basicConfig(level=logging.WARNING)

# 定义了在MEC环境中agent可以执行的动作。
class Action(object):
    def __init__(self):
        self.move = None
        self.collect = None
        self.offloading = []
        self.bandwidth = 0
        # 执行动作的列表
        self.execution = []

# 用于维护和跟踪代理在MEC环境中的状态
class AgentState(object):
    def __init__(self):
        self.position = None
        # agent的观测数据
        self.obs = None

# 表示MEC环境中的一个边缘设备
class EdgeDevice(object):
    edge_count = 0
    # 初始化
    def __init__(self, obs_r, pos, spd, collect_r, max_buffer_size, movable=True, mv_bt=0, trans_bt=0):  # pos(x,y,h)
        self.no = EdgeDevice.edge_count
        EdgeDevice.edge_count += 1
        # 观测半径
        self.obs_r = obs_r  # observe radius
        self.init_pos = pos
        self.position = pos
        self.move_r = spd
        self.collect_r = collect_r
        self.mv_battery_cost = mv_bt
        self.trans_battery_cost = trans_bt
        self.data_buffer = {}
        # 设备的最大数据缓冲区大小
        self.max_buffer_size = max_buffer_size
        self.idle = True  # collecting idle
        self.movable = movable
        self.state = AgentState()
        self.action = Action()
        self.done_data = []
        self.offloading_idle = True
        self.total_data = {}
        self.computing_rate = 2e4
        self.computing_idle = True
        self.index_dim = 2
        # 存储设备正在收集数据的传感器编号
        self.collecting_sensors = []
        self.ptr = 0.2
        self.h = 5
        self.noise = 1e-13
        self.trans_rate = 0
    # 更新边缘设备的位置
    def move(self, new_move, h):
        # 是否处于空闲状态
        if self.idle:
            self.position += new_move
            # 电池消耗
            # 计算 new_move 向量的欧几里得范数，也就是移动的距离。这个距离值然后加到 self.mv_battery_cost 上，更新设备的总移动耗电量
            self.mv_battery_cost += np.linalg.norm(new_move)
    # 获取和整理边缘设备上存储的所有待处理数据的状态
    def get_total_data(self):
        # 这个数组用于存储和表示设备上的所有数据状态
        total_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.total_data:
            # 填充数据状态
            for j, k in enumerate(list(self.total_data.keys())):
                # print(self.total_data[k])
                total_data_state[0, j] = self.total_data[k][0]
                total_data_state[1, j] = self.total_data[k][1]
        return total_data_state
    # 提取和格式化边缘设备上已处理完成的数据的状态
    def get_done_data(self):
        # 用于存储和表示已处理完成的数据的状态
        done_data_state = np.zeros([self.index_dim, self.max_buffer_size])
        if self.done_data:
            for m, k in enumerate(self.done_data):
                done_data_state[0, m] = k[0]
                done_data_state[1, m] = k[1]
        return done_data_state
    # 更新设备的数据缓冲区将数据包根据其类型或来源分类存储
    def data_update(self, pak):
        if pak[1] in self.data_buffer.keys():
            self.data_buffer[pak[1]].append(pak)
        else:
            self.data_buffer[pak[1]] = [pak]
    # 处理边缘设备上的数据并更新设备的数据状态
    def edge_exe(self, tmp_size, t=1):  # one-sum local execution
        # 没有数据需要处理
        if not self.total_data:
            return [0] * self.max_buffer_size
        # age update增加其“年龄”
        for k in self.total_data.keys():
            self.total_data[k][1] += t
        # 检查数据的长度是否达到了最大缓冲区大小
        if len(self.done_data) >= self.max_buffer_size:
            return tmp_size
        # process检查是否有数据需要处理
        if self.total_data and sum(self.action.execution):
            # 包含待处理的数据项和它们的详情。
            data2process = [[k, d] for k, d in self.total_data.items()]
            # 设备正在进行数据处理
            self.computing_idle = False
            if np.argmax(self.action.execution) >= len(data2process):
                # 重置执行动作数组
                self.action.execution = [0] * self.max_buffer_size
                # 随机选择一个数据项进行处理
                self.action.execution[np.random.randint(len(data2process))] = 1
            # 遍历待处理的数据项
            for i, data in enumerate(data2process):
                if len(self.done_data) >= self.max_buffer_size:
                    break
                # print([i, tmp_size])
                # 增加 tmp_size 中相应元素的值，计算两个值中较小的一个，并将其加到 tmp_size[i]
                # self.total_data[data2process[i][0]][0] 是当前文件的总大小，self.computing_rate * self.action.execution[i] * t 表示在给定时间内，能处理多少文件。这里的 self.computing_rate 的工作效率，self.action.execution[i] 确定他是否正在处理这个文件，t 就是时间。
                tmp_size[i] += min(self.total_data[data2process[i][0]][0], self.computing_rate * self.action.execution[i] * t)
                # 从原始数据中减去处理的数据量
                self.total_data[data2process[i][0]][0] -= self.computing_rate * self.action.execution[i] * t
                # 查数据是否已被完全处理
                if self.total_data[data2process[i][0]][0] <= 0:
                    # 如果数据已处理完毕，更新 tmp_size
                    self.total_data[data2process[i][0]][0] = tmp_size[i]
                    # 将处理完成的数据添加到 self.done_data
                    self.done_data.append(self.total_data[data2process[i][0]])
                    # 移除已处理完的数据
                    self.total_data.pop(data2process[i][0])
                    # 重置
                    tmp_size[i] = 0
        return tmp_size
    # 数据处理
    # tmp_size 是一个累积变量，记录处理过的数据量。t 是时间单位，默认为 1。
    def process(self, tmp_size, t=1):  # one-hot local execution
        # 没有数据需要处理
        if not self.total_data:
            return 0
        # age update
        for k in self.total_data.keys():
            self.total_data[k][1] += t
        if len(self.done_data) >= self.max_buffer_size:
            return 0
        # process查是否有数据需要处理
        if self.total_data and sum(self.action.execution):
            # 创建一个列表，包含待处理的数据项和它们的详情。
            data2process = [[k, d] for k, d in self.total_data.items()]
            # 检查当前激活的执行动作（即 self.action.execution 数组中值为 1 的元素）是否指向一个有效的数据项。
            if self.action.execution.index(1) >= len(data2process):
                # 取消当前的执行动作
                self.action.execution[self.action.execution.index(1)] = 0
                # 随机选择 data2process 列表中的一个新的数据项，并将对应的执行动作设置为 1
                self.action.execution[np.random.randint(len(data2process))] = 1
                # print(self.action.execution)
            # 正在进行数据处理
            self.computing_idle = False
            tmp_size += min(self.total_data[data2process[self.action.execution.index(
                1)][0]][0], self.computing_rate * t)
            # 从当前处理的数据项中减去实际处理的数据量
            # 使用 self.action.execution.index(1) 来找到激活的执行动作指向的数据项，然后减去 self.computing_rate * t，即在时间 t 内的处理量。
            self.total_data[data2process[self.action.execution.index(
                1)][0]][0] -= self.computing_rate * t
            # 检查当前正在处理的数据项是否已经完全处理完毕。它通过检查数据项的剩余大小是否小于或等于 0 来判断。这里使用 self.action.execution.index(1) 来找到当前激活的执行动作指向的数据项。
            if self.total_data[data2process[self.action.execution.index(1)][0]][0] <= 0:
                # 数据项已经被完全处理
                self.total_data[data2process[self.action.execution.index(
                    1)][0]][0] = tmp_size
                # 已完成处理的数据项添加到 self.done_data 列表中
                self.done_data.append(self.total_data[data2process[self.action.execution.index(
                    1)][0]])
                # 从 self.total_data 中移除已经处理完毕的数据项。
                self.total_data.pop(data2process[self.action.execution.index(
                    1)][0])
                tmp_size = 0
        return tmp_size

# 用于从一系列边缘设备（agents）的数据缓冲区中提取并汇总数据的最新“年龄”信息。
def agent_com(agent_list):
    # 初始化一个空字典 age_dict，用于存储不同数据项的最新年龄信息。
    age_dict = {}
    # 遍历 agent_list 中的每个边缘设备
    for u in agent_list:
        # 对于每个边缘设备，遍历其数据缓冲区 u.data_buffer 中的每个数据项。k 是数据项的键（来源），v 是对应的数据列表。
        for k, v in u.data_buffer.items():
            # 如果没有，说明这是首次遇到这个数据项类型。
            if k not in age_dict:
                # 将 v 列表中最后一个元素（即最新的数据）的年龄（v[-1][1]）赋给 age_dict[k]。这里 v[-1] 指的是数据列表中的最新项，而 [1] 表示年龄字段。
                age_dict[k] = v[-1][1]
                # 如果 age_dict 中已有 k 的条目，检查当前的数据项 v[-1] 是否比 age_dict 中存储的年龄更新。
            elif age_dict[k] > v[-1][1]:
                # 更新 age_dict[k] 为这个更新的年龄值。
                age_dict[k] = v[-1][1]
    return age_dict

# 传感器
class Sensor(object):
    sensor_cnt = 0

    def __init__(self, pos, data_rate, bandwidth, max_ds, lam=0.5, weight=1):
        self.no = Sensor.sensor_cnt
        Sensor.sensor_cnt += 1
        self.position = pos
        self.weight = weight
        self.data_rate = data_rate  # generate rate
        self.bandwidth = bandwidth
        self.trans_rate = 8e3  # to be completed
        self.data_buffer = []
        self.max_data_size = max_ds
        # 表示数据缓冲区是否有数据。这里是通过检查 data_buffer 的长度来确定的。
        self.data_state = bool(len(self.data_buffer))
        self.collect_state = False
        # 泊松分布的参数
        self.lam = lam
        # 噪声功率
        self.noise_power = 1e-13 * self.bandwidth
        # 决定是否生成新数据
        self.gen_threshold = 0.3
    # 生成传感器数据
    def data_gen(self, t=1):
        # update age数据缓冲区不为空
        if self.data_buffer:
            # 遍历数据缓冲区中的每个数据项
            for i in range(len(self.data_buffer)):
                # 增加其“年龄”
                self.data_buffer[i][1] += t
        # 计算新生成的数据量。这里使用泊松分布模拟随机数据生成过程
        new_data = self.data_rate * np.random.poisson(self.lam)
        # new_data = min(new_data, self.max_data_size)
        # 检查新生成的数据量是否超过了传感器的最大数据大小 (self.max_data_size) 或随机生成的数是否超过了生成阈值 
        if new_data >= self.max_data_size or random.random() >= self.gen_threshold:
            return
        if new_data:
            self.data_buffer.append([new_data, 0, self.no])
            self.data_state = True

# 字典，存储了不同类型环境（郊区、城市、密集城市、高层城市）下数据收集通道的参数。每种环境都有一组特定的参数值，这些参数可能涉及信号传播的特性。
# 模拟和计算无线通信环境中的数据传输效率
collecting_channel_param = {'suburban': (4.88, 0.43, 0.1, 21),
                            'urban': (9.61, 0.16, 1, 20),
                            'dense-urban': (12.08, 0.11, 1.6, 23),
                            'high-rise-urban': (27.23, 0.08, 2.3, 34)}

collecting_params = collecting_channel_param['urban']
# 信号衰减 信号传播 噪声
a = collecting_params[0]

b = collecting_params[1]
yita0 = collecting_params[2]
yita1 = collecting_params[3]
# 载波频率
carrier_f = 2.5e9

# 计算数据收集率
def collecting_rate(sensor, agent):
    # 计算传感器和代理之间的欧几里得距离
    d = np.linalg.norm(np.array(sensor.position) - np.array(agent.position))
    # 计算信号传播损耗的概率
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / d) - a)))
    # 信号损耗
    L = Pl * yita0 + yita1 * (1 - Pl)
    # 计算信噪比
    gamma = agent.ptr_col / (L * sensor.noise_power**2)
    # 香农公式
    rate = sensor.bandwidth * np.log2(1 + gamma)
    return rate

# 传感器数据收集
def data_collecting(sensors, agent, hovering_time):
    for k in agent.total_data.keys():
        agent.total_data[k][1] += 1
    if agent.idle and (len(agent.total_data.keys()) < agent.max_buffer_size):
        # obs_sensor = []
        data_properties = []
        # for k in agent.data_buffer.keys():
        #     for i, d in enumerate(agent.data_buffer[k]):
        #         agent.data_buffer[k][i][1] += 1
        for sensor in sensors:
            if not sensor.data_buffer:
                continue
            # 传感器与代理之间的距离小于或等于代理的数据收集半径 传感器当前不处于数据收集状态 代理还未收集过该传感器的数据
            if (np.linalg.norm(np.array(sensor.position) - np.array(agent.position)) <= agent.collect_r) and not(sensor.collect_state) and not(sensor.no in agent.total_data.keys()):
                # 传感器的收集状态设置为 True
                sensor.collect_state = True
                # 将传感器的编号添加到代理的收集列表中
                agent.collecting_sensors.append(sensor.no)
                # 正在进行数据收集
                agent.idle = False
                # 检查代理的 total_data 字典中的数据项数量是否已达到最大缓冲区大小
                if len(agent.total_data.keys()) >= agent.max_buffer_size:
                    continue
                # obs_sensor.append(sensor)
                # if not (sensor.no in agent.data_buffer.keys()):
                #     agent.data_buffer[sensor.no] = []
                # 计算从该传感器收集的总数据量。
                tmp_size = 0
                # trans_rate = collecting_rate(sensor, agent)
                for data in sensor.data_buffer:
                    tmp_size += data[0]
                    # data[1] += tmp_size / self.trans_rate  # age update
                if sensor.no in agent.data_buffer.keys():
                    # 如果传感器编号已存在，将收集的数据量 tmp_size 追加到对应列表中；如果不存在，为该编号创建一个新列表并添加 
                    agent.data_buffer[sensor.no].append(tmp_size)
                else:
                    
                    agent.data_buffer[sensor.no] = [tmp_size]
                # 将收集的数据量除以传感器的传输速率，计算收集所需的时间，并将该时间添加到 data_properties 列表中。
                data_properties.append(tmp_size / sensor.trans_rate)
                # 将收集的数据信息（包括数据量、数据的最新“年龄”和传感器编号）添加到代理的 total_data 字典。
                agent.total_data[sensor.no] = [tmp_size, sensor.data_buffer[-1][1], sensor.no]
                # agent.total_data[sensor.no] = [tmp_size, np.average([x[1] for x in sensor.data_buffer]), sensor.no]
                sensor.data_buffer = []

        if data_properties:
            # 设置悬停时间为 data_properties 中的最大值
            hovering_time = max(data_properties)
            # print([data_properties, hovering_time])
            return hovering_time
        else:
            return 0
    # finish collection
    elif not agent.idle:
        hovering_time -= 1
        if hovering_time <= 0:
            # 空闲
            agent.idle = True
            for no in agent.collecting_sensors:
                sensors[no].collect_state = False
            agent.collecting_sensors = []
            hovering_time = 0
        return hovering_time
    else:
        return 0


def offloading(agent, center_pos, t=1):
    # 检查是否有已处理完成的数据
    if not agent.done_data:
        # 如果没有，返回False和空字典
        return (False, {})

    # 增加每个已完成数据的年龄或等待时间
    for data in agent.done_data:
        data[1] += t

    # 检查是否有激活的卸载动作
    if sum(agent.action.offloading):
        # 检查激活的卸载动作是否指向有效的数据项
        if agent.action.offloading.index(1) >= len(agent.done_data):
            # 如果不是，重置当前卸载动作
            agent.action.offloading[agent.action.offloading.index(1)] = 0
            # 随机选择新的数据项进行卸载
            agent.action.offloading[np.random.randint(len(agent.done_data))] = 1
        
        # 设置代理的卸载状态为非空闲
        agent.offloading_idle = False
        # 计算代理到中心位置的距离
        dist = np.linalg.norm(np.array(agent.position) - np.array(center_pos))
        # 计算数据传输速率
        agent.trans_rate = trans_rate(dist, agent)  # 待完善的部分

    else:
        # 如果没有激活的卸载动作，返回False和空字典
        return False, {}

    # 从选定的数据项中减去根据传输速率计算的数据量
    agent.done_data[agent.action.offloading.index(1)][0] -= agent.trans_rate * t

    # 检查数据是否已被完全卸载
    if agent.done_data[agent.action.offloading.index(1)][0] <= 0:
        # 提取相关信息以备返回
        sensor_indx = agent.done_data[agent.action.offloading.index(1)][2]
        sensor_aoi = agent.done_data[agent.action.offloading.index(1)][1]
        sensor_data = agent.data_buffer[sensor_indx][0]
        # 从缓冲区中删除卸载的数据
        del agent.data_buffer[sensor_indx][0]
        del agent.done_data[agent.action.offloading.index(1)]
        # 设置代理为空闲状态
        agent.offloading_idle = True
        # 返回True和包含卸载数据详情的字典
        return True, {sensor_indx: [sensor_data, sensor_aoi]}

    # 如果数据未完全卸载，返回False和空字典
    return False, {}


def trans_rate(dist, agent):  
    # 计算传输带宽，单位为赫兹（Hz）
    W = 1e6 * agent.action.bandwidth

    # 计算信号传播损耗的概率
    Pl = 1 / (1 + a * np.exp(-b * (np.arctan(agent.h / dist) - a)))

    # 计算自由空间路径损耗（Free Space Path Loss，FSPL）
    fspl = (4 * np.pi * carrier_f * dist / (3e8))**2

    # 计算总的信号损耗（包括路径损耗和其他损耗）
    L = Pl * fspl * 10**(yita0 / 20) + 10**(yita1 / 20) * fspl * (1 - Pl)

    # 使用香农公式计算数据传输速率
    rate = W * np.log2(1 + agent.ptr / (L * agent.noise * W))

    # 输出传输速率信息，用于调试和分析
    print('agent-{} rate: {},{},{},{},{}'.format(agent.no, dist, agent.action.bandwidth, Pl, L, rate))

    # 返回计算的数据传输速率
    return rate



class MEC_world(object):
    def __init__(self, map_size, agent_num, sensor_num, obs_r, speed, collect_r, max_size=1, sensor_lam=0.5):
        # 初始化移动边缘计算环境
        self.agents = []  # 存储边缘设备（代理）的列表
        self.sensors = []  # 存储传感器的列表
        self.map_size = map_size  # 地图的尺寸
        self.center = (map_size / 2, map_size / 2)  # 地图中心位置
        self.sensor_count = sensor_num  # 传感器的数量
        self.agent_count = agent_num  # 代理的数量
        self.max_buffer_size = max_size  # 代理的最大数据缓冲区大小
        sensor_bandwidth = 1000  # 传感器的带宽
        max_ds = sensor_lam * 2  # 传感器的最大数据大小
        data_gen_rate = 1  # 传感器的数据生成率
        self.offloading_slice = 1  # 数据卸载的时间间隔
        self.execution_slice = 1  # 数据处理的时间间隔
        self.time = 0  # 当前时间
        self.DS_map = np.zeros([map_size, map_size])  # 数据源地图
        self.DS_state = np.ones([map_size, map_size, 2])  # 数据源状态
        self.hovering_list = [0] * self.agent_count  # 代理的悬停时间列表
        self.tmp_size_list = [0] * self.agent_count  # 代理的临时数据大小列表
        self.offloading_list = []  # 数据卸载列表
        self.finished_data = []  # 完成处理的数据列表
        self.obs_r = obs_r  # 代理的观测半径
        self.move_r = speed  # 代理的移动速度
        self.collect_r = collect_r  # 代理的数据收集半径
        self.sensor_age = {}  # 存储传感器的年龄信息
        self.grid_interval = 25  # 网格间隔
        self.grid_size = self.map_size // self.grid_interval
        # 假设 sensor_num 是传感器的数量
        self.waiting_time = [0 for _ in range(sensor_num)]  # 为每个传感器初始化等待时间
        self.sensor_direction = [random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2]) for _ in range(sensor_num)]
        # 初始化传感器位置
        self.sensor_pos = [
            [random.randint(1, (self.map_size // self.grid_interval) - 2) * self.grid_interval for _ in range(sensor_num)],
            [random.randint(1, (self.map_size // self.grid_interval) - 2) * self.grid_interval for _ in range(sensor_num)]
        ]
        # 创建传感器实例
        for i in range(30):
            self.sensors.append(
                Sensor(np.array([self.sensor_pos[0][i], self.sensor_pos[1][i]]), data_gen_rate, sensor_bandwidth, max_ds, lam=sensor_lam))
            self.sensor_age[i] = 0
            self.DS_map[self.sensor_pos[0][i], self.sensor_pos[1][i]] = 1

        # 初始化代理位置
        self.agent_pos_init = [
            random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num),
            random.sample([i for i in range(int(0.15 * self.map_size), int(0.85 * self.map_size))], agent_num)
        ]
        # 创建代理实例
        for i in range(agent_num):
            self.agents.append(
                EdgeDevice(self.obs_r, np.array([self.agent_pos_init[0][i], self.agent_pos_init[1][i]]), speed, collect_r, self.max_buffer_size))
    def move_sensor(self, sensor_index):
        if 0 <= sensor_index < 30:
            direction = self.sensor_direction[sensor_index]
            speed = 5  # 固定速度为 5

            # 计算位移
            dx = dy = 0
            if direction == 0:  # 向右
                dx = speed
            elif direction == math.pi:  # 向左
                dx = -speed
            elif direction == math.pi / 2:  # 向上
                dy = -speed
            elif direction == 3 * math.pi / 2:  # 向下
                dy = speed

            # 计算新位置
            new_x = self.sensor_pos[0][sensor_index] + dx
            new_y = self.sensor_pos[1][sensor_index] + dy

            # 确保不在边界上移动
            if not (self.grid_interval <= new_x < self.map_size - self.grid_interval and
                    self.grid_interval <= new_y < self.map_size - self.grid_interval):
                # 如果接近边界，调头
                self.sensor_direction[sensor_index] += math.pi
                self.sensor_direction[sensor_index] %= (2 * math.pi)
                return

            # 碰撞检测
            for i in range(30):
                if i != sensor_index:
                    other_x, other_y = self.sensor_pos[0][i], self.sensor_pos[1][i]
                    if abs(new_x - other_x) < 5 and abs(new_y - other_y) < 5:
                        # 如果预测到碰撞，调头
                        self.sensor_direction[sensor_index] += math.pi
                        self.sensor_direction[sensor_index] %= (2 * math.pi)
                        return

            # 只在网格点上转弯
            if new_x % self.grid_interval == 0 and new_y % self.grid_interval == 0:
                if random.random() < 0.5:  # 假设转弯概率为 0.5
                    self.sensor_direction[sensor_index] = random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2])

            # 更新位置
            self.sensor_pos[0][sensor_index], self.sensor_pos[1][sensor_index] = new_x, new_y
        else:
            print("无效的传感器索引")

    def step(self):
        # 更新传感器的年龄信息
        for k in self.sensor_age.keys():
            self.sensor_age[k] += 1

        # 数据生成和数据源状态更新
        logging.info("data generation")
        for sensor in self.sensors:
            sensor.data_gen()
            if sensor.data_buffer:
                data_size = sum(i[0] for i in sensor.data_buffer)
                # 更新数据源状态，注意坐标系转换
                self.DS_state[sensor.position[1], sensor.position[0]] = [data_size, sensor.data_buffer[0][1]]

        # 边缘设备处理、卸载和收集操作
        logging.info("edge operation")
        age_dict = {}
        for i, agent in enumerate(self.agents):
            # 边缘设备处理数据
            self.tmp_size_list[i] = agent.process(self.tmp_size_list[i])
            # 边缘设备卸载数据
            finish_flag, data_dict = offloading(agent, self.center)
            # 更新完成数据和年龄信息
            if finish_flag:
                for sensor_id, data in data_dict.items():
                    self.finished_data.append([data[0], data[1], sensor_id])
                    if sensor_id in age_dict.keys():
                        age_dict[sensor_id].append(data[1])
                    else:
                        age_dict[sensor_id] = [data[1]]
            # 边缘设备收集数据
            self.hovering_list[i] = data_collecting(self.sensors, agent, self.hovering_list[i])

        # 更新传感器的最新年龄信息
        for id in age_dict.keys():
            self.sensor_age[id] = sorted(age_dict[id])[0]
        print('hovering:{}'.format(self.hovering_list))
        for i, sensor in enumerate(self.sensors):
            self.move_sensor(i)

