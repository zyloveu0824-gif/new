# -*- coding: UTF-8 -*-
import numpy as np
import gym
from gym import spaces
import numpy as np
from .space_def import circle_space
from .space_def import onehot_space
from .space_def import sum_space
from gym.envs.registration import EnvSpec
import logging
from matplotlib import pyplot as plt
# from IPython import display

# 配置日志记录的基本设置，设置日志等级为警告
logging.basicConfig(level=logging.WARNING)

# # 创建一个新的matplotlib图表
# plt.figure()
# # 打开matplotlib的交互模式
# plt.ion()

def get_circle_plot(pos, r):
    # 定义一个函数来生成圆形的坐标点

    # 创建一个数组，表示圆形的x坐标，范围是从-r到r，步长为0.01
    x_c = np.arange(-r, r, 0.01)

    # 计算对应x坐标的上半圆的y坐标
    up_y = np.sqrt(r**2 - np.square(x_c))

    # 计算对应x坐标的下半圆的y坐标
    down_y = - up_y

    # 将x坐标调整到以pos[0]为中心
    x = x_c + pos[0]

    # 将上半圆的y坐标调整到以pos[1]为中心
    y1 = up_y + pos[1]

    # 将下半圆的y坐标调整到以pos[1]为中心
    y2 = down_y + pos[1]

    # 返回调整后的圆形坐标点
    return [x, y1, y2]


class MEC_MARL_ENV(gym.Env):
    # 定义环境的元数据
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, world, alpha=0.5, beta=0.2, aggregate_reward=False, discrete=True,
                 reset_callback=None, info_callback=None, done_callback=None):
        # 系统初始化
        self.world = world  # 移动边缘计算世界的实例
        self.obs_r = world.obs_r  # 代理的观测半径
        self.move_r = world.move_r  # 代理的移动速度
        self.collect_r = world.collect_r  # 代理的数据收集半径
        self.max_buffer_size = self.world.max_buffer_size  # 代理的最大缓冲区大小
        self.agents = self.world.agents  # 代理列表
        self.agent_num = self.world.agent_count  # 代理数量
        self.sensor_num = self.world.sensor_count  # 传感器数量
        self.sensors = self.world.sensors  # 传感器列表
        self.DS_map = self.world.DS_map  # 数据源地图
        self.map_size = self.world.map_size  # 地图尺寸
        self.DS_state = self.world.DS_state  # 数据源状态
        self.alpha = alpha  # alpha参数（可能用于奖励或决策）
        self.beta = beta  # beta参数（可能用于奖励或决策）

        self.reset_callback = reset_callback  # 环境重置的回调函数
        self.info_callback = info_callback  # 信息获取的回调函数
        self.done_callback = done_callback  # 结束条件的回调函数

        # 游戏模式
        self.aggregate_reward = aggregate_reward  # 是否共享相同的奖励
        self.discrete_flag = discrete  # 是否使用离散动作空间
        self.state = None  # 当前状态
        self.time = 0  # 当前时间
        self.images = []  # 用于渲染的图像列表

        # 配置动作空间和观测空间
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            if self.discrete_flag:
                # 如果使用离散动作空间
                act_space = spaces.Tuple((circle_space.Discrete_Circle(agent.move_r), 
                                          onehot_space.OneHot(self.max_buffer_size), 
                                          sum_space.SumOne(self.agent_num), 
                                          onehot_space.OneHot(self.max_buffer_size)))
                # 动作空间定义：移动、卸载、带宽、执行
                obs_space = spaces.Tuple((spaces.MultiDiscrete([self.map_size, self.map_size]), 
                                          spaces.Box(0, np.inf, [agent.obs_r * 2, agent.obs_r * 2, 2])))
                # 观测空间定义：位置、观测地图
                self.action_space.append(act_space)
                self.observation_space.append(obs_space)
        self.render()  # 渲染环境
    def step(self, agent_action, center_action):
        # 初始化观测、奖励、完成标志和额外信息
        obs = []
        reward = []
        done = []
        info = {'n': []}
        self.agents = self.world.agents  # 获取代理列表

        # 设置动作
        logging.info("set actions")
        for i, agent in enumerate(self.agents):
            # 为每个代理设置动作
            self._set_action(agent_action[i], center_action, agent)

        # 更新状态
        self.world.step()

        # 获取新的观测
        logging.info("agent observation")
        for agent in self.agents:
            # 为每个代理获取观测、完成标志、奖励和额外信息
            obs.append(self.get_obs(agent))  # 获取观测
            done.append(self._get_done(agent))  # 获取完成标志
            reward.append(self._get_age())  # 获取奖励（待完成）
            info['n'].append(self._get_info(agent))  # 获取额外信息

        # 更新状态
        self.state = obs

        # 计算总奖励
        reward_sum = np.sum(reward)
        logging.info("get reward")
        # 如果使用聚合奖励，所有代理共享同一奖励
        if self.aggregate_reward:
            reward = [reward_sum] * self.agent_num

        # 返回新状态、奖励、完成标志和额外信息
        return self.state, reward, done, info

    def reset(self):
        # 重置世界状态
        self.world.finished_data = []  # 清空世界中已完成数据的列表

        # 重置渲染器
        # self._reset_render()

        # 重置传感器
        for sensor in self.sensors:
            sensor.data_buffer = []  # 清空传感器的数据缓冲区
            sensor.collect_state = False  # 将传感器的收集状态设置为未收集

        # 重置代理
        for agent in self.agents:
            agent.idle = True  # 将代理的状态设置为空闲
            agent.data_buffer = {}  # 清空代理的数据缓冲区
            agent.total_data = {}  # 清空代理的总数据字典
            agent.done_data = []  # 清空代理的已完成数据列表
            agent.collecting_sensors = []  # 清空代理的收集传感器列表

    def _set_action(self, act, center_action, agent):
        # 初始化代理的移动动作为零向量
        agent.action.move = np.zeros(2)

        # 设置代理的执行动作（例如数据处理）
        agent.action.execution = act[1]

        # 设置代理的带宽，基于中心动作
        agent.action.bandwidth = center_action[agent.no]

        # 如果代理可移动且处于空闲状态
        if agent.movable and agent.idle:
            # 检查动作向量的范数是否超出移动范围
            if np.linalg.norm(act[0]) > agent.move_r:
                # 如果超出范围，将动作向量缩放到移动范围内
                act[0] = [int(act[0][0] * agent.move_r / np.linalg.norm(act[0])), 
                          int(act[0][1] * agent.move_r / np.linalg.norm(act[0]))]

            # 如果动作向量为零，随机产生一个小的移动
            if not np.count_nonzero(act[0]) and np.random.rand() > 0.5:
                mod_x = np.random.normal(loc=0, scale=1)
                mod_y = np.random.normal(loc=0, scale=1)
                mod_x = int(min(max(-1, mod_x), 1) * agent.move_r / 2)
                mod_y = int(min(max(-1, mod_y), 1) * agent.move_r / 2)
                act[0] = [mod_x, mod_y]

            # 更新代理的移动动作
            agent.action.move = np.array(act[0])

            # 计算并更新代理的新位置
            new_x = agent.position[0] + agent.action.move[0]
            new_y = agent.position[1] + agent.action.move[1]

            # 确保代理不会移动到地图外
            if new_x < 0 or new_x > self.map_size - 1:
                agent.action.move[0] = -agent.action.move[0]
            if new_y < 0 or new_y > self.map_size - 1:
                agent.action.move[1] = -agent.action.move[1]
            agent.position += agent.action.move

        # 如果代理处于卸载空闲状态
        if agent.offloading_idle:
            # 设置代理的卸载动作
            agent.action.offloading = act[2]

        # 打印代理的动作信息，用于调试
        print('agent-{} action: move{}, exe{}, off{}, band{}'.format(
            agent.no, agent.action.move, agent.action.execution, agent.action.offloading, agent.action.bandwidth))

    # get info used for benchmarking获取用于基准测试的信息

    def _get_info(self, agent):
        # 如果定义了获取信息的回调函数，使用它来获取信息
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def get_obs(self, agent):
        # 初始化代理的观测空间
        obs = np.zeros([agent.obs_r * 2 + 1, agent.obs_r * 2 + 1, 2])

        # 计算观测区域的左上角点
        lu = [max(0, agent.position[0] - agent.obs_r),
              min(self.map_size, agent.position[1] + agent.obs_r + 1)]

        # 计算观测区域的右下角点
        rd = [min(self.map_size, agent.position[0] + agent.obs_r + 1),
              max(0, agent.position[1] - agent.obs_r)]

        # 计算观测地图中的左上角坐标
        ob_lu = [agent.obs_r - agent.position[0] + lu[0],
                 agent.obs_r - agent.position[1] + lu[1]]

        # 计算观测地图中的右下角坐标
        ob_rd = [agent.obs_r + rd[0] - agent.position[0],
                 agent.obs_r + rd[1] - agent.position[1]]

        # 遍历观测区域并从数据源状态地图中复制相应区域的数据
        for i in range(ob_rd[1], ob_lu[1]):
            map_i = rd[1] + i - ob_rd[1]
            obs[i][ob_lu[0]:ob_rd[0]] = self.DS_state[map_i][lu[0]:rd[0]]

        # 将观测结果存储到代理的观测属性中
        agent.obs = obs
        return obs

    def get_statemap(self):
        # 初始化传感器和代理的状态地图
        sensor_map = np.ones([self.map_size, self.map_size, 2])
        agent_map = np.ones([self.map_size, self.map_size, 2])

        # 遍历所有传感器，记录它们的位置和数据
        for sensor in self.sensors:
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][0] = sum([i[0] for i in sensor.data_buffer])
            sensor_map[int(sensor.position[1])][int(sensor.position[0])][1] = sum([i[1] for i in sensor.data_buffer]) / max(len(sensor.data_buffer), 1)

        # 遍历所有代理，记录它们的位置和已完成数据
        for agent in self.agents:
            agent_map[int(agent.position[1])][int(agent.position[0])][0] = sum([i[0] for i in agent.done_data])
            agent_map[int(agent.position[1])][int(agent.position[0])][1] = sum([i[1] for i in agent.done_data]) / max(len(agent.done_data), 1)

        # 返回传感器和代理的状态地图
        return sensor_map, agent_map

    def get_center_state(self):
        # 初始化代理的缓冲区列表和位置列表
        buffer_list = np.zeros([self.agent_num, 2, self.max_buffer_size])
        pos_list = np.zeros([self.agent_num, 2])

        # 遍历所有代理，记录它们的位置和已完成数据
        for i, agent in enumerate(self.agents):
            pos_list[i] = agent.position
            for j, d in enumerate(agent.done_data):
                buffer_list[i][0][j] = d[0]
                buffer_list[i][1][j] = d[1]

        # 返回代理的缓冲区状态和位置
        return buffer_list, pos_list

    def get_buffer_state(self):
        # 初始化代理的执行和已完成数据数量列表
        exe = []
        done = []

        # 遍历所有代理，记录它们的执行和已完成数据数量
        for agent in self.agents:
            exe.append(len(agent.total_data))
            done.append(len(agent.done_data))

        # 返回代理的执行和已完成数据数量
        return exe, done

    def _get_done(self, agent):
        # 检查是否定义了完成条件的回调函数
        if self.done_callback is None:
            return 0
        return self.done_callback(agent, self.world)

    def _get_age(self):
        # 计算并返回传感器的平均年龄
        return np.mean(list(self.world.sensor_age.values()))

    def _get_reward(self):
        # 计算并返回代理的奖励
        # 目前简单地返回传感器的平均年龄作为奖励
        return np.mean(list(self.world.sensor_age.values()))
        # 备注：以下是更复杂的奖励计算方法，当前已注释
        # state_reward = sum(sum(self.DS_state)) / self.sensor_num
        # done_reward = [[i[0], i[1]] for i in self.world.finished_data]
        # if not done_reward:
        #     done_reward = np.array([0, 0])
        # else:
        #     done_reward = np.average(np.array(done_reward), axis=0)
        # buffer_reward = 0
        # for agent in self.agents:
        #     if agent.done_data:
        #         buffer_reward += np.mean([d[1] for d in agent.done_data])
        # buffer_reward = buffer_reward / self.agent_num
        # return self.alpha * done_reward[1] + self.beta * (state_reward[1] + self.sensor_num - self.map_size * self.map_size) + (1 - self.alpha - self.beta) * buffer_reward

    def render(self, name=None, epoch=None, save=False):
        # 渲染环境的当前状态
        plt.figure()
        # 绘制传感器的位置
        plt.scatter(self.world.sensor_pos[0], self.world.sensor_pos[1], c='cornflowerblue', alpha=0.9)

        # 遍历并绘制所有代理的位置
        for agent in self.world.agents:
            plt.scatter(agent.position[0], agent.position[1], c='orangered', alpha=0.9)
            plt.annotate(agent.no + 1, xy=(agent.position[0], agent.position[1]), xytext=(agent.position[0] + 0.1, agent.position[1] + 0.1))
            # 绘制代理的观测半径和收集半径
            obs_plot = get_circle_plot(agent.position, self.obs_r)
            collect_plot = get_circle_plot(agent.position, self.collect_r)
            plt.fill_between(obs_plot[0], obs_plot[1], obs_plot[2], where=obs_plot[1] > obs_plot[2], color='darkorange', alpha=0.02)
            plt.fill_between(collect_plot[0], collect_plot[1], collect_plot[2], where=collect_plot[1] > collect_plot[2], color='darkorange', alpha=0.05)

        # 设置图表属性
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(['Sensors', 'Edge Agents'])
        plt.axis('square')
        plt.xlim([0, self.map_size])
        plt.ylim([0, self.map_size])
        plt.title('all entity position(epoch%s)' % epoch)

        # 根据参数决定是显示还是保存图表
        if not save:
            plt.show()
            return
        plt.savefig('%s/%s.png' % (name, epoch))
        plt.close()

    def close(self):
        # 关闭环境
        return None
