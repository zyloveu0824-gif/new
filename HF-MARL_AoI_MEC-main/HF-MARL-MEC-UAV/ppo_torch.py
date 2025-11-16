import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import scipy.io as sio
def discrete_circle_sample_count(n):
    count = 0
    move_dict = {}
    for x in range(-n, n + 1):
        y_l = int(np.floor(np.sqrt(n**2 - x**2)))
        for y in range(-y_l, y_l + 1):
            move_dict[count] = np.array([y, x])
            count += 1
    return (count), move_dict
class SensorActor(nn.Module):
    def __init__(self, input_dim_list, cnn_kernel_size):
        super(SensorActor, self).__init__()
        self.input_dim = input_dim_list[0]  # Assuming the input dimensions are provided correctly
        self.transpose = nn.Conv1d(in_channels=self.input_dim[1], out_channels=1, kernel_size=1)  # Emulating transpose and dense operation
        self.computing_dev = nn.Linear(self.input_dim[2], 1)  # The output of transpose is expected to have the last dimension size `input_dim[2]`

    def forward(self, x):
        # Transpose operation by using a 1x1 Conv1d to simulate the transpose and reduction
        x = self.transpose(x.permute(0, 2, 1))  # Permuting the dimensions to fit Conv1d requirements
        x = F.relu(x)
        x = torch.squeeze(x, -1)  # Squeeze the singleton dimension
        computing_dev = torch.sigmoid(self.computing_dev(x))
        return computing_dev

# Example of using the model
# Assuming the input_dim_list and cnn_kernel_size are known
# sensor_actor_model = SensorActor(input_dim_list=[(None, 2, max_buffer_size)], cnn_kernel_size=some_kernel_size)
# output = sensor_actor_model(some_input_tensor)

class AgentActor(nn.Module):
    def __init__(self, input_dim_list, cnn_kernel_size, move_r):
        super(AgentActor, self).__init__()
        self.cnn_map = nn.Sequential(
            nn.Conv2d(input_dim_list[0][2], input_dim_list[0][2], kernel_size=cnn_kernel_size, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=int(input_dim_list[0][0] / (2 * move_r + 1))),
            nn.AlphaDropout(0.2)
        )
        self.dense_map = nn.Linear(input_dim_list[0][2], 1)  # Assuming correct flattening later
        self.total_mlp = nn.Linear(input_dim_list[1][2], input_dim_list[1][1])
        self.done_mlp = nn.Linear(input_dim_list[2][2], input_dim_list[2][1])
        self.bandwidth_dense = nn.Linear(1, 1)

    def forward(self, state_map, total_buffer, done_buffer, bandwidth):
        # State map path
        x = self.cnn_map(state_map)
        x = x.view(x.size(0), -1)  # Flatten
        move_out = F.relu(self.dense_map(x))

        # Total buffer path
        total_buffer = total_buffer.permute(0, 2, 1)
        total_mlp_out = F.relu(self.total_mlp(total_buffer))
        total_mlp_out = total_mlp_out.permute(0, 2, 1)
        exe_op = F.softmax(total_mlp_out, dim=1)

        # Done buffer path
        done_buffer = done_buffer.permute(0, 2, 1)
        done_mlp_out = F.relu(self.done_mlp(done_buffer))
        done_mlp_out = done_mlp_out.permute(0, 2, 1)

        # Bandwidth processing
        bandwidth = bandwidth.unsqueeze(-1)
        bandwidth_out = F.relu(self.bandwidth_dense(bandwidth))

        # Combine done buffer output with bandwidth
        off_op = F.softmax(torch.cat((done_mlp_out, bandwidth_out), dim=-1), dim=1)

        # Combine operation distributions
        op_dist = torch.cat((exe_op, off_op), dim=1)

        return move_out, op_dist

# Example of using the model
# Assuming the input dimensions and cnn_kernel_size are known
# agent_actor_model = AgentActor(input_dim_list=[(height, width, channels), (max_buffer_size, 2), (max_buffer_size, 2), (bandwidth_dim)], cnn_kernel_size=some_kernel_size, move_r=some_radius)
# move_output, op_distribution = agent_actor_model(state_map_tensor, total_buffer_tensor, done_buffer_tensor, bandwidth_tensor)

class CenterActor(nn.Module):
    def __init__(self, input_dim_list, cnn_kernel_size):
        super(CenterActor, self).__init__()
        self.buffer_dense = nn.Linear(input_dim_list[0][-1], 1)
        self.pos_dense = nn.Linear(input_dim_list[1][-1], 2)
        self.final_dense = nn.Linear(3, 1)  # Assuming the concatenation of buffer and pos outputs a vector of size 3

    def forward(self, done_buffer_list, pos_list):
        # Processing done buffer list
        buffer_state = F.relu(self.buffer_dense(done_buffer_list))
        buffer_state = torch.squeeze(buffer_state, -1)

        # Processing pos list
        pos = F.relu(self.pos_dense(pos_list))

        # Concatenating buffer state and pos
        combined = torch.cat([buffer_state, pos], dim=-1)
        bandwidth_out = F.relu(self.final_dense(combined))
        bandwidth_out = torch.squeeze(bandwidth_out, -1)

        # Softmax layer
        bandwidth_out = F.softmax(bandwidth_out, dim=-1)

        return bandwidth_out
    
class SensorCritic(nn.Module):
    def __init__(self, input_dim_list, cnn_kernel_size):
        super(SensorCritic, self).__init__()
        # Assuming the number of channels after roi pooling is the last dimension in input_dim_list[0]
        self.sensor_dense = nn.Linear(input_dim_list[0][2], 1)  # Adjust according to the actual input dimension
        self.final_dense = nn.Linear(input_dim_list[0][2] + input_dim_list[1][0], 1)  # Adjust the dimensions accordingly

    def forward(self, sensor_buffer, computing_dev):
        # Process the sensor buffer
        sensor_buffer = sensor_buffer.permute(0, 2, 1)  # Transposing the last two dimensions
        sensor_mlp = F.relu(self.sensor_dense(sensor_buffer))
        sensor_mlp = torch.squeeze(sensor_mlp, -1)  # Remove the singleton dimension

        # Concatenate the output of sensor MLP with computing device input
        r_out = torch.cat([sensor_mlp, computing_dev], dim=-1)
        r_out = F.relu(self.final_dense(r_out))

        return r_out

# Example of using the model
# Assuming the input dimensions are known
# sensor_critic_model = SensorCritic(input_dim_list=[(None, 2, 5), (None, 1)], cnn_kernel_size=some_kernel_size)
# reward_output = sensor_critic_model(sensor_buffer_tensor, computing_dev_tensor)

class AgentCritic(nn.Module):
    def __init__(self, input_dim_list, cnn_kernel_size):
        super(AgentCritic, self).__init__()
        # Initial Dense layer before CNN for state map
        self.initial_dense_map = nn.Linear(input_dim_list[0][-1], 1)  # Assuming the last dimension represents features
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=cnn_kernel_size, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=cnn_kernel_size * 2),
            nn.AlphaDropout(0.2),
            nn.Flatten(),
            nn.Linear(input_dim_list[0][0] * input_dim_list[0][1] // (cnn_kernel_size ** 2 * 4), 2),  # Calculate size after pooling
            nn.ReLU()
        )
        # Dense layers for other inputs
        self.band_mlp = nn.Linear(input_dim_list[5][-1], 1)
        self.total_mlp = nn.Linear(input_dim_list[1][2], 2)
        self.done_mlp = nn.Linear(input_dim_list[2][2], 2)
        self.move_mlp = nn.Linear(input_dim_list[3][0] * input_dim_list[3][1], 1)  # Assuming move is already flattened
        self.onehot_mlp = nn.Linear(input_dim_list[4][-1], 1)

        # Final dense layer to output reward
        self.final_dense = nn.Linear(8, 1)  # Sum of all outputs before concatenation

    def forward(self, state_map, total_buffer, done_buffer, move, onehot_op, bandwidth):
        # Processing state_map with CNN
        state_map = F.relu(self.initial_dense_map(state_map))
        state_map = state_map.unsqueeze(1)  # Add channel dimension for CNN
        map_out = self.map_cnn(state_map)

        # MLP layers for other inputs
        band_out = F.relu(self.band_mlp(bandwidth))
        total_buffer = total_buffer.permute(0, 2, 1)
        total_out = F.relu(self.total_mlp(total_buffer))
        done_buffer = done_buffer.permute(0, 2, 1)
        done_out = F.relu(self.done_mlp(done_buffer))
        move_out = F.relu(self.move_mlp(move.view(move.size(0), -1)))  # Flatten move
        onehot_out = F.relu(self.onehot_mlp(onehot_op.squeeze(-1)))

        # Concatenate all outputs
        all_out = torch.cat([map_out, band_out, total_out, done_out, move_out, onehot_out], dim=-1)
        reward_out = F.relu(self.final_dense(all_out))

        return reward_out

# Example of using the model
# Assuming the input dimensions and cnn_kernel_size are known
# agent_critic_model = AgentCritic(input_dim_list=[(height, width, channels), (buffer_size, 2), (buffer_size, 2), (move_size, move_size), (op_size,), (band_size,)], cnn_kernel_size=some_kernel_size)
# reward_output = agent_critic_model(state_map_tensor, total_buffer_tensor, done_buffer_tensor, move_tensor, onehot_op_tensor, bandwidth_tensor)

class CenterCritic(nn.Module):
    def __init__(self, input_dim_list, cnn_kernel_size):
        super(CenterCritic, self).__init__()
        self.buffer_dense1 = nn.Linear(input_dim_list[0][-1], 1)
        self.buffer_dense2 = nn.Linear(1, 1)
        self.pos_dense = nn.Linear(input_dim_list[1][-1], 1)

    def forward(self, done_buffer_list, pos_list, bandwidth_vec):
        # Process buffer list
        buffer_state = F.relu(self.buffer_dense1(done_buffer_list))
        buffer_state = torch.squeeze(buffer_state, -1)
        buffer_state = F.relu(self.buffer_dense2(buffer_state))
        buffer_state = torch.squeeze(buffer_state, -1)

        # Process position list
        pos = F.relu(self.pos_dense(pos_list))
        pos = torch.squeeze(pos, -1)

        # Concatenate and process outputs
        r_out = torch.cat([buffer_state, pos, bandwidth_vec], dim=-1)
        r_out = F.relu(self.final_dense(r_out))

        return r_out

# Example of using the model
# center_critic_model = CenterCritic(input_dim_list=[(None, buffer_size), (None, pos_size), (None, bandwidth_size)], cnn_kernel_size=some_kernel_size)
# output = center_critic_model(done_buffer_tensor, pos_list_tensor, bandwidth_vec_tensor)

def update_target_net(model, target_model, tau=0.8):
    with torch.no_grad():
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1 - tau))

# Example of updating the target network
# update_target_net(source_model, target_model, tau=0.8)

def circle_argmax(move_dist, move_r):
    # Squeezing the move_dist tensor to remove the singleton dimension
    move_dist_np = move_dist.squeeze(-1).numpy()  # Convert to numpy array for processing
    max_val = np.max(move_dist_np)  # Find the maximum value
    max_pos = np.argwhere(move_dist_np == max_val)  # Find indices where the maximum value occurs
    
    # Calculate the Euclidean distance from the center (move_r, move_r)
    pos_dist = np.linalg.norm(max_pos - np.array([move_r, move_r]), axis=1)
    
    # Return the position of the minimum distance among the maximum probability positions
    return max_pos[np.argmin(pos_dist)]

class MAACAgent:
    def __init__(self, env, tau, gamma, lr_sa, lr_sc, lr_aa, lr_ac, lr_ca, lr_cc, batch, epsilon=0.2):
        self.env = env
        self.sensors = self.env.sensors
        self.sensor_num = self.env.sensor_num
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.epsilon = epsilon

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch
        self.sensor_memory = {}  # 所有sensor的经验缓冲区 sensor_memory字典
        self.agent_memory = {}  # 所有UAV的经验缓冲区 agent_memory字典
        self.softmax_memory = {}  # 没用到
        self.center_memory = []  # center的经验缓冲区 center_memory字典
        self.sample_prop = 1 / 4  # replay 
        self.summaries = {}

        # Initialize networks
        self.sensor_actors = [SensorActor() for _ in range(self.sensor_num)]
        self.agent_actors = [AgentActor() for _ in range(self.agent_num)]
        self.center_actor = CenterActor()
        
        self.sensor_critics = [SensorCritic() for _ in range(self.sensor_num)]
        self.agent_critics = [AgentCritic() for _ in range(self.agent_num)]
        self.center_critic = CenterCritic()

        # Initialize target networks
        self.target_sensor_actors = [SensorActor() for _ in range(self.sensor_num)]
        self.target_agent_actors = [AgentActor() for _ in range(self.agent_num)]
        self.target_center_actor = CenterActor()
        
        self.target_sensor_critics = [SensorCritic() for _ in range(self.sensor_num)]
        self.target_agent_critics = [AgentCritic() for _ in range(self.agent_num)]
        self.target_center_critic = CenterCritic()

        # Initialize optimizers
        self.sensor_actor_opt = [optim.Adam(actor.parameters(), lr=lr_sa) for actor in self.sensor_actors]
        self.sensor_critic_opt = [optim.Adam(critic.parameters(), lr=lr_sc) for critic in self.sensor_critics]
        self.agent_actor_opt = [optim.Adam(actor.parameters(), lr=lr_aa) for actor in self.agent_actors]
        self.agent_critic_opt = [optim.Adam(critic.parameters(), lr=lr_ac) for critic in self.agent_critics]
        self.center_actor_opt = optim.Adam(self.center_actor.parameters(), lr=lr_ca)
        self.center_critic_opt = optim.Adam(self.center_critic.parameters(), lr=lr_cc)

        # Update target networks initially
        self.update_target_networks()

    def update_target_networks(self):
        for target, source in zip(self.target_sensor_actors, self.sensor_actors):
            self.soft_update(target, source, self.tau)
        for target, source in zip(self.target_agent_actors, self.agent_actors):
            self.soft_update(target, source, self.tau)
        self.soft_update(self.target_center_actor, self.center_actor, self.tau)
        for target, source in zip(self.target_sensor_critics, self.sensor_critics):
            self.soft_update(target, source, self.tau)
        for target, source in zip(self.target_agent_critics, self.agent_critics):
            self.soft_update(target, source, self.tau)
        self.soft_update(self.target_center_critic, self.center_critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def actor_act(self, epoch):
        tmp = random.random()  # 生成一个0到1的随机浮点数
        if tmp >= self.epsilon and epoch >= 16:  # 若①(1-epsilon)概率；②epoch大于16
            # *************************** Sensor 动作处理 ***************************
            sensor_state_list = []  # 存储sensor状态的列表
            sensor_softmax_list = []  # 存储sensor动作的softmax形式列表
            sensor_act_list = []  # 存储sensor动作列表

            for i, sensor in enumerate(self.sensors):
                # 从MEC环境中获取sensor的数据并增加一个批次维度
                sensor_data_state = torch.tensor(sensor.get_sensor_data(), dtype=torch.float32).unsqueeze(0)
                sensor_state_list.append([sensor_data_state])  # 添加sensor状态到列表

                # 使用sensor的actor网络进行预测
                sensor_action_output = self.sensor_actors[i](sensor_data_state)
                sensor_comdev = sensor_action_output.squeeze(0)  # 假设输出直接可用

                # 转换网络输出为softmax形式，添加一个维度模拟tf.expand_dims操作
                sensor_com_softmax = torch.nn.functional.softmax(sensor_comdev, dim=-1).unsqueeze(0)

                sensor_act_list.append([sensor_comdev])
                sensor_softmax_list.append([sensor_com_softmax])  # 添加softmax处理后的结果到列表

             # *************************** Agent 动作处理 ***************************
            agent_act_list = []  # Agent动作列表
            softmax_list = []  # Agent动作的softmax形式列表
            cur_state_list = []  # 当前状态列表
            band_vec = torch.zeros(self.agent_num)  # 初始化UAV动作中的带宽列表

            for i, agent in enumerate(self.agents):
                # 从环境中获取agent的观察，并为每个观察添加一个批次维度
                state_map = torch.tensor(self.env.get_obs(agent), dtype=torch.float32).unsqueeze(0)
                total_data_state = torch.tensor(agent.get_total_data(), dtype=torch.float32).unsqueeze(0)
                done_data_state = torch.tensor(agent.get_done_data(), dtype=torch.float32).unsqueeze(0)
                band = torch.tensor([agent.action.bandwidth], dtype=torch.float32).unsqueeze(0)
                band_vec[i] = agent.action.bandwidth

                # 将组合状态送入agent_actor网络进行预测
                action_output = self.agent_actors[i](state_map, total_data_state, done_data_state, band)
                move_dist = action_output[0].squeeze(0)
                op_dist = action_output[1].squeeze(0)

                # 保存调试文件（等价于TensorFlow中的sio.savemat）
                sio.savemat('debug.mat', {'state': self.env.get_obs(agent).cpu().numpy(), 'move': move_dist.cpu().numpy()})

                # 根据动作概率选择最优动作
                move_ori = self.circle_argmax(move_dist.cpu().numpy(), self.env.move_r)
                move = [move_ori[1] - self.env.move_r, move_ori[0] - self.env.move_r]

                # 记录并更新UAV的移动距离
                agent.t_distance = np.linalg.norm(np.array(move))
                agent.e_distance += agent.t_distance
                agent.total_distance += agent.t_distance
                self.UAVs_total_distance += agent.t_distance
                self.summaries['agent%s-e_distance' % i] = agent.e_distance
                self.summaries['agent%s-total_distance' % i] = agent.total_distance

                # 处理UAV的计算和卸载动作
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.argmax(op_dist[0].cpu().numpy())] = 1
                offloading[np.argmax(op_dist[1].cpu().numpy())] = 1

                # 创建移动和操作的概率分布
                move_softmax = torch.zeros_like(move_dist)
                op_softmax = torch.zeros_like(op_dist)
                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(op_dist[0])] = 1
                op_softmax[1][np.argmax(op_dist[1])] = 1

                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax.unsqueeze(0), op_softmax.unsqueeze(0)])

            # *************************** Center 行动处理 ***************************
            # 从MEC环境中获取center_actor的输入，并打包成“云中心状态列表”
            done_buffer_list, pos_list = self.env.get_center_state()  # 获取每个UAV计算完成的数据和UAV位置坐标
            done_buffer_list = torch.tensor(done_buffer_list, dtype=torch.float32).unsqueeze(0)  # 转换为张量并增加批次维度
            pos_list = torch.tensor(pos_list, dtype=torch.float32).unsqueeze(0)
            band_vec = band_vec.unsqueeze(0)  # 带宽向量增加批次维度

            # 将云中心状态列表送入center_actor网络进行预测，获得神经网络的输出（bandwidth_vec）
            new_bandvec = self.center_actor(done_buffer_list, pos_list, band_vec)  # 假设center_actor是一个PyTorch模型

            # *************************** 环境交互 ***************************
            # 将agent和center的actor网络中输出的UAV和云中心的动作 输入到MEC环境中，进行环境状态更新
            new_state_maps, new_rewards, average_age, fairness_index, done, info = self.env.step(agent_act_list, new_bandvec.squeeze(0), sensor_act_list)  # 假设环境的step方法接受PyTorch张量
            # *************************** 记录内存 ***************************
            # 对所有sensor进行循环
            for i, sensor in enumerate(self.sensors):
                # 在环境更新后，获取sensor的新状态
                sensor_data_state = torch.tensor(sensor.get_sensor_data(), dtype=torch.float32).unsqueeze(0)
                new_sensor_state = [sensor_data_state]  # 更新后的sensor状态列表
                
                # 将某个sensor的当前状态、输出、新的奖励和新状态存储到sensor_memory字典
                if sensor.no in self.sensor_memory:
                    self.sensor_memory[sensor.no].append([sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state])
                else:
                    self.sensor_memory[sensor.no] = [[sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state]]
            
            # 对所有UAV进行循环处理
            for i, agent in enumerate(self.agents):
                state_map = torch.tensor(new_state_maps[i], dtype=torch.float32).unsqueeze(0)  # 添加批次维度并转换为张量
                # 获取UAV当前观察状态，并添加批次维度
                state_map = torch.tensor(self.env.get_obs(agent), dtype=torch.float32).unsqueeze(0)
                # 获取并处理UAV的总数据状态和完成数据状态
                total_data_state = torch.tensor(agent.get_total_data(), dtype=torch.float32).unsqueeze(0)
                done_data_state = torch.tensor(agent.get_done_data(), dtype=torch.float32).unsqueeze(0)
                # 处理带宽数据
                band = torch.tensor([agent.action.bandwidth], dtype=torch.float32).unsqueeze(0)
                # 组合新的状态列表
                new_states = [state_map, total_data_state, done_data_state, band]

                # 将某个UAV的当前状态、softmax输出、新的奖励值、新状态以及完成状态存储到agent_memory字典
                if agent.no in self.agent_memory:
                    self.agent_memory[agent.no].append([cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [[cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]

            # 更新后，获取中心节点相关的数据
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = torch.tensor(new_done_buffer_list, dtype=torch.float32).unsqueeze(0)
            new_pos_list = torch.tensor(new_pos_list, dtype=torch.float32).unsqueeze(0)

            # 将中心节点的当前状态、输出、新的奖励和新状态存储到center_memory列表
            self.center_memory.append([[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])
        
        # 如果是随机动作的情况
        else:
            # 对所有sensor进行处理
            sensor_state_list = []  # sensor状态列表，只有一种状态——sensor_buffer
            sensor_softmax_list = []  # sensor动作的softmax形式列表
            sensor_act_list = []  # sensor动作列表
            for i, sensor in enumerate(self.sensors):
                # 从环境中获取sensor的数据，并增加一个批次维度
                sensor_data_state = torch.tensor(sensor.get_sensor_data(), dtype=torch.float32).unsqueeze(0)
                sensor_state = [sensor_data_state]
                sensor_state_list.append(sensor_state)

                # 随机选择sensor动作
                sensor_comdev = np.random.rand(self.s_index_dim)  # 使用numpy生成随机动作数据
                sensor_comdev_tensor = torch.tensor(sensor_comdev, dtype=torch.float32).unsqueeze(0)  # 转换为张量并增加维度

                # 将随机生成的动作添加到动作列表
                sensor_act_list.append([sensor_comdev_tensor])

                # 将随机动作的输出转换成softmax形式（实际上这里没有使用softmax，仅为了示例添加维度）
                sensor_softmax_list.append([sensor_comdev_tensor])
            # 对所有agents进行处理
            agent_act_list = []  # agent动作列表
            softmax_list = []  # agent动作的softmax形式列表
            cur_state_list = []  # 当前状态列表
            band_vec = np.zeros(self.agent_num)  # 初始化带宽列表
            
            for i, agent in enumerate(self.agents):
                # 从环境中获取agent的观察数据，并添加批次维度
                state_map = torch.tensor(self.env.get_obs(agent), dtype=torch.float32).unsqueeze(0)
                total_data_state = torch.tensor(agent.get_total_data(), dtype=torch.float32).unsqueeze(0)
                done_data_state = torch.tensor(agent.get_done_data(), dtype=torch.float32).unsqueeze(0)
                band = torch.tensor([agent.action.bandwidth], dtype=torch.float32).unsqueeze(0)
                
                assemble_state = [state_map, total_data_state, done_data_state, band]
                cur_state_list.append(assemble_state)
                
                # 随机选择agent的动作
                move = random.choice(list(self.move_dict.values()))
                execution = np.zeros(agent.max_buffer_size)
                offloading = np.zeros(agent.max_buffer_size)
                execution[np.random.randint(agent.max_buffer_size)] = 1
                offloading[np.random.randint(agent.max_buffer_size)] = 1
                
                # 记录和计算移动距离
                agent.t_distance = np.linalg.norm(np.array(move))
                agent.e_distance += agent.t_distance
                agent.total_distance += agent.t_distance
                self.UAVs_total_distance += agent.t_distance

                # 创建用于神经网络输出的softmax形式数据
                move_softmax = torch.zeros((2 * self.env.move_r + 1, 2 * self.env.move_r + 1, 1))
                op_softmax = torch.zeros(self.buffstate_shape)
                move_ori = [move[1] + self.env.move_r, move[0] + self.env.move_r]
                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(execution)] = 1
                op_softmax[1][np.argmax(offloading)] = 1
                
                move_softmax = move_softmax.unsqueeze(0)
                op_softmax = op_softmax.unsqueeze(0)
                
                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax, op_softmax])

            # 对所有中心（center）进行处理
            # 从环境中获取中心的输入状态并打包
            done_buffer_list, pos_list = self.env.get_center_state()  # 获取每个UAV的计算完成数据和位置坐标
            done_buffer_list = torch.tensor(done_buffer_list, dtype=torch.float32).unsqueeze(0)  # 转为张量并增加批次维度
            pos_list = torch.tensor(pos_list, dtype=torch.float32).unsqueeze(0)
            band_vec = torch.tensor(band_vec, dtype=torch.float32).unsqueeze(0)  # 带宽数据增加批次维度

            # 随机选择中心的动作
            new_bandvec = np.random.rand(self.agent_num)  # 生成随机带宽分配
            new_bandvec /= np.sum(new_bandvec)  # 归一化带宽分配

            # 将agent和center的动作输入到环境中进行状态更新
            new_state_maps, new_rewards, average_age, fairness_index, done, info = self.env.step(agent_act_list, new_bandvec, sensor_act_list)

            # 记录内存部分
            # 对所有sensor进行循环
            for i, sensor in enumerate(self.sensors):
                # 在环境更新后，获取sensor的新状态
                sensor_data_state = torch.tensor(sensor.get_sensor_data(), dtype=torch.float32).unsqueeze(0)  # 将数据转为张量并增加批次维度
                new_sensor_state = [sensor_data_state]
                # 更新sensor的内存记录
                if sensor.no in self.sensor_memory:
                    self.sensor_memory[sensor.no].append([sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state])
                else:
                    self.sensor_memory[sensor.no] = [[sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state]]

            # 对所有UAV进行循环
            for i, agent in enumerate(self.agents):
                state_map = torch.tensor(new_state_maps[i], dtype=torch.float32)
                total_data_state = torch.tensor(agent.get_total_data(), dtype=torch.float32).unsqueeze(0)
                done_data_state = torch.tensor(agent.get_done_data(), dtype=torch.float32).unsqueeze(0)
                band = torch.tensor([agent.action.bandwidth], dtype=torch.float32).unsqueeze(0)
                new_states = [state_map, total_data_state, done_data_state, band]
                # 更新UAV的内存记录
                if agent.no in self.agent_memory:
                    self.agent_memory[agent.no].append([cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [[cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]

            # 更新后，获取中心的相关数据
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = torch.tensor(new_done_buffer_list, dtype=torch.float32).unsqueeze(0)
            new_pos_list = torch.tensor(new_pos_list, dtype=torch.float32).unsqueeze(0)
            # 更新中心的内存记录
            self.center_memory.append([[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])

            # 总结所有UAVs的总移动距离信息
            self.summaries['UAVs_total_distance'] = self.UAVs_total_distance
            # 返回最新的奖励信息
            return new_rewards[-1], average_age[-1], fairness_index[-1]
        
    def replay(self):
        # 0. sensor replay
        # 对sensor_memory字典中的所有UAV及UAV的经验进行循环
        mu = 0.1
        for no, sensor_memory in self.sensor_memory.items():
            # 若经验缓冲区长度 < 批大小（经验缓冲区中样本数量不足）
            if len(sensor_memory) < self.batch_size:
                # 跳过sensor replay training
                continue
            # 对经验缓冲区内的数据进行采样
            # 采样规则：最新的0.25*batch_size条经验数据+最新的2*batch_size条经验数据中随机抽取的0.75*batch_size条
            samples = sensor_memory[-int(self.batch_size * self.sample_prop):] + random.sample(sensor_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))

            # 从采样到的经验数据中提取并整理各类数据
            states = torch.stack([torch.tensor(sample[0], dtype=torch.float32) for sample in samples])
            actions = torch.stack([torch.tensor(sample[1], dtype=torch.float32) for sample in samples])
            rewards = torch.tensor([sample[2] for sample in samples], dtype=torch.float32).unsqueeze(-1)
            new_states = torch.stack([torch.tensor(sample[3], dtype=torch.float32) for sample in samples])

            # 使用target网络预测下一时刻的动作和奖励
            with torch.no_grad():
                new_actions = self.target_sensor_actors[no](new_states)
                sq_future = self.target_sensor_critics[no](new_states, new_actions)

            # 计算目标Q值
            target_qs = rewards + sq_future * self.gamma

            # 训练critic网络
            self.sensor_critics[no].train()
            optimizer = self.sensor_critic_opts[no]
            optimizer.zero_grad()
            q_values = self.sensor_critics[no](states, actions)
            loss_fn = torch.nn.MSELoss()
            sc_loss = loss_fn(q_values, target_qs)
            sc_loss.backward()
            optimizer.step()

            # 训练actor网络
            self.sensor_actors[no].train()
            optimizer = self.sensor_actor_opts[no]
            optimizer.zero_grad()
            predicted_actions = self.sensor_actors[no](states)
            actor_loss = -self.sensor_critics[no](states, predicted_actions).mean()
            actor_loss.backward()
            optimizer.step()

            # 记录训练的损失
            self.summaries['sensor%s-critic_loss' % no] = sc_loss.item()
            self.summaries['sensor%s-actor_loss' % no] = actor_loss.item()

        for no, agent_memory in self.agent_memory.items():
            # 若经验缓冲区长度 < 批大小（经验缓冲区中样本数量不足）
            if len(agent_memory) < self.batch_size:
                # 跳过agent replay training
                continue

            # 对经验缓冲区内的数据进行采样
            samples = agent_memory[-int(self.batch_size * self.sample_prop):] + random.sample(agent_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))

            # 从采样到的经验数据中提取并整理各类数据
            states = torch.stack([torch.tensor(sample[0], dtype=torch.float32) for sample in samples])
            actions = torch.stack([torch.tensor(sample[1], dtype=torch.float32) for sample in samples])
            rewards = torch.tensor([sample[2] for sample in samples], dtype=torch.float32).unsqueeze(-1)
            new_states = torch.stack([torch.tensor(sample[3], dtype=torch.float32) for sample in samples])

            # 使用target网络预测下一时刻的动作和奖励
            with torch.no_grad():
                new_actions = self.target_agent_actors[no](new_states)
                q_future = self.target_agent_critics[no](new_states, new_actions)

            # 计算目标Q值
            target_qs = rewards + q_future * self.gamma

            # 训练critic网络
            self.agent_critics[no].train()
            critic_optimizer = self.agent_critic_opts[no]
            critic_optimizer.zero_grad()
            q_values = self.agent_critics[no](states, actions)
            loss_fn = torch.nn.MSELoss()
            ac_loss = loss_fn(q_values, target_qs)
            ac_loss.backward()
            critic_optimizer.step()

            # 训练actor网络
            self.agent_actors[no].train()
            actor_optimizer = self.agent_actor_opts[no]
            actor_optimizer.zero_grad()
            predicted_actions = self.agent_actors[no](states)
            actor_loss = -self.agent_critics[no](states, predicted_actions).mean()
            actor_loss.backward()
            actor_optimizer.step()

            # 记录训练的损失
            self.summaries['agent%s-critic_loss' % no] = ac_loss.item()
            self.summaries['agent%s-actor_loss' % no] = actor_loss.item()
         # 2. center replay
        if len(self.center_memory) < self.batch_size:
            return

        center_samples = self.center_memory[-int(self.batch_size * self.sample_prop):] + random.sample(self.center_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))

        # 提取并整理经验数据
        done_buffer_list = torch.stack([torch.tensor(sample[0][0], dtype=torch.float32) for sample in center_samples])
        pos_list = torch.stack([torch.tensor(sample[0][1], dtype=torch.float32) for sample in center_samples])
        bandvec_act = torch.stack([torch.tensor(sample[1], dtype=torch.float32) for sample in center_samples])
        c_rewards = torch.tensor([sample[2] for sample in center_samples], dtype=torch.float32).unsqueeze(-1)

        new_done_buffer_list = torch.stack([torch.tensor(sample[3][0], dtype=torch.float32) for sample in center_samples])
        new_pos_list = torch.stack([torch.tensor(sample[3][1], dtype=torch.float32) for sample in center_samples])

        # 使用target网络预测下一时刻的奖励
        with torch.no_grad():
            new_c_actions = self.target_center_actor(new_done_buffer_list, new_pos_list)
            cq_future = self.target_center_critic(new_done_buffer_list, new_pos_list, new_c_actions)

        c_target_qs = c_rewards + cq_future * self.gamma

        # 训练center critic网络
        self.center_critic.train()
        critic_optimizer = self.center_critic_optimizer
        critic_optimizer.zero_grad()
        cq_values = self.center_critic(done_buffer_list, pos_list, bandvec_act)
        cc_loss = torch.mean((cq_values - c_target_qs) ** 2)
        cc_loss.backward()
        critic_optimizer.step()

        # 训练center actor网络
        self.center_actor.train()
        actor_optimizer = self.center_actor_optimizer
        actor_optimizer.zero_grad()
        c_act = self.center_actor(done_buffer_list, pos_list)
        ca_loss = -torch.mean(self.center_critic(done_buffer_list, pos_list, c_act))
        ca_loss.backward()
        actor_optimizer.step()

        # 记录损失
        self.summaries['center-critic_loss'] = cc_loss.item()
        self.summaries['center-actor_loss'] = ca_loss.item()

    def save_model(self, episode, time_str):
        # 保存所有sensor的actor和critic模型
        for i in range(self.sensor_num):
            torch.save(self.sensor_actors[i].state_dict(), 'logs/models/{}/sensor-actor-{}_episode{}.pt'.format(time_str, i, episode))
            torch.save(self.sensor_critics[i].state_dict(), 'logs/models/{}/sensor-critic-{}_episode{}.pt'.format(time_str, i, episode))
        # 保存所有UAV的actor和critic模型
        for i in range(self.agent_num):
            torch.save(self.agent_actors[i].state_dict(), 'logs/models/{}/agent-actor-{}_episode{}.pt'.format(time_str, i, episode))
            torch.save(self.agent_critics[i].state_dict(), 'logs/models/{}/agent-critic-{}_episode{}.pt'.format(time_str, i, episode))
        # 保存center的actor和critic模型
        torch.save(self.center_actor.state_dict(), 'logs/models/{}/center-actor_episode{}.pt'.format(time_str, episode))
        torch.save(self.center_critic.state_dict(), 'logs/models/{}/center-critic_episode{}.pt'.format(time_str, episode))



