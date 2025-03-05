from torch.distributions import OneHotCategorical
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import gym
from collections import deque
import random
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
from PIL import Image
import os
from IPython.display import display, Image as IPImage
from copy import deepcopy
from typing import List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    'DISH_DISP_DISTANCE_REW': 0,
    'POT_DISTANCE_REW': 0,
    'SOUP_DISTANCE_REW': 0
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
# layout = "counter_circuit_o_1order"


class Actor(nn.Module):
    def __init__(self, n_state, n_action, n_hidden=64, n_layer=3, dropout=False, p_dropout=0.2):
        """
        mapping state to q(state,action) to get q-values
        using 3 layer full connected nn
        n_state (int): Dimension of each state, 8 for lunarLander env, 6 continues, 2 discrete (0 or 1)
        n_action (int): Dimension of each action, 4 for lunarLander agent, 

        """
        super(Actor, self).__init__()
        layers = [n_state]+[n_hidden]*(n_layer-1)+[n_action]
        networklayers = []
        for i, n_current in enumerate(layers[:-1]):
            networklayers.append(nn.Linear(n_current, layers[i+1]))
            if dropout:
                networklayers.append(nn.Dropout(p=p_dropout))
            if i < n_layer - 1:
                networklayers.append(nn.ReLU())
        self.fc = nn.Sequential(*networklayers)
        # self.fc = nn.Sequential(nn.Linear(n_state, n_hidden),
        #                        nn.ReLU(),
        #                        nn.Linear(n_hidden, n_hidden),
        #                        nn.ReLU(),
        #                        nn.Linear(n_hidden, n_action))

    def forward(self, x):
        return self.fc(x)


class Critic_centralized(nn.Module):
    # Inspired by https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py
    def __init__(self, n_state=96, n_action=6, n_agents=2, n_hidden=64):
        """
        mapping state to q(state,action) to get q-values
        using 3 layer full connected nn
        n_state (int): Dimension of all state, 96 for overcooked-AI as obs[0] has the same information as obs[1], just the index is different
                       So, n_state here is the obs[0]. For other MADDPGs, n_state need to include all observed information for all agents.
        n_action (int): Dimension of all action, 12 for overcooked-AI(6x2)

        """
        super(Critic_centralized, self).__init__()
        # Here, because states are fully observed for all agents, thus, just using agent0's obs as the state information
        # Otherwise, the input of the first layer should be n_state*n_agents
        self.ln1 = nn.Linear(n_state, 128)
        self.ln2 = nn.Linear(128+n_action*n_agents, n_hidden)
        # self.ln3 = nn.Linear(64, 64)
        self.ln3 = nn.Linear(n_hidden, 1)

    def forward(self, obs0, actions):
        x = F.relu(self.ln1(obs0))
        x = torch.cat([x, actions], 1)  # dim0:sample, dim1:obs
        x = self.ln3(F.relu(self.ln2(x)))
        return x


class GST():
    """
        Gapped Straight-Through Estimator

        With help from: https://github.com/chijames/GST/blob/267ab3aa202d7a0cfd5b5861bd3dcad87faefd9f/model/basic.py
    """

    def __init__(self, temperature=0.7, gap=1.0):
        self.temperature = temperature
        self.gap = gap

    @torch.no_grad()
    def _calculate_movements(self, logits, DD):
        max_logit = logits.max(dim=-1, keepdim=True)[0]
        selected_logit = torch.gather(
            logits, dim=-1, index=DD.argmax(dim=-1, keepdim=True))
        m1 = (max_logit - selected_logit) * DD
        m2 = (logits + self.gap - max_logit).clamp(min=0.0) * (1 - DD)
        return m1, m2

    def __call__(self, logits, need_gradients=True):
        DD = OneHotCategorical(logits=logits).sample()
        if need_gradients:
            m1, m2 = self._calculate_movements(logits, DD)
            surrogate = F.softmax((logits + m1 - m2) /
                                  self.temperature, dim=-1)
            """Returns `value` but backpropagates gradients through `surrogate`."""
            return surrogate + (DD - surrogate).detach()
        else:
            return DD


class MADDPG(object):  # Define ONE Agent
    def __init__(self, n_agents, n_state, n_action, ge_temp=0.7, n_hidden=64, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99,
                 batch_size=64, replay_mem=10000, update_frequency=5, tau=1e-3, gradient_clip=1.0, policy_regulariser=0.001):
        self.n_state = n_state
        self.n_action = n_action
        self.n_hidden = n_hidden
        # learning settings
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        # deep networks
        self.n_agents = n_agents
        self.actors = [Actor(n_state, n_action).to(device)
                       for i in range(n_agents)]
        self.critics = [Critic_centralized(n_state, n_action, n_agents).to(device)
                        for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.actor_optimizers = [
            optim.Adam(a.parameters(), lr=self.lr_actor, eps=0.001) for a in self.actors]
        self.critic_optimizers = [
            optim.Adam(c.parameters(), lr=self.lr_critic, eps=0.001) for c in self.critics]
        self.gradient_clip = gradient_clip

        # parameter sharing among critics
        for agent_id in range(1, self.n_agents):
            self.critics[agent_id] = self.critics[0]
            self.critics_target[agent_id] = self.critics_target[0]
            self.critic_optimizers[agent_id] = self.critic_optimizers[0]

        self.critic_criterion = nn.MSELoss()  # Loss function

        # Replay memory buffer
        self.buffer = deque(maxlen=int(replay_mem))
        self.t_step = 0
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.tau = tau

        # Gradient estimator
        self.gradient_estimator = GST(temperature=ge_temp)
        self.policy_regulariser = policy_regulariser

        # To device
        # for a in self.actors:
        #    a.to(device)
        # for c in self.critics:
        #    c.to(device)

    def choose_agent_action(self, s, agentidx: int):
        """Choose next action based on actor logits"""
        state = torch.tensor(s).float().unsqueeze(0).to(device)

        policy_output = self.actors[agentidx](state)
        gs_output = self.gradient_estimator(
            policy_output, need_gradients=False)
        return torch.argmax(gs_output, dim=-1)

        # First get q(s,a) values
        state = torch.tensor(s).float().unsqueeze(0).to(device)

        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state)
        self.q_local.train()

        if np.random.random() < epsilon:
            return np.random.randint(self.n_action)
        return torch.argmax(action_values).to('cpu').item()

    def acts(self, obs: List):
        actions = [self.choose_agent_action(
            obs[i], i) for i in range(self.n_agents)]
        return actions

    def step(self, state, action, reward, next_state, done):
        # Add to buffer
        self.buffer.append([state, action, reward, next_state, done])
        # update based on update_frequency
        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            # Only learn when there is enough samples in the buffer to make the update robust
            if len(self.buffer) > self.batch_size:
                experiences = random.sample(self.buffer, self.batch_size)
                # experiences = self.buffer[np.random.choice(self.buffer.shape[0], self.batch_size, replace=False),:]
                # print("exp shape:",np.array(experiences).shape)
                self.update_Qlearning(experiences)

    def update_critics(self, samples_batch):
        target_actions = torch.concat(target_actions_per_agent, axis=1)
        sampled_actions = torch.concat(sampled_actions_per_agent, axis=1)

        Q_next_target = self.critic(torch.concat(
            (all_nobs, target_actions), dim=1))
        target_ys = rewards + (1 - dones) * gamma * Q_next_target
        behaviour_ys = self.critic(torch.concat(
            (all_obs, sampled_actions), dim=1))

        loss = F.mse_loss(behaviour_ys, target_ys.detach())

        self.optim_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.gradient_clip)
        self.optim_critic.step()

        agent.update_critic(
            all_obs=batched_obs,
            all_nobs=batched_nobs,
            target_actions_per_agent=target_actions_one_hot,
            sampled_actions_per_agent=sampled_actions_one_hot,
            rewards=rewards[ii].unsqueeze(dim=1),
            dones=sample['dones'][ii].unsqueeze(dim=1),
            gamma=self.gamma,
        )

    def update_policy(self, agentidx: int):
        pass

    def act_target(self, s, agentidx: int):
        state = torch.tensor(s).float().unsqueeze(0).to(device)
        policy_output = self.actors_target[agentidx](state)
        gs_output = self.gradient_estimator(
            policy_output, need_gradients=False)
        return torch.argmax(gs_output, dim=-1)

    def predict(self, s):
        """get the actions for the given s, and agentidx"""
        state = torch.tensor(s).float().unsqueeze(0).to(device)
        self.q_local.eval()
        with torch.no_grad():
            action_values = self.q_local(state)
        self.q_local.train()
        return action_values

    def update_Qlearning(self, experiences):
        """
        update Q table based on Q-learning method
        experiences are sampled from self.buffer, which is list of [state, action, reward, next_state, done]
        """

        # extract states,actions, rewards, next_states,dones
        states = torch.from_numpy(
            np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e[4] for e in experiences if e is not None])).float().to(device)
        # print("actions",actions)
        # states, actions, rewards, next_states, dones = experiences
        # states=torch.tensor(states).float().to(device)
        # actions=torch.tensor(actions).int().to(device)
        # rewards=torch.tensor(rewards).float().to(device)
        # next_states=torch.tensor(next_states).float().to(device)
        # dones=torch.tensor(dones).to(device)
        # print(states)

        # Using matrix calculations rather than loops
        # For each next_states, get the maxQ, here no need to backprop, so detach()!
        q_max_next = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Q target values from bellman Q-learning equation, if current is done, then just the reward
        q_targets = rewards + self.gamma * q_max_next * (1 - dones)
        # Calculate expected value from local network
        q_expected = self.q_local(states).gather(1, actions)

        # loss backprop to update q_local network
        loss = self.criterion(q_targets, q_expected)
        # for g in self.optimizer.param_groups:
        #    g['lr'] = 0.001
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        for p_local, p_target in zip(self.q_local.parameters(), self.q_target.parameters()):
            # p_target = tau*p_local + (1 - tau)*p_target
            p_target.data.copy_(self.tau*p_local.data +
                                (1.0-self.tau)*p_target.data)

    def reset_lr(self, newlr):
        self.lr = newlr
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def reset_batchsize(self, newbatchsize):
        self.batch_size = int(newbatchsize)

    def reset_tau(self, tau_new):
        self.tau = tau_new

    def reset_update_frequency(self, newfrequency):
        self.update_frequency = newfrequency


class trainDQN(object):
    def __init__(self, layout="cramped_room", n_hidden=64, lr=1e-3, gamma=0.99,
                 batch_size=128, replay_mem=10000, update_frequency=5, tau=1e-3,
                 epsilon=0.9, eps_decay=0.99, epsilon_min=0.001):
        mdp = OvercookedGridworld.from_layout_name(
            layout, rew_shaping_params=reward_shaping)
        base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
        env = gym.make("Overcooked-v0", base_env=base_env,
                       featurize_fn=base_env.featurize_state_mdp)
        N_statespace = env.observation_space.shape[0]
        N_actions = env.action_space.n
        env.close()

        self.envname = layout
        # env.observation_space.shape[0]
        self.agent0 = DQNLearningAgent(N_statespace, N_actions, n_hidden=n_hidden,
                                       lr=lr, gamma=gamma, batch_size=batch_size,
                                       replay_mem=replay_mem, update_frequency=update_frequency, tau=tau)
        self.agent1 = DQNLearningAgent(N_statespace, N_actions, n_hidden=n_hidden,
                                       lr=lr, gamma=gamma, batch_size=batch_size,
                                       replay_mem=replay_mem, update_frequency=update_frequency, tau=tau)
        # Greedy epsilon settings
        self.epsilon = epsilon  # Change it as episode increases
        self.eps_decay = eps_decay  # episilon decay after each episode
        self.epsilon_min = epsilon_min  # minimun exploring rate
        self.epsilon_max = epsilon

        self.converging_epsilon = 0.0001
        self.verbose = True

        # Train settings:
        self.episode_start = 0
        self.lr_max = 1e-2
        self.lr_min = 1e-5

    def train(self, targetScore=7, targetN=10, N_converge_threshold=5,
              min_episode=50, maxNewEpisode=2000,
              saveEvery100=5, save_name_tag: str = "gamma",
              batch_size=128, lr=1e-4, update_frequency=5, tau=1e-3,
              fine_tune_lr=False):
        # init train settings
        if save_name_tag == "gamma":
            # GAMMA({self.agent.gamma})
            save_name_tag = f"{save_name_tag}({self.agent.gamma})"
        elif save_name_tag == "lr":
            save_name_tag = f"lr({self.agent.lr})"
        elif save_name_tag == "batch_size":
            save_name_tag = f"{save_name_tag}({self.agent.batch_size})"
        elif save_name_tag == "update_frequency":
            save_name_tag = f"{save_name_tag}({self.agent.update_frequency})"
        elif save_name_tag == "n_hidden":
            save_name_tag = f"{save_name_tag}({self.agent.n_hidden})"
        elif save_name_tag == "tau":
            save_name_tag = f"{save_name_tag}({self.agent.tau})"

        save_name_tag = self.envname+save_name_tag
        saveEvery100 = 100*int(saveEvery100)
        if batch_size != self.agent.batch_size:
            self.agent.reset_batchsize(batch_size)
        if lr != self.agent.lr:
            self.agent.reset_lr(newlr=lr)
        if update_frequency != self.agent.update_frequency:
            self.agent.reset_update_frequency(update_frequency)
        if tau != self.agent.tau:
            self.agent.reset_tau(tau)

        env = gym.make(self.envname)  # render_mode='human'
        # Training until self.Q converge
        scores = []
        scores_last = deque(maxlen=int(targetN))
        best_avg_last = 0
        # while not converge:
        converge = False
        n_epsmin = 0
        N_converge = 0
        N_reset_eps = 500  # If not converge after epsilon arrived at epsilon_min, reset epsilon
        # N_converge_threshold= 10
        i = 0
        while (not (converge and N_converge >= N_converge_threshold)) and i < maxNewEpisode:
            obs = env.reset()
            overall_reward = 0
            done = False
            while not done:
                obs0 = obs["both_agent_obs"][0]
                obs1 = obs["both_agent_obs"][1]
                a0 = self.agent0.choose_action(obs0, self.epsilon)
                a1 = self.agent1.choose_action(obs1, self.epsilon)
                obs, R, done, info = env.step([a0, a1])
                obs0_new = obs["both_agent_obs"][0]
                obs1_new = obs["both_agent_obs"][1]
                # Reshape rewards
                if env.agent_idx == 0:
                    agent_0_reward = info["shaped_r_by_agent"][0]
                    agent_1_reward = info["shaped_r_by_agent"][1]
                else:
                    agent_0_reward = info["shaped_r_by_agent"][1]
                    agent_1_reward = info["shaped_r_by_agent"][0]
                if done:
                    # reward equal to bootstrap values
                    agent_0_reward += torch.max(
                        self.agent0.predict(obs0_new)).to('cpu').item()
                    agent_1_reward += torch.max(
                        self.agent1.predict(obs1_new)).to('cpu').item()

                # Learning
                self.agent0.step(state=obs0, action=a0, reward=agent_0_reward+R,  # info["shaped_r_by_agent"][0]+R,reshapedR[0]+R,
                                 next_state=obs0_new, done=done)
                self.agent1.step(state=obs1, action=a1, reward=agent_1_reward+R,  # info["shaped_r_by_agent"][1]+R,reshapedR[1]+R,
                                 next_state=obs1_new, done=done)
                # if info['shaped_r_by_agent'][0]+R != 0:
                #    print(
                #        f"current reward for agent0: {info['shaped_r_by_agent'][0]+R}")
                # Each served soup generates 20 reward
                num_soups_made += int(R / 20)
            scores.append(num_soups_made)
            scores_last.append(num_soups_made)
            i = i+1

            # Renew training settings
            meanscore_last = np.mean(scores_last)

            if self.epsilon > self.epsilon_min:
                # update epsilon greedy after each episode
                if num_soups_made <= 7:
                    decayweight2 = 7-num_soups_made
                    self.epsilon = self.epsilon * \
                        (self.eps_decay+decayweight2)/(1+decayweight2)

            if self.epsilon < self.epsilon_min:
                # update n_epsmin
                n_epsmin = n_epsmin+1
                # If converged once, then not reset
                if n_epsmin % N_reset_eps == 0 and (not converge):
                    print(f'Episode {i}\tReset epsilon:{self.epsilon_max}')
                    # self.save_checkpoint(avgScore=meanscore_last,filename=f"Score{meanscore_last:.2f}_checkpoint.pth.tar")
                    # reset epsilon, and reduce learning rate
                    self.epsilon = self.epsilon_max
                    if fine_tune_lr:
                        newlr0 = self.agent0.lr*0.8
                        newlr1 = self.agent1.lr*0.8
                        if newlr0 < self.lr_min:
                            newlr0 = self.lr_max
                        if newlr1 < self.lr_min:
                            newlr1 = self.lr_max
                        print(
                            f'Reset learning Rate newlr0:{newlr0},newlr1:{newlr1}')
                        self.agent0.reset_lr(newlr=newlr0)
                        self.agent1.reset_lr(newlr=newlr1)

            # Converge if scores does not change
            if i > min_episode and meanscore_last >= targetScore:
                converge = True
                if i % 10 == 0:
                    # After first converge, get another N_converge_threshold time, to be stable
                    N_converge = N_converge + 1
                    # print(f'Episode {i}\tAverage Score: {meanscore_last:.2f}')
                    if meanscore_last > best_avg_last:
                        best_avg_last = meanscore_last
                        # _episode{i}_avgscore{meanscore_last:.2f}
                        self.save_checkpoint(episode=i+self.episode_start, avgScore=meanscore_last,
                                             filename=f'bestcheckpoint_{save_name_tag}.pth.tar')
            if converge and meanscore_last < targetScore:
                converge = False
                N_converge = 0
            # Print Info
            if i % 100 == 0:
                print(
                    f'Episode {i+self.episode_start}\tAverage Score: {meanscore_last:.2f}')
                if i % saveEvery100 == 0:
                    # Save scores
                    df = pd.DataFrame(data={"episode": [
                                      self.episode_start+x for x in range(i-saveEvery100+1, i+1)], "Score": scores[(i-saveEvery100):i]})
                    df.to_csv(
                        f'score_history_{save_name_tag}.csv', mode='a', index=False, header=False)
                    # Test save_checkpoint and reload
                    if i % saveEvery100 == 0:
                        self.save_checkpoint(episode=i+self.episode_start, avgScore=meanscore_last,
                                             filename=f"episode{i+self.episode_start}_{save_name_tag}_checkpoint.pth.tar")

        # Save most recent checkpoint
        self.save_checkpoint(episode=i+self.episode_start, avgScore=meanscore_last,
                             filename=f'checkpoint_episode{i+self.episode_start}_{save_name_tag}_last.pth.tar')
        if i % saveEvery100 != 0:
            # such as i= 667, then save scores[0:667]
            # or i=1232, then save scores[1000:1232]
            j = i-i % saveEvery100
            df = pd.DataFrame(data={"episode": [
                              self.episode_start+x for x in range(j+1, i+1)], "Score": scores[j:i]})
            df.to_csv(f'score_history_{save_name_tag}.csv',
                      mode='a', index=False, header=False)
        env.close()
        return scores

    def test(self, targetN=100, render_mode="rgb_array", seed=2023, verbose=False):
        # render_mode="human"
        env_test = gym.make(self.envname, render_mode=render_mode)
        # width=600, height=600
        env_test.reset(seed=seed)
        # render()
        # Test current trained results
        scores = []
        steps = []
        lastStatus = []
        for i in range(targetN):
            s1 = env_test.reset()[0]
            overall_reward = 0
            done = False
            stepsToFinish = 0
            reward = 0
            done = False
            truncated = False
            while not done:
                # if rendering:
                #    env_test.render()
                a1 = self.agent.choose_action(s1, epsilon=0)
                s2, reward, done, truncated, info = env_test.step(a1)
                # if truncated and render_mode=="rgb_array":
                #    print(f"truncated:{s2}\nreward:{reward}\ndone:{done}\ninformation:{info}")
                #    img = env_test.render()
                #    imageio.imwrite("truncated.png", img)
                done = done or truncated

                s1 = s2
                overall_reward = overall_reward + reward
                stepsToFinish = stepsToFinish + 1
            scores.append(overall_reward)
            steps.append(stepsToFinish)
            if truncated:
                lastStatus.append(0)  # Terminated
            else:
                if reward == -100:
                    lastStatus.append(-1)  # crash
                else:
                    lastStatus.append(1)  # sucess

        env_test.close()
        avgScore = np.mean(scores)
        if verbose:
            print(
                f'{targetN} episode average score of the current agent: {avgScore:.2f}')
        return scores, steps, lastStatus

    def testOne(self, render_mode="rgb_array", filename="test", seed=2023, verbose=False):
        if render_mode != "None":
            # render_mode="human"
            env_test = gym.make(self.envname, render_mode=render_mode)
        else:
            env_test = gym.make(self.envname)
        # width=600, height=600
        env_test.reset(seed=seed)
        # render()
        # Test current trained results
        rewards = []
        frames = []
        s1 = env_test.reset()[0]
        overall_reward = 0
        lastStatus = 0  # 0: truncated, 1: success, -1: crash
        done = False
        reward = 0
        done = False
        truncated = False
        while not done:
            if render_mode == "rgb_array":
                frames.append(env_test.render())
            a1 = self.agent.choose_action(s1, epsilon=0)
            s2, reward, done, truncated, info = env_test.step(a1)
            done = done or truncated
            s1 = s2
            overall_reward = overall_reward + reward
            rewards.append(reward)

        if truncated:
            lastStatus = 0  # Terminated
        else:
            if reward == -100:
                lastStatus = -1  # crash
            else:
                lastStatus = 1  # sucess
        if render_mode == "rgb_array":
            imageio.mimsave(filename + ".gif", frames, fps=60)
            imageio.imwrite(f"{filename}_laststate.png", frames[-1])
        env_test.close()
        if verbose:
            print(
                f'the current agent got {overall_reward:.2f} total reward in {len(rewards)} steps. Final status: {lastStatus}')
        return rewards, lastStatus

    def save_checkpoint(self, episode, avgScore, filename="checkpoint.pth"):
        state = {
            'episode': episode,
            'average_score': avgScore,
            'gamma': self.agent.gamma,
            'q_local_state_dict': self.agent.q_local.state_dict(),
            'q_target_state_dict': self.agent.q_target.state_dict(),
            'q_local_optimizer_state_dict': self.agent.optimizer.state_dict()
        }
        save_dir = "./"
        torch.save(state, save_dir+filename)

    def load_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = torch.load(filename)
        self.episode_start = checkpoint['episode']
        self.agent.q_local.load_state_dict(checkpoint['q_local_state_dict'])
        self.agent.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.agent.optimizer.load_state_dict(
            checkpoint['q_local_optimizer_state_dict'])
        # Move to device
        self.agent.q_local = self.agent.q_local.to(device)
        self.agent.q_target = self.agent.q_target.to(device)
        self.optimizer_to(self.agent.optimizer, device)

    def optimizer_to(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(
                                device)
