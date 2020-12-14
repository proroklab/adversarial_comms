import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from matplotlib import colors
from functools import partial
from enum import Enum
import copy

from ray.rllib.env.multi_agent_env import MultiAgentEnv

# https://bair.berkeley.edu/blog/2018/12/12/rllib/

DEFAULT_OPTIONS = {
    'world_shape': [24, 24],
    'state_size': 48,
    'collapse_state': False,
    'termination_no_new_coverage': 10,
    'max_episode_len': -1,
    "min_coverable_area_fraction": 0.6,
    "map_mode": "random",
    "n_agents": [5],
    "disabled_teams_step": [False],
    "disabled_teams_comms": [False],
    'communication_range': 8.0,
    'one_agent_per_cell': False,
    'ensure_connectivity': True,
    'reward_type': 'semi_cooperative',
    #"operation_mode": 'all', # greedy_only, coop_only, don't default for now
    'episode_termination': 'early',
    'agent_observability_radius': None,
}

X = 1
Y = 0

class Dir(Enum):
    RIGHT  = 0
    LEFT   = 1
    UP     = 2
    DOWN   = 3

class WorldMap():
    def __init__(self, random_state, shape, min_coverable_area_fraction):
        self.shape = tuple(shape)
        self.min_coverable_area_fraction = min_coverable_area_fraction
        self.reset(random_state)

    def reset(self, random_state, mode="random"):
        self.coverage = np.zeros(self.shape, dtype=np.int)
        if mode == "random":
            if self.min_coverable_area_fraction == 1.0:
                self.map = np.zeros(self.shape, dtype=np.uint8)
            else:
                self.map = np.ones(self.shape, dtype=np.uint8)
                p = np.array([random_state.randint(0, self.shape[c]) for c in [Y, X]])
                while self.get_coverable_area_faction() < self.min_coverable_area_fraction:
                    d_p = np.array([[0, 1], [0, -1], [-1, 0], [1, 0]][random_state.randint(0, 4)])#*random_state.randint(1, 5)
                    p_new = np.clip(p + d_p, [0,0], np.array(self.shape)-1)
                    self.map[min(p[Y],p_new[Y]):max(p[Y],p_new[Y])+1, min(p[X],p_new[X]):max(p[X],p_new[X])+1] = 0
                    #print(min(p[Y],p_new[Y]),max(p[Y],p_new[Y])+1, min(p[X],p_new[X]),max(p[X],p_new[X])+1, np.sum(self.map))
                    p = p_new
        elif mode == "split_half_fixed" or mode == "split_half_fixed_block" or mode == "split_half_fixed_block_same_side":
            self.map = np.zeros(self.shape, dtype=np.uint8)
            self.map[:, int(self.shape[X]/2)] = 1
            if mode == "split_half_fixed":
                self.map[int(self.shape[Y]/2), int(self.shape[X]/2)] = 0

    def get_coverable_area_faction(self):
        coverable_area = ~(self.map > 0)
        return np.sum(coverable_area)/(self.map.shape[X]*self.map.shape[Y])

    def get_coverable_area(self):
        coverable_area = ~(self.map>0)
        return np.sum(coverable_area)

    def get_covered_area(self):
        coverable_area = ~(self.map>0)
        return np.sum((self.coverage > 0) & coverable_area)

    def get_coverage_fraction(self):
        coverable_area = ~(self.map>0)
        covered_area = (self.coverage > 0) & coverable_area
        return np.sum(covered_area)/np.sum(coverable_area)

class Action(Enum):
    NOP         = 0
    MOVE_RIGHT  = 1
    MOVE_LEFT   = 2
    MOVE_UP     = 3
    MOVE_DOWN   = 4

class Robot():
    def __init__(self,
                 index,
                 random_state,
                 world,
                 state_size,
                 collapse_state,
                 termination_no_new_coverage,
                 agent_observability_radius,
                 one_agent_per_cell):
        self.index = index
        self.world = world
        self.termination_no_new_coverage = termination_no_new_coverage
        self.state_size = state_size
        self.collapse_state = collapse_state
        self.initialized_rendering = False
        self.agent_observability_radius = agent_observability_radius
        self.one_agent_per_cell = one_agent_per_cell
        self.pose = np.array([-1, -1]) # assign negative pose so that during reset agents are not placed at same initial position
        self.reset(random_state)

    def reset(self, random_state, pose_mean=np.array([0, 0]), pose_var=1):
        def random_pos(var):
            return np.array([
                int(np.clip(random_state.normal(loc=pose_mean[c], scale=var), 0, self.world.map.shape[c]-1))
            for c in [Y, X]])

        current_pose_var = pose_var
        self.pose = random_pos(current_pose_var)
        self.prev_pose = self.pose.copy()
        while self.world.map.map[self.pose[Y], self.pose[X]] == 1 or (self.world.is_occupied(self.pose, self) and self.one_agent_per_cell):
            self.pose = random_pos(current_pose_var)
            current_pose_var += 0.1

        self.coverage = np.zeros(self.world.map.shape, dtype=np.bool)
        self.state = None
        self.no_new_coverage_steps = 0
        self.reward = 0

    def step(self, action):
        action = Action(action)

        delta_pose = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.NOP:         [ 0,  0]
        }[action]

        is_valid_pose = lambda p: all([p[c] >= 0 and p[c] < self.world.map.shape[c] for c in [Y, X]])
        is_obstacle = lambda p: self.world.map.map[p[Y]][p[X]] == 1

        self.prev_pose = self.pose.copy()
        desired_pos = self.pose + delta_pose
        if is_valid_pose(desired_pos) and (not self.world.is_occupied(desired_pos, self) or not self.one_agent_per_cell) and not is_obstacle(desired_pos):
            self.pose = desired_pos

        if self.world.map.coverage[self.pose[Y], self.pose[X]] == 0:
            self.world.map.coverage[self.pose[Y], self.pose[X]] = self.index
            self.reward = 1
            self.no_new_coverage_steps = 0
        else:
            self.reward = 0
            self.no_new_coverage_steps += 1

        self.coverage[self.pose[Y], self.pose[X]] = True
        #self.reward -= 1 # subtract each time step

    def update_state(self):
        coverage = self.coverage.copy().astype(np.int)
        if self.collapse_state:
            yy, xx = np.mgrid[:self.coverage.shape[Y], :self.coverage.shape[X]]
            for (cx, cy) in zip(xx[self.coverage]-self.pose[X], yy[self.coverage]-self.pose[Y]):
                if abs(cx) < self.state_size/2 and abs(cy) < self.state_size/2:
                    continue
                u = max(abs(cx), abs(cy))
                p_sq = np.round(self.pose + int(self.state_size/2)*np.array([cy/u, cx/u])).astype(np.int)
                coverage[p_sq[Y], p_sq[X]] += 1

        state_output_shape = np.array([self.state_size]*2, dtype=int)
        state_data = [
            self.to_coordinate_frame(self.world.map.map, state_output_shape, fill=1),
            self.to_coordinate_frame(coverage, state_output_shape, fill=0)
        ]
        if self.agent_observability_radius is not None:
            pose_map = np.zeros(self.world.map.shape, dtype=np.uint8)

            for team in self.world.teams.values():
                for r in team:
                    if not r is self and np.sum((r.pose - self.pose)**2) < self.agent_observability_radius**2:
                        pose_map[r.pose[Y], r.pose[X]] = 2
            pose_map[self.pose[Y], self.pose[X]] = 1
            state_data.append(self.to_coordinate_frame(pose_map, state_output_shape, fill=0))
        self.state = np.stack(state_data, axis=-1).astype(np.uint8)

        done = self.no_new_coverage_steps == self.termination_no_new_coverage
        return self.state, self.reward, done, {}

    def to_abs_frame(self, data):
        half_state_size = int(self.state_size / 2)
        return np.roll(data, self.pose, axis=(0, 1))[half_state_size:, half_state_size:]

    def to_coordinate_frame(self, m, output_shape, fill=0):
        half_out_shape = np.array(output_shape/2, dtype=np.int)
        padded = np.pad(m,([half_out_shape[Y]]*2,[half_out_shape[X]]*2), mode='constant', constant_values=fill)
        return padded[self.pose[Y]:self.pose[Y] + output_shape[Y], self.pose[X]:self.pose[X] + output_shape[Y]]

class CoverageEnv(gym.Env, EzPickle):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.seed()

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(env_config)

        self.fig = None
        self.map_colormap = colors.ListedColormap(['white', 'black', 'gray'])  # free, obstacle, unknown

        hsv = np.ones((self.cfg['n_agents'][1], 3))
        hsv[..., 0] = np.linspace(160/360, 250/360, self.cfg['n_agents'][1] + 1)[:-1]
        self.teams_agents_color = {
            0: [(1, 0, 0)],
            1: colors.hsv_to_rgb(hsv)
        }

        '''
        hsv = np.ones((sum(self.cfg['n_agents']), 3))
        hsv[..., 0] = np.linspace(0, 1, sum(self.cfg['n_agents']) + 1)[:-1]
        self.teams_agents_color = {}
        current_index = 0
        for i, n_agents in enumerate(self.cfg['n_agents']):
            self.teams_agents_color[i] = colors.hsv_to_rgb(hsv[current_index:current_index+n_agents])
            current_index += n_agents
        '''

        hsv = np.ones((len(self.cfg['n_agents']), 3))
        hsv[..., 0] = np.linspace(0, 1, len(self.cfg['n_agents']) + 1)[:-1]
        self.teams_colors = ['r', 'b'] #colors.hsv_to_rgb(hsv)

        n_all_agents = sum(self.cfg['n_agents'])
        self.observation_space = spaces.Dict({
            'agents': spaces.Tuple((
                spaces.Dict({
                    'map': spaces.Box(0, np.inf, shape=(self.cfg['state_size'], self.cfg['state_size'], 2 if self.cfg['agent_observability_radius'] is None else 3)),
                    'pos': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg['world_shape'][Y], self.cfg['world_shape'][X]]), dtype=np.int),
                }),
            )*n_all_agents), # Do not add this as additional dimension of map and pos since this way it is easier to handle in the model
            'gso': spaces.Box(-np.inf, np.inf, shape=(n_all_agents, n_all_agents)),
            'state': spaces.Box(low=0, high=2, shape=self.cfg['world_shape']+[2+len(self.cfg['n_agents'])]),
        })
        self.action_space = spaces.Tuple((spaces.Discrete(5),)*sum(self.cfg['n_agents']))

        self.map = WorldMap(self.world_random_state, self.cfg['world_shape'], self.cfg['min_coverable_area_fraction'])
        self.teams = {}
        agent_index = 1
        for i, n_agents in enumerate(self.cfg['n_agents']):
            self.teams[i] = []
            for j in range(n_agents):
                self.teams[i].append(
                    Robot(
                        agent_index,
                        self.agent_random_state,
                        self,
                        self.cfg['state_size'],
                        self.cfg['collapse_state'],
                        self.cfg['termination_no_new_coverage'],
                        self.cfg['agent_observability_radius'],
                        self.cfg['one_agent_per_cell']
                    )
                )
                agent_index += 1

        self.reset()

    def is_occupied(self, p, agent_ignore=None):
        for team_key, team in self.teams.items():
            if self.cfg['disabled_teams_step'][team_key]:
                continue
            for o in team:
                if o is agent_ignore:
                    continue
                if p[X] == o.pose[X] and p[Y] == o.pose[Y]:
                    return True
        return False

    def seed(self, seed=None):
        self.agent_random_state, seed_agents = seeding.np_random(seed)
        self.world_random_state, seed_world = seeding.np_random(seed)
        return [seed_agents, seed_world]

    def reset(self):
        self.dones = {key: [False for _ in team] for key, team in self.teams.items()}
        self.timestep = 0
        self.map.reset(self.world_random_state, self.cfg['map_mode'])

        def random_pos_seed(team_key):
            rnd = self.agent_random_state
            if self.cfg['map_mode'] == "random":
                return np.array([rnd.randint(0, self.map.shape[c]) for c in [Y, X]])
            if self.cfg['map_mode'] == "split_half_fixed":
                return np.array([
                    rnd.randint(0, self.map.shape[Y]),
                    rnd.randint(0, int(self.map.shape[X]/3))
                ])
            elif self.cfg['map_mode'] == "split_half_fixed_block":
                if team_key == 0:
                    return np.array([
                        rnd.randint(0, self.map.shape[Y]),
                        rnd.randint(0, int(self.map.shape[X] / 3))
                    ])
                else:
                    return np.array([
                        rnd.randint(0, self.map.shape[Y]),
                        rnd.randint(2*int(self.map.shape[X] / 3), self.map.shape[X])
                    ])
            elif self.cfg['map_mode'] == "split_half_fixed_block_same_side":
                return np.array([
                    rnd.randint(0, self.map.shape[Y]),
                    rnd.randint(2*int(self.map.shape[X] / 3), self.map.shape[X])
                ])

        pose_seed = None
        for team_key, team in self.teams.items():
            if not self.cfg['map_mode'] == "random" or pose_seed is None:
                # shared pose_seed if random map mode
                pose_seed = random_pos_seed(team_key)
                while self.map.map[pose_seed[Y], pose_seed[X]] == 1:
                    pose_seed = random_pos_seed(team_key)
            for r in team:
                r.reset(self.agent_random_state, pose_mean=pose_seed, pose_var=1)
        return self.step([Action.NOP]*sum(self.cfg['n_agents']))[0]

    def compute_gso(self, team_id=0):
        own_team_agents = [(agent, self.cfg['disabled_teams_comms'][team_id]) for agent in self.teams[team_id]]
        other_agents = [(agent, self.cfg['disabled_teams_comms'][other_team_id]) for other_team_id, team in self.teams.items() for agent in team if not team_id == other_team_id]

        all_agents = own_team_agents + other_agents # order is important since in model the data is concatenated in this order as well
        dists = np.zeros((len(all_agents), len(all_agents)))
        done_matrix = np.zeros((len(all_agents), len(all_agents)), dtype=np.bool)
        for agent_y in range(len(all_agents)):
            for agent_x in range(agent_y):
                dst = np.sum(np.array(all_agents[agent_x][0].pose - all_agents[agent_y][0].pose)**2)
                dists[agent_y, agent_x] = dst
                dists[agent_x, agent_y] = dst

                d = all_agents[agent_x][1] or all_agents[agent_y][1]
                done_matrix[agent_y, agent_x] = d
                done_matrix[agent_x, agent_y] = d

        current_dist = self.cfg['communication_range']
        A = dists < (current_dist**2)
        active_row = ~np.array([a[1] for a in all_agents])
        if self.cfg['ensure_connectivity']:
            def is_connected(m):
                def walk_dfs(m, index):
                    for i in range(len(m)):
                        if m[index][i]:
                            m[index][i] = False
                            walk_dfs(m, i)

                m_c = m.copy()
                walk_dfs(m_c, 0)
                return not np.any(m_c.flatten())

            # set done teams as generally connected since they should not be included by increasing connectivity
            while not is_connected(A[active_row][:, active_row]):
                current_dist *= 1.1
                A = (dists < current_dist**2)

        # Mask out done agents
        A = (A & ~done_matrix).astype(np.int)

        # normalization: refer https://github.com/QingbiaoLi/GraphNets/blob/master/Flocking/Utils/dataTools.py#L601
        np.fill_diagonal(A, 0)
        deg = np.sum(A, axis = 1) # nNodes (degree vector)
        D = np.diag(deg)
        Dp = np.diag(np.nan_to_num(np.power(deg, -1/2)))
        L = A # D-A
        gso = Dp @ L @ Dp
        return gso

    def step(self, actions):
        self.timestep += 1
        action_index = 0
        for i, team in enumerate(self.teams.values()):
            for agent in team:
                if not self.cfg['disabled_teams_step'][i]:
                    agent.step(actions[action_index])
                action_index += 1

        states, rewards = {}, {}
        for team_key, team in self.teams.items():
            states[team_key] = []
            rewards[team_key] = {}
            for i, agent in enumerate(team):
                state, reward, done, _ = agent.update_state()
                states[team_key].append(state)
                rewards[team_key][i] = reward
                if done:
                    self.dones[team_key][i] = True
        dones = {}
        world_done = self.timestep == self.cfg['max_episode_len'] or self.map.get_coverage_fraction() == 1.0
        for key in self.teams.keys():
            dones[key] = world_done
            if self.cfg['episode_termination'] == 'early' or self.cfg['episode_termination'] == 'early_any':
                dones[key] = world_done or any(self.dones[key])
            elif self.cfg['episode_termination'] == 'early_all':
                dones[key] = world_done or all(self.dones[key])
            elif self.cfg['episode_termination'] == 'early_right':
                # early term only if at least one agent has reached right side of env
                # before that fixed episode length (world_done)
                agent_is_in_right_half = False
                for agent in self.teams[key]:
                    if agent.pose[X] < self.cfg['world_shape'][X]/2:
                        agent.no_new_coverage_steps = 0
                    else:
                        agent_is_in_right_half = True

                if agent_is_in_right_half:
                    dones[key] = any(self.dones[key])
                else:
                    dones[key] = world_done
            elif self.cfg['episode_termination'] == 'default':
                pass
            else:
                raise NotImplementedError("Unknown termination mode", self.cfg['episode_termination'])

        if self.cfg['operation_mode'] == "all":
            pass
        elif self.cfg['operation_mode'] == "greedy_only" or self.cfg['operation_mode'] == "adversary_only":
            dones[1] = dones[0]
        elif self.cfg['operation_mode'] == "coop_only":
            dones[0] = dones[1]
        else:
            raise NotImplementedError("Unknown operation_mode")
        done = any(dones.values()) # Currently we cannot run teams independently, all have to stop at the same time

        pose_map = np.zeros(self.map.shape + (len(self.teams),), dtype=np.uint8)
        for i, team in enumerate(self.teams.values()):
            for r in team:
                pose_map[r.pose[Y], r.pose[X], i] = 1
        global_state = np.concatenate([np.stack([self.map.map, self.map.coverage > 0], axis=-1), pose_map], axis=-1)
        state = {
            'agents': tuple([{
                'map': states[key][agent_i],
                'pos': self.teams[key][agent_i].pose
            } for key in self.teams.keys() for agent_i in range(self.cfg['n_agents'][key])]),
            'gso': self.compute_gso(0),
            'state': global_state
        }

        for key in self.teams.keys():
            if self.cfg['reward_type'] == 'semi_cooperative':
                pass
            elif self.cfg['reward_type'] == 'split_right':
                for agent_key in rewards[key].keys():
                    if self.teams[key][agent_key].pose[X] < self.cfg['world_shape'][X]/2:
                        rewards[key][agent_key] = 0
            else:
                raise NotImplementedError("Unknown reward type", self.cfg['reward_type'])
        if self.cfg['operation_mode'] == "all":
            pass
        elif self.cfg['operation_mode'] == "greedy_only":
            # copy all rewards from the greedy agent to the cooperative agents
            rewards[1] = {agent_key: sum(rewards[0].values()) for agent_key in rewards[1].keys()}
        elif self.cfg['operation_mode'] == "adversary_only":
            # The greedy agent's reward is the negative sum of all agent's rewards
            all_negative = -sum([sum(team_rewards.values()) for team_rewards in rewards.values()])
            rewards[0] = {agent_key: all_negative for agent_key in rewards[0].keys()}

            # copy all rewards from the greedy agent to the cooperative agents
            rewards[1] = {agent_key: sum(rewards[0].values()) for agent_key in rewards[1].keys()}
        elif self.cfg['operation_mode'] == "coop_only":
            # copy all rewards from the coop agent to the greedy agents
            rewards[0] = {agent_key: sum(rewards[1].values()) for agent_key in rewards[0].keys()}
        else:
            raise NotImplementedError("Unknown operation_mode")

        flattened_rewards = {}
        agent_index = 0
        for key in self.teams.keys():
            for r in rewards[key].values():
                flattened_rewards[agent_index] = r
                agent_index += 1
        info = {
            'current_global_coverage': self.map.get_coverage_fraction(),
            'coverable_area': self.map.get_coverable_area(),
            'rewards_teams': rewards,
            'rewards': flattened_rewards
        }
        return state, sum([sum(t.values()) for i, t in enumerate(rewards.values()) if not self.cfg['disabled_teams_step'][i]]), done, info

    def clear_patches(self, ax):
        [p.remove() for p in reversed(ax.patches)]
        [t.remove() for t in reversed(ax.texts)]

    def render_adjacency(self, A, team_id, ax, color='b', stepsize=1.0):
        A = A.copy()
        own_team_agents = [agent for agent in self.teams[team_id]]
        other_agents = [agent for other_team_id, team in self.teams.items() for agent in team if not team_id == other_team_id]
        all_agents = own_team_agents + other_agents
        for agent_id, agent in enumerate(all_agents):
            for connected_agent_id in np.arange(len(A)):
                if A[agent_id][connected_agent_id] > 0:
                    current_agent_pose = agent.prev_pose + (agent.pose - agent.prev_pose) * stepsize
                    other_agent = all_agents[connected_agent_id]
                    other_agent_pose = other_agent.prev_pose + (other_agent.pose - other_agent.prev_pose) * stepsize
                    ax.add_patch(patches.ConnectionPatch(
                        [current_agent_pose[X], current_agent_pose[Y]],
                        [other_agent_pose[X], other_agent_pose[Y]],
                        "data", edgecolor='g', facecolor='none', lw=1, ls=":", alpha=0.3
                    ))

                    A[connected_agent_id][agent_id] = 0  # don't draw same connection again

    def render_global_coverages(self, ax):
        if not hasattr(self, 'im_cov_global'):
            self.im_cov_global = ax.imshow(np.zeros(self.map.shape), vmin=0, vmax=100)
        all_team_colors = [(0, 0, 0, 0)] + [tuple(list(c) + [0.5]) for team_colors in self.teams_agents_color.values() for c in team_colors]
        coverage = self.map.coverage.copy()
        if self.cfg['map_mode'] == 'split_half_fixed':
            # mark coverage on left side as gray
            color_index_left_side = len(all_team_colors)
            all_team_colors += [(0, 0, 0, 0.5)] # gray
            xx, _ = np.meshgrid(
                np.arange(0, coverage.shape[X], 1),
                np.arange(0, coverage.shape[Y], 1)
            )
            coverage[(xx < coverage.shape[X]/2) & (coverage > 0)] = color_index_left_side

        self.im_cov_global.set_data(colors.ListedColormap(all_team_colors)(coverage))

    def render_local_coverages(self, ax):
        if not hasattr(self, 'im_robots'):
            self.im_robots = {}
            for team_key, team in self.teams.items():
                if self.cfg['disabled_teams_step'][team_key]:
                    continue
                self.im_robots[team_key] = []
                for _ in team:
                    self.im_robots[team_key].append(ax.imshow(np.zeros(self.map.shape), vmin=0, vmax=1, alpha=0.5))

        self.im_map.set_data(self.map_colormap(self.map.map))
        for (team_key, team), team_colors in zip(self.teams.items(), self.teams_agents_color.values()):
            if self.cfg['disabled_teams_step'][team_key]:
                continue
            team_im = self.im_robots[team_key]
            for (agent_i, agent), color, im in zip(enumerate(team), team_colors, team_im):
                im.set_data(colors.ListedColormap([(0, 0, 0, 0), color])(agent.coverage))

    def render_overview(self, ax, stepsize=1.0):
        if not hasattr(self, 'im_map'):
            ax.set_xticks([])
            ax.set_yticks([])
            self.im_map = ax.imshow(np.zeros(self.map.shape), vmin=0, vmax=3)

        self.im_map.set_data(self.map_colormap(self.map.map))
        for (team_key, team), team_colors in zip(self.teams.items(), self.teams_agents_color.values()):
            if self.cfg['disabled_teams_step'][team_key]:
                continue
            for (agent_i, agent), color in zip(enumerate(team), team_colors):
                rect_size = 1
                pose_microstep = agent.prev_pose + (agent.pose - agent.prev_pose)*stepsize
                rect = patches.Rectangle((pose_microstep[1] - rect_size / 2, pose_microstep[0] - rect_size / 2), rect_size, rect_size,
                                         linewidth=1, edgecolor=self.teams_colors[team_key], facecolor='none')
                ax.add_patch(rect)
                #ax.text(agent.pose[1]+1, agent.pose[0], f"{agent_i}", color=self.teams_colors[team_key], clip_on=True)

        #last_reward = sum([r.reward for r in self.robots.values()])
        #ax.set_title(
        #    f'Global coverage: {int(self.map.get_coverage_fraction()*100)}%\n'
        #    #f'Last reward (r): {last_reward:.2f}'
        #)

    def render_connectivity(self, ax, agent_id, K):
        if K <= 1:
            return

        for connected_agent_id in np.arange(self.cfg['n_agents'])[self.A[agent_id] == 1]:
            current_agent_pose = self.robots[agent_id].pose
            connected_agent_d_pose = self.robots[connected_agent_id].pose - current_agent_pose
            ax.add_patch(patches.Arrow(
                current_agent_pose[X],
                current_agent_pose[Y],
                connected_agent_d_pose[X],
                connected_agent_d_pose[Y],
                edgecolor='b',
                facecolor='none'
            ))
            self.render_connectivity(ax, connected_agent_id, K-1)

    def render(self, mode='human', stepsize=1.0):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(3, 3))
            self.ax_overview = self.fig.add_subplot(1, 1, 1, aspect='equal')

        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview, stepsize)
        #self.render_local_coverages(self.ax_overview)
        self.render_global_coverages(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview, stepsize=stepsize)

        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        return self.fig

class CoverageEnvExplAdv(CoverageEnv):
    def __init__(self, cfg):
        super().__init__(cfg)

    def render(self, interpreter_obs, mode='human', stepsize=1.0):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(6, 3))
            gs = self.fig.add_gridspec(ncols=2, nrows=1)
            gs.update(wspace=0, hspace=0)
            self.ax_overview = self.fig.add_subplot(gs[0])
            ax_expl = self.fig.add_subplot(gs[1])
            ax_expl.set_xticks([])
            ax_expl.set_yticks([])
            self.im_expl_cov = ax_expl.imshow(np.zeros((1, 1)), vmin=0, vmax=1)
            self.im_expl_map = ax_expl.imshow(np.zeros((1, 1)), vmin=0, vmax=1)

        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview, stepsize)
        self.render_global_coverages(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview, stepsize=stepsize)

        adv_coverage = interpreter_obs[0][0]
        cmap_own_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), list(self.teams_agents_color[0][0])+[0.5]])
        cmap_map = colors.ListedColormap([(0,0,0,0), (0,0,0,1)])  # free, obstacle, unknown
        self.im_expl_cov.set_data(cmap_own_cov(self.teams[0][0].to_abs_frame(adv_coverage)))
        self.im_expl_map.set_data(cmap_map(self.map.map))

        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        return self.fig

class CoverageEnvAdvDec(CoverageEnv):
    def __init__(self, cfg):
        super().__init__(cfg)

    def render(self, interpreter_obs, mode='human', stepsize=1.0):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(6, 3))
            gs = self.fig.add_gridspec(ncols=2, nrows=1)
            gs.update(wspace=0, hspace=0)
            self.ax_overview = self.fig.add_subplot(gs[0])
            ax_expl = self.fig.add_subplot(gs[1])
            ax_expl.set_xticks([])
            ax_expl.set_yticks([])
            self.im_expl_cov = ax_expl.imshow(np.zeros((1, 1)), vmin=0, vmax=1)
            self.im_expl_map = ax_expl.imshow(np.zeros((1, 1)), vmin=0, vmax=1)

        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview, stepsize)
        self.render_global_coverages(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview, stepsize=stepsize)

        cmap_own_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), list(self.teams_agents_color[0][0])+[0.5]])
        cmap_map = colors.ListedColormap([(0,0,0,0), (0,0,0,1)])  # free, obstacle, unknown
        self.im_expl_cov.set_data(cmap_own_cov(interpreter_obs[0][1]))
        #self.im_expl_map.set_data(cmap_map(self.map.map))
        self.im_expl_map.set_data(cmap_map(interpreter_obs[0][0]))

        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        return self.fig

class CoverageEnvSingleSaliency(CoverageEnv):
    def __init__(self, cfg):
        super().__init__(cfg)

    def render(self, interpreter_obs, interpr_index=0, mode='human', stepsize=1.0):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(6, 3))
            gs = self.fig.add_gridspec(ncols=3, nrows=1)
            gs.update(wspace=0, hspace=0)
            self.ax_overview = self.fig.add_subplot(gs[0])
            ax_expl_map = self.fig.add_subplot(gs[1])
            ax_expl_cov = self.fig.add_subplot(gs[2])
            ax_expl_map.set_xticks([])
            ax_expl_map.set_yticks([])
            ax_expl_cov.set_xticks([])
            ax_expl_cov.set_yticks([])
            self.im_expl_cov = ax_expl_cov.imshow(np.zeros((1, 1)), vmin=0, vmax=1)
            self.im_expl_map = ax_expl_map.imshow(np.zeros((1, 1)), vmin=0, vmax=1)

        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview, stepsize)
        self.render_global_coverages(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview, stepsize=stepsize)

        #cmap_own_cov = colors.LinearSegmentedColormap.from_list("cmap_cov", [(0, 0, 0, 0), list(self.teams_agents_color[0][0])+[0.5]])
        #cmap_map = colors.ListedColormap([(0,0,0,0), (0,0,0,1)])  # free, obstacle, unknown

        saliency_limits = (np.min(interpreter_obs), np.max(interpreter_obs))
        self.im_expl_cov.set_clim(saliency_limits[0], saliency_limits[1])
        self.im_expl_map.set_clim(saliency_limits[0], saliency_limits[1])
        self.im_expl_cov.set_data(interpreter_obs[interpr_index][:, :, 1])
        self.im_expl_map.set_data(interpreter_obs[interpr_index][:, :, 0])

        #self.im_expl_cov.set_data(cmap_own_cov(interpreter_obs[interpr_index][1]))
        #self.im_expl_map.set_data(cmap_map(self.map.map))
        #self.im_expl_map.set_data(cmap_map(interpreter_obs[interpr_index][0]))

        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        return self.fig

class CoverageEnvSaliency(CoverageEnv):
    def __init__(self, cfg):
        super().__init__(cfg)

    def render(self, mode='human', saliency_obs=None, interpreter_obs=None):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(constrained_layout=True, figsize=(16, 10))
            grid_spec = self.fig.add_gridspec(ncols=max(self.cfg['n_agents']) * 3,
                                              nrows=1 + 2 * len(self.cfg['n_agents']),
                                              height_ratios=[1] + [1, 1] * len(self.cfg['n_agents']))

            self.ax_overview = self.fig.add_subplot(grid_spec[0, :])

            self.ax_im_agent = {}
            for team_key, team in self.teams.items():
                self.ax_im_agent[team_key] = []
                for i in range(self.cfg['n_agents'][team_key]):
                    self.ax_im_agent[team_key].append({})
                    for j, col_id in enumerate(['map', 'coverage']):
                        self.ax_im_agent[team_key][i][col_id] = {}
                        for k, row_id in enumerate(['obs', 'sal']):
                            ax = self.fig.add_subplot(grid_spec[j + 1 + team_key * 2, i * 3 + k])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            self.ax_im_agent[team_key][i][col_id][row_id] = {'ax': ax, 'im': None}
                        self.ax_im_agent[team_key][i][col_id]['sal']['im'] = self.ax_im_agent[team_key][i][col_id]['sal']['ax'].imshow(np.zeros((1, 1)), vmin=-5, vmax=5)
                        #self.ax_im_agent[team_key][i][col_id]['int']['im'] = self.ax_im_agent[team_key][i][col_id]['int']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=1)
                    self.ax_im_agent[team_key][i]['map']['obs']['im'] = self.ax_im_agent[team_key][i]['map']['obs']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=3)
                    self.ax_im_agent[team_key][i]['coverage']['obs']['im'] = self.ax_im_agent[team_key][i]['coverage']['obs']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=1, alpha=0.3)

        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview)

        if saliency_obs is not None:
            saliency_limits = (np.min(saliency_obs), np.max(saliency_obs))
        img_map_id = 0
        for team_key, team in self.teams.items():
            for i, robot in enumerate(team):

                self.ax_im_agent[team_key][i]['map']['obs']['im'].set_data(
                    self.map_colormap(robot.to_abs_frame(robot.state[..., 0])))
                this_coverage_colormap = colors.ListedColormap([(0, 0, 0, 0), self.teams_agents_color[team_key][i]])
                self.ax_im_agent[team_key][i]['coverage']['obs']['im'].set_data(
                    this_coverage_colormap(robot.to_abs_frame(robot.state[..., 1])))

                if saliency_obs is not None:
                    self.ax_im_agent[team_key][i]['map']['sal']['im'].set_data(
                        robot.to_abs_frame(saliency_obs[img_map_id][..., 0]))
                    self.ax_im_agent[team_key][i]['map']['sal']['im'].set_clim(saliency_limits[0], saliency_limits[1])
                    self.ax_im_agent[team_key][i]['coverage']['sal']['im'].set_data(
                        robot.to_abs_frame(saliency_obs[img_map_id][..., 1]))
                    self.ax_im_agent[team_key][i]['coverage']['sal']['im'].set_clim(saliency_limits[0], saliency_limits[1])

                if interpreter_obs is not None:
                    self.ax_im_agent[team_key][i]['map']['int']['im'].set_data(
                        robot.to_abs_frame(interpreter_obs[img_map_id][1]))
                    self.ax_im_agent[team_key][i]['coverage']['int']['im'].set_data(
                        robot.to_abs_frame(interpreter_obs[img_map_id][0]))

                img_map_id += 1

                self.ax_im_agent[team_key][i]['map']['obs']['ax'].set_title(
                    f'{i}') #\nc: {0:.2f}\nr: {robot.reward:.2f}')

        # self.render_connectivity(self.ax_overview, 0, 3)
        self.fig.canvas.draw()
        return self.fig
