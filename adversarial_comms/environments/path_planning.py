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
    'world_shape': [12, 12],
    'state_size': 24,
    'max_episode_len': 50,
    "n_agents": [8],
    "disabled_teams_step": [False],
    "disabled_teams_comms": [False],
    'communication_range': 5.0,
    'ensure_connectivity': True,
    'position_mode': 'random', # random or fixed
    'agents': {
        'visibility_distance': 3,
        'relative_coord_frame': True
    }
}

X = 1
Y = 0

class Dir(Enum):
    RIGHT  = 0
    LEFT   = 1
    UP     = 2
    DOWN   = 3

class WorldMap():
    def __init__(self, shape, mode):
        self.shape = shape
        self.mode = mode
        self.reset()

    def reset(self):
        if self.mode == "traffic":
            self.map = np.array([
                [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
            ])
        elif self.mode == "warehouse":
            self.map = np.zeros(self.shape, dtype=np.uint8)
            for y in range(1, self.shape[Y]-1, 2):
                for x in range(1, self.shape[X]-1, 6):
                    self.map[y:y+1,x:x+4] = True
        else:
            raise NotImplementedError

class Action(Enum):
    NOP         = 0
    MOVE_RIGHT  = 1
    MOVE_LEFT   = 2
    MOVE_UP     = 3
    MOVE_DOWN   = 4

class Robot():
    def __init__(self,
                 world,
                 agent_observability_radius,
                 state_size,
                 coordinate_frame_is_local):
        self.world = world
        self.state_size = state_size
        self.coordinate_frame_is_local = coordinate_frame_is_local
        self.agent_observability_radius = agent_observability_radius
        self.reset([0, 0], [0, 0])

    def reset(self, pose, goal):
        self.pose = np.array(pose, dtype=np.int)
        self.prev_pose = self.pose.copy()
        self.goal = np.array(goal, dtype=np.int)

    def step(self, action):
        action = Action(action)

        delta_pose = {
            Action.MOVE_RIGHT:  [ 0,  1],
            Action.MOVE_LEFT:   [ 0, -1],
            Action.MOVE_UP:     [-1,  0],
            Action.MOVE_DOWN:   [ 1,  0],
            Action.NOP:         [ 0,  0]
        }[action]

        def is_occupied(p):
            for team_key, team in self.world.teams.items():
                if self.world.cfg['disabled_teams_step'][team_key]:
                    continue
                for o in team:
                    if p[X] == o.pose[X] and p[Y] == o.pose[Y] and o is not self:
                        return True
            return False

        is_valid_pose = lambda p: all([p[c] >= 0 and p[c] < self.world.map.shape[c] for c in [Y, X]])
        is_obstacle = lambda p: self.world.map.map[p[Y]][p[X]]

        self.prev_pose = self.pose.copy()
        desired_pos = self.pose + delta_pose
        if is_valid_pose(desired_pos) and not is_occupied(desired_pos) and not is_obstacle(desired_pos):
            self.pose = desired_pos

    def update_state(self):
        pose_map = np.zeros(self.world.map.shape, dtype=np.uint8)
        for team in self.world.teams.values():
            for r in team:
                if not r is self and np.sum((r.pose - self.pose)**2) <= self.agent_observability_radius**2:
                    pose_map[r.pose[Y], r.pose[X]] = 2
        pose_map[self.pose[Y], self.pose[1]] = 1

        goal_map = np.zeros(self.world.map.shape, dtype=np.bool)
        cy, cx = self.goal - self.pose
        if abs(cx) < self.state_size/2 and abs(cy) < self.state_size/2:
            goal_map[self.goal[Y], self.goal[X]] = True
        else:
            u = max(abs(cx), abs(cy))
            p_sq = np.round(self.pose + int(self.state_size / 2) * np.array([cy / u, cx / u])).astype(np.int)
            goal_map[p_sq[Y], p_sq[X]] = True

        self.state = np.stack([self.to_coordinate_frame(self.world.map.map, 1), self.to_coordinate_frame(goal_map, 0), self.to_coordinate_frame(pose_map, 0)], axis=-1).astype(np.uint8)
        done = all(self.pose == self.goal)
        return self.state, done

    def to_coordinate_frame(self, m, fill=0):
        if self.coordinate_frame_is_local:
            half_state_shape = np.array([self.state_size/2]*2, dtype=np.int)
            padded = np.pad(m,([half_state_shape[Y]]*2,[half_state_shape[X]]*2), mode='constant', constant_values=fill)
            return padded[self.pose[Y]:self.pose[Y] + self.state_size, self.pose[X]:self.pose[X] + self.state_size]
        else:
            return m

class PathPlanningEnv(gym.Env, EzPickle):
    def __init__(self, env_config):
        EzPickle.__init__(self)
        self.seed()

        self.cfg = copy.deepcopy(DEFAULT_OPTIONS)
        self.cfg.update(env_config)

        self.fig = None
        self.map_colormap = colors.ListedColormap(['white', 'black', 'gray'])  # free, obstacle, unknown

        hsv = np.ones((sum(self.cfg['n_agents']), 3))
        hsv[..., 0] = np.linspace(0, 1, sum(self.cfg['n_agents']) + 1)[:-1]
        self.teams_agents_color = {}
        current_index = 0
        for i, n_agents in enumerate(self.cfg['n_agents']):
            self.teams_agents_color[i] = colors.hsv_to_rgb(hsv[current_index:current_index+n_agents])
            current_index += n_agents

        hsv = np.ones((len(self.cfg['n_agents']), 3))
        hsv[..., 0] = np.linspace(0, 1, len(self.cfg['n_agents']) + 1)[:-1]
        self.teams_colors = ['r', 'b'] #colors.hsv_to_rgb(hsv)

        n_all_agents = sum(self.cfg['n_agents'])
        self.observation_space = spaces.Dict({
            'agents': spaces.Tuple((
                spaces.Dict({
                    'map': spaces.Box(0, np.inf, shape=(self.cfg['state_size'], self.cfg['state_size'], 3)),
                    'pos': spaces.Box(low=np.array([0,0]), high=np.array([self.cfg['world_shape'][Y], self.cfg['world_shape'][X]]), dtype=np.int),
                }),
            )*n_all_agents), # Do not add this as additional dimension of map and pos since this way it is easier to handle in the model
            'gso': spaces.Box(-np.inf, np.inf, shape=(n_all_agents, n_all_agents)),
            'state': spaces.Box(low=0, high=3, shape=self.cfg['world_shape']+[sum(self.cfg['n_agents'])]),
        })
        self.action_space = spaces.MultiDiscrete([5]*sum(self.cfg['n_agents']))

        self.map = WorldMap(self.cfg['world_shape'], self.cfg['world_mode'])

        self.teams = {
            i: [
                Robot(
                    self,
                    self.cfg['agents']['visibility_distance'],
                    self.cfg['state_size'],
                    self.cfg['agents']['relative_coord_frame']
                ) for _ in range(n_agents)
            ] for i, n_agents in enumerate(self.cfg['n_agents'])
        }

        self.reset()

    def seed(self, seed=None):
        self.random_state, seed_agents = seeding.np_random(seed)
        return [seed_agents]

    def reset(self):
        self.timestep = 0
        self.dones = {key: [False for _ in team] for key, team in self.teams.items()}
        self.map.reset()

        def sample_random_pos():
            x = self.random_state.randint(0, self.map.shape[X])
            y = self.random_state.randint(0, self.map.shape[Y])
            return np.array([y, x])

        def sample_valid_random_pos(up_to=None):
            def get_agents():
                return [o for team in self.teams.values() for o in team][:up_to]
            def is_occupied(p):
                return any([all(p == o.pose) for o in get_agents()])
            def is_other_goal(p):
                return any([all(p == o.goal) for o in get_agents()])
            is_obstacle = lambda p: self.map.map[p[Y]][p[X]]

            pose_seed = sample_random_pos()
            while is_obstacle(pose_seed) or is_occupied(pose_seed) or is_other_goal(pose_seed):
                pose_seed = sample_random_pos()
            return pose_seed

        agent_index = 0
        for team_key, team in self.teams.items():
            if self.cfg['disabled_teams_step'][team_key]:
                continue
            for agent in team:
                agent.reset(sample_valid_random_pos(agent_index), sample_valid_random_pos(agent_index))
                agent_index += 1

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
            for j, agent in enumerate(team):
                if not self.cfg['disabled_teams_step'][i]: # and not self.dones[i][j]:
                    agent.step(actions[action_index])
                action_index += 1

        states, rewards = {}, {}
        for team_key, team in self.teams.items():
            states[team_key] = []
            rewards[team_key] = {}
            for i, agent in enumerate(team):
                state, done = agent.update_state()
                states[team_key].append(state)
                rewards[team_key][i] = 1 if done else 0  # reward while at goal, incentives moving as quickly as possible
                if done:
                    self.dones[team_key][i] = True

        if self.cfg['reward_type'] == 'local':
            pass
        elif self.cfg['reward_type'] == 'greedy_only':
            rewards[1] = {agent_key: sum(rewards[0].values()) for agent_key in rewards[1].keys()}
        elif self.cfg['reward_type'] == 'coop_only':
            rewards[0] = {agent_key: sum(rewards[1].values()) for agent_key in rewards[0].keys()}
        else:
            raise NotImplementedError("Unknown reward type", self.cfg['reward_type'])

        done = self.timestep == self.cfg['max_episode_len'] # or all(self.dones[1])

        global_state = np.stack([self.map.map.copy() for _ in range(sum(self.cfg['n_agents']))], axis=-1).astype(np.uint8)
        global_state_layer = 0
        for team in self.teams.values():
            for r in team:
                global_state[r.pose[Y], r.pose[X], global_state_layer] = 2
                global_state[r.goal[Y], r.goal[X], global_state_layer] = 3
                global_state_layer += 1

        state = {
            'agents': tuple([{
                'map': states[key][agent_i],
                'pos': self.teams[key][agent_i].pose
            } for key in self.teams.keys() for agent_i in range(self.cfg['n_agents'][key])]),
            'gso': self.compute_gso(0),
            'state': global_state
        }

        flattened_rewards = {}
        agent_index = 0
        for key in self.teams.keys():
            for r in rewards[key].values():
                flattened_rewards[agent_index] = r
                agent_index += 1
        info = {
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
                        "data", edgecolor='g', facecolor='none', lw=1, ls=":"
                    ))

                    A[connected_agent_id][agent_id] = 0  # don't draw same connection again

    def render_overview(self, ax, stepsize=1.0):
        if not hasattr(self, 'im_map'):
            ax.set_xticks([])
            ax.set_yticks([])
            self.im_map = ax.imshow(np.zeros(self.map.shape), vmin=0, vmax=1)

        self.im_map.set_data(self.map_colormap(self.map.map))
        agent_i = 0
        for (team_key, team) in self.teams.items():
            if self.cfg['disabled_teams_step'][team_key]:
                continue
            for agent in team:
                rect_size = 1
                pose_microstep = agent.prev_pose + (agent.pose - agent.prev_pose)*stepsize
                rect = patches.Rectangle((pose_microstep[1] - rect_size / 2, pose_microstep[0] - rect_size / 2), rect_size, rect_size,
                                         linewidth=1, edgecolor=self.teams_colors[team_key], facecolor='none')
                ax.add_patch(rect)
                ax.text(pose_microstep[1]-0.45, pose_microstep[0], f"{agent_i}", color=self.teams_colors[team_key])
                agent_i += 1

        #ax.set_title(
        #    f'Global coverage: {int(self.map.get_coverage_fraction()*100)}%\n'
        #)

    def render_goals(self, ax):
        agent_i = 0
        for team_key, team in self.teams.items():
            if self.cfg['disabled_teams_step'][team_key]:
                continue
            for agent in team:
                rect = patches.Circle((agent.goal[1], agent.goal[0]), 0.1,
                                      linewidth=1, facecolor=self.teams_colors[team_key])
                ax.add_patch(rect)
                ax.text(agent.goal[1] - 0.45, agent.goal[0] + 0.5, f"{agent_i}", color=self.teams_colors[team_key])
                agent_i += 1

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

    def render_future_steps(self, future_steps, ax, stepsize=1.0):
        if future_steps is None:
            return

        current_agent_i = 0
        for team_key, team in self.teams.items():
            for agent in team:
                if current_agent_i not in future_steps:
                    current_agent_i += 1
                    continue

                previous_agent_pose = future_steps[current_agent_i][0].copy()
                for i, current_pos in enumerate(future_steps[current_agent_i][1:]):
                    if i == len(future_steps[current_agent_i][1:])-1:
                        current_pos = previous_agent_pose + (
                                    future_steps[current_agent_i][-1] - previous_agent_pose) * stepsize
                        ax.add_patch(
                            patches.Rectangle((current_pos[1] - 1/2, current_pos[0] - 1/2), 1, 1,
                                                       linewidth=1, edgecolor=self.teams_colors[team_key],
                                                       facecolor='none', ls=":")
                         )
                    ax.add_patch(patches.ConnectionPatch(
                        [previous_agent_pose[X], previous_agent_pose[Y]],
                        [current_pos[X], current_pos[Y]],
                        "data", edgecolor=self.teams_colors[team_key], facecolor='none', lw=2
                    ))

                    previous_agent_pose = current_pos.copy()

                current_agent_i += 1

    def render(self, mode='human', future_steps=None, stepsize=1.0):
        if self.fig is None:
            self.fig = plt.figure(figsize=(3, 3))
            self.ax_overview = self.fig.add_subplot(1, 1, 1, aspect='equal')

        self.clear_patches(self.ax_overview)
        self.render_future_steps(future_steps, self.ax_overview, stepsize)
        self.render_overview(self.ax_overview, stepsize)
        self.render_goals(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview, stepsize=stepsize)

        self.fig.canvas.draw()
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        return self.fig

class PathPlanningEnvSaliency(PathPlanningEnv):
    def __init__(self, cfg):
        super().__init__(cfg)

    def render(self, mode='human', saliency_obs=None, saliency_pos=None):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(constrained_layout=True, figsize=(16, 10))
            grid_spec = self.fig.add_gridspec(ncols=max(self.cfg['n_agents']) * 2,
                                              nrows=1 + 3 * len(self.cfg['n_agents']),
                                              height_ratios=[1] + [1, 1, 1] * len(self.cfg['n_agents']))

            self.ax_overview = self.fig.add_subplot(grid_spec[0, :])

            self.ax_im_agent = {}
            for team_key, team in self.teams.items():
                self.ax_im_agent[team_key] = []
                for i in range(self.cfg['n_agents'][team_key]):
                    self.ax_im_agent[team_key].append({})
                    for j, col_id in enumerate(['map', 'goal', 'pos']):
                        self.ax_im_agent[team_key][i][col_id] = {}
                        for k, row_id in enumerate(['obs', 'sal']):
                            ax = self.fig.add_subplot(grid_spec[j + 1 + team_key * 2, i * 2 + k])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            self.ax_im_agent[team_key][i][col_id][row_id] = {'ax': ax, 'im': None}
                        self.ax_im_agent[team_key][i][col_id]['sal']['im'] = \
                        self.ax_im_agent[team_key][i][col_id]['sal']['ax'].imshow(
                            np.zeros((1, 1)), vmin=-5, vmax=5)
                    self.ax_im_agent[team_key][i]['map']['obs']['im'] = self.ax_im_agent[team_key][i]['map']['obs']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=3)
                    self.ax_im_agent[team_key][i]['goal']['obs']['im'] = self.ax_im_agent[team_key][i]['goal']['obs']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=1)
                    self.ax_im_agent[team_key][i]['pos']['obs']['im'] = self.ax_im_agent[team_key][i]['pos']['obs']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=1)


        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview)

        if saliency_obs is not None:
            saliency_limits = (np.min(saliency_obs), np.max(saliency_obs))
        saliency_map_id = 0
        for team_key, team in self.teams.items():
            for i, robot in enumerate(team):
                self.ax_im_agent[team_key][i]['map']['obs']['im'].set_data(self.map_colormap(robot.state[..., 0]))
                self.ax_im_agent[team_key][i]['goal']['obs']['im'].set_data(self.map_colormap(robot.state[..., 1]))
                self.ax_im_agent[team_key][i]['pos']['obs']['im'].set_data(self.map_colormap(robot.state[..., 2]))

                if saliency_obs is not None:
                    self.ax_im_agent[team_key][i]['map']['sal']['im'].set_data(saliency_obs[saliency_map_id][..., 0])
                    self.ax_im_agent[team_key][i]['map']['sal']['im'].set_clim(saliency_limits[0], saliency_limits[1])
                    self.ax_im_agent[team_key][i]['coverage']['sal']['im'].set_data(saliency_obs[saliency_map_id][..., 1])
                    self.ax_im_agent[team_key][i]['coverage']['sal']['im'].set_clim(saliency_limits[0], saliency_limits[1])

                    saliency_map_id += 1

                if False:  # saliency_pos is not None:
                    print("T", saliency_pos[i][2:].numpy())
                    self.ax_im_agent[i]['map']['sal']['ax'].set_title(
                        f'{saliency_pos[i][0]:.2f}\n{saliency_pos[i][1]:.2f}\n{np.mean(saliency_pos[i][2:].numpy()):.2f}')

                #self.ax_im_agent[team_key][i]['map']['obs']['ax'].set_title(
                #    f'{i}\nc: {0:.2f}\nr: {robot.reward:.2f}')

        # self.render_connectivity(self.ax_overview, 0, 3)
        self.fig.canvas.draw()
        return self.fig

class PathPlanningEnvOverview(PathPlanningEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.map_colormap = colors.ListedColormap(['white', 'black', 'blue', 'red', 'green'])  # free, obstacle, pos, goal

    def render(self, mode='human'):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(constrained_layout=True, figsize=(16, 10))
            grid_spec = self.fig.add_gridspec(ncols=max(self.cfg['n_agents']),
                                              nrows=1 + len(self.cfg['n_agents']),
                                              height_ratios=[1] + [1] * len(self.cfg['n_agents']))

            self.ax_overview = self.fig.add_subplot(grid_spec[0, :])

            self.ax_im_agent = {}
            for team_key, team in self.teams.items():
                self.ax_im_agent[team_key] = []
                for i in range(self.cfg['n_agents'][team_key]):
                    self.ax_im_agent[team_key].append({})
                    for j, col_id in enumerate(['overview']):
                        self.ax_im_agent[team_key][i][col_id] = {}
                        for k, row_id in enumerate(['obs']):
                            ax = self.fig.add_subplot(grid_spec[j + 1 + team_key , i + k])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            self.ax_im_agent[team_key][i][col_id][row_id] = {'ax': ax, 'im': None}
                    self.ax_im_agent[team_key][i]['overview']['obs']['im'] = self.ax_im_agent[team_key][i]['overview']['obs']['ax'].imshow(np.zeros((1, 1)), vmin=0, vmax=4)


        self.clear_patches(self.ax_overview)
        self.render_overview(self.ax_overview)
        self.render_goals(self.ax_overview)
        A = self.compute_gso(0)
        self.render_adjacency(A, 0, self.ax_overview)

        saliency_map_id = 0
        for team_key, team in self.teams.items():
            for i, robot in enumerate(team):
                state = robot.state[..., 0].copy().astype(np.uint8) # map
                state[robot.state[..., 1]==1] = 2
                state[robot.state[..., 2]==1] = 3
                state[robot.state[..., 2]==2] = 4
                #state[robot.state[..., 2]] = 3
                self.ax_im_agent[team_key][i]['overview']['obs']['im'].set_data(self.map_colormap(state))

                self.ax_im_agent[team_key][i]['overview']['obs']['ax'].set_title(
                    f'{i}') #\nc: {0:.2f}\nr: {robot.reward:.2f}')

        # self.render_connectivity(self.ax_overview, 0, 3)
        self.fig.canvas.draw()
        return self.fig
