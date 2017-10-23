from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from pysc2.env.environment import TimeStep
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES
import sys
import numpy as np


def timestep_to_gym_step(timestep: TimeStep):
    obs = timestep.observation["screen"][SCREEN_FEATURES.player_relative.index]
    obs = obs[..., np.newaxis]
    done = False
    info = None
    return obs, timestep.reward, done, info


class SC2AtariEnv:
    def __init__(self, sc_env, dim,
            id=np.random.randint(1000),
            verbose_freq=1,
            agg_n_episodes=100,
            reselect_army_freq=5
    ):
        """
        :param sc_env: SC2Env
        :param dim: screen dimension of sc2_env
        :param id: "name" for this environment
        :param verbose_freq: print results every n episode, 0 is no printing
        :param agg_n_episodes: print results of this many last episodes
        :param reselect_army_freq: reselect army every n timesteps,
            is needed if get new units like in DefeatRoaches,
            too frequent selecting might hurt the scores little because one timestep is wasted
        """
        self.sc2_env = sc_env
        self.dim = dim
        self.verbose_freq = verbose_freq

        self.action_space = Discrete(dim ** 2)
        self.observation_space = Box(
            low=0, high=SCREEN_FEATURES.player_relative.scale, shape=[dim, dim, 1]
        )
        self.rolling_episode_score = np.zeros(agg_n_episodes, dtype=np.float32)
        self.agg_n_episodes = agg_n_episodes
        self.id = id
        self.attack_move_action_id = [
            k for k in actions.FUNCTIONS
            if k.name == 'Attack_screen'
        ][0].id
        self.reselect_army_freq = reselect_army_freq
        self.step_counter = 0
        self.episode_counter = 0

    def _summarise_episode(self, timestep):
        episode_score = timestep.observation["score_cumulative"][0]
        self.rolling_episode_score[self.episode_counter % self.agg_n_episodes] = episode_score
        self.episode_counter += 1

        if self.verbose_freq > 0 and self.episode_counter % self.verbose_freq == 0:
            n_last = min(self.episode_counter, self.agg_n_episodes)
            r = self.rolling_episode_score[:n_last]
            print("env: %d, episode %d, score: %.1f, last %d - avg %.1f max %.1f" % (
                self.id, self.episode_counter, episode_score, n_last, r.mean(), r.max()
            ))
            sys.stdout.flush()

    def _step_with_attack_move(self, action):
        coords = np.unravel_index(action, (self.dim,) * 2)
        # Note: at least pysc2 1.2 needs reversed coordinates
        action = [actions.FunctionCall(self.attack_move_action_id, [[0], coords[::-1]])]
        try:
            timestep = self.sc2_env.step(action)[0]
        except ValueError:
            # if attack move is not available select full rectangle and try again
            self._reselect_army()
            timestep = self.sc2_env.step(action)[0]

        return timestep

    def _reselect_army(self):
        select_army_op = actions.FunctionCall(actions.FUNCTIONS.select_army.id, [[0]])
        # we are just selecting but not returning this observation
        self.sc2_env.step([select_army_op])

    def step(self, action):
        """
        :param action: int, representing the coordinates in flat space
        :return: TimeStep which is the observation after step
        """
        if self.reselect_army_freq > 0 and self.step_counter % self.reselect_army_freq == 0:
            self._reselect_army()

        timestep = self._step_with_attack_move(action)

        if timestep.last():
            self._summarise_episode(timestep)

        self.step_counter += 1
        return timestep_to_gym_step(timestep)

    def reset(self):
        timesteps = self.sc2_env.reset()
        return timestep_to_gym_step(timesteps[0])[0]

    def close(self):
        self.sc2_env.close()
