import pommerman
from pommerman.agents import BaseAgent
import torch
import numpy as np
from models import create_policy
import gym
import random

DEFAULT_FEATURE_CONFIG = {
    'recode_agents': True,
    'compact_powerups': True,
    'compact_structure': True,
    'rescale': True,
}


def make_np_float(feature):
    return np.array(feature).astype(np.float32)


def _rescale(x):
    return (x - 0.5) * 2.0


def get_feature_channels(config):
    num_channels = 15
    if config['recode_agents']:
        num_channels -= 2
    if config['compact_powerups']:
        num_channels -= 2
    if config['compact_structure']:
        num_channels -= 2
    return num_channels


def get_unflat_obs_space(channels=15, board_size=11, rescale=True):
    min_board_obs = np.zeros((channels, board_size, board_size))
    max_board_obs = np.ones_like(min_board_obs)
    min_other_obs = np.zeros(3)
    max_other_obs = np.ones_like(min_other_obs)

    if rescale:
        min_board_obs = _rescale(min_board_obs)
        max_board_obs = _rescale(max_board_obs)
        min_other_obs = _rescale(min_other_obs)
        max_other_obs = _rescale(max_other_obs)

    return gym.spaces.Tuple([
        gym.spaces.Box(min_board_obs, max_board_obs),
        gym.spaces.Box(min_other_obs, max_other_obs)])


def featurize(obs, agent_id, config):
    max_item = pommerman.constants.Item.Agent3.value

    ob = obs["board"]
    ob_bomb_blast_strength = obs["bomb_blast_strength"].astype(np.float32) / pommerman.constants.AGENT_VIEW_SIZE
    ob_bomb_life = obs["bomb_life"].astype(np.float32) / pommerman.constants.DEFAULT_BOMB_LIFE

    # print('obs board: ', ob, type(ob), np.array(ob).shape)

    # one hot encode the board items
    ob_values = max_item + 1
    ob_hot = np.eye(ob_values)[ob]
    # print('max items: ', max_item)
    # print('obs hot board: ', ob_hot, type(ob_hot), np.array(ob_hot).shape)

    # replace agent item channels with friend, enemy, self channels
    if config['recode_agents']:
        self_value = pommerman.constants.Item.Agent0.value + agent_id
        enemies = np.logical_and(ob >= pommerman.constants.Item.Agent0.value, ob != self_value)
        self = (ob == self_value)
        friends = (ob == pommerman.constants.Item.AgentDummy.value)
        ob_hot[:, :, 9] = friends.astype(np.float32)
        ob_hot[:, :, 10] = self.astype(np.float32)
        ob_hot[:, :, 11] = enemies.astype(np.float32)
        ob_hot = np.delete(ob_hot, np.s_[12::], axis=2)

    if config['compact_powerups']:
        # replace powerups with single channel
        powerup = ob_hot[:, :, 6] * 0.5 + ob_hot[:, :, 7] * 0.66667 + ob_hot[:, :, 8]
        ob_hot[:, :, 6] = powerup
        ob_hot = np.delete(ob_hot, [7, 8], axis=2)

    # replace bomb item channel with bomb life
    ob_hot[:, :, 3] = ob_bomb_life

    if config['compact_structure']:
        ob_hot[:, :, 0] = 0.5 * ob_hot[:, :, 0] + ob_hot[:, :, 5]  # passage + fog
        ob_hot[:, :, 1] = 0.5 * ob_hot[:, :, 2] + ob_hot[:, :, 1]  # rigid + wood walls
        ob_hot = np.delete(ob_hot, [2], axis=2)
        # replace former fog channel with bomb blast strength
        ob_hot[:, :, 5] = ob_bomb_blast_strength
    else:
        # insert bomb blast strength next to bomb life
        ob_hot = np.insert(ob_hot, 4, ob_bomb_blast_strength, axis=2)

    self_ammo = make_np_float([obs["ammo"]])
    self_blast_strength = make_np_float([obs["blast_strength"]])
    self_can_kick = make_np_float([obs["can_kick"]])
    #print("ammo: {0}, blast_strength: {1},can_kick: {2}".format(self_ammo, self_blast_strength, self_can_kick))
    ob_hot = ob_hot.transpose((2, 0, 1))  # PyTorch tensor layout compat

    if config['rescale']:
        ob_hot = _rescale(ob_hot)
        self_ammo = _rescale(self_ammo / 10)
        self_blast_strength = _rescale(self_blast_strength / pommerman.constants.AGENT_VIEW_SIZE)
        self_can_kick = _rescale(self_can_kick)

    return np.concatenate([
        np.reshape(ob_hot, -1), self_ammo, self_blast_strength, self_can_kick])


class TjuAgent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(TjuAgent, self).__init__( *args, **kwargs)
        torch.set_num_threads(1)
        device = torch.device("cpu")
        model_path0 = './best_models/0.pt'
        model_path1 = './best_models/1.pt'
        # model_path1 = './best_models/1.pt'
        model_path2 = './best_models/2.pt'
        model_path3 = './best_models/3.pt'
        # model_path3 = './best_models/3.pt'
        bs = 11
        channels = get_feature_channels(DEFAULT_FEATURE_CONFIG)
        obs_unflat = get_unflat_obs_space(channels, bs, DEFAULT_FEATURE_CONFIG['rescale'])
        min_flat_obs = np.concatenate([obs_unflat.spaces[0].low.flatten(), obs_unflat.spaces[1].low])
        max_flat_obs = np.concatenate([obs_unflat.spaces[0].high.flatten(), obs_unflat.spaces[1].high])
        self.recent_states = []
        self.list_len = 20
        self.observation_space = gym.spaces.Box(min_flat_obs, max_flat_obs)
        self.action_space = gym.spaces.Discrete(6)
        self.state_dict0, self.ob_rms0 = torch.load(model_path0)
        self.actor_critic0 = create_policy(self.observation_space,
                                           self.action_space,
                                           name='pomm',
                                           nn_kwargs={
                                               'batch_norm': True,
                                               'recurrent': False,
                                               'hidden_size': 512,
                                           },
                                           train=False)
        self.state_dict1, self.ob_rms1 = torch.load(model_path1)
        self.actor_critic1 = create_policy(self.observation_space,
                                           self.action_space,
                                           name='pomm',
                                           nn_kwargs={
                                               'batch_norm':True,
                                               'recurrent':False,
                                               'hidden_size':512,
                                           },
                                           train=False)
        self.state_dict2, self.ob_rms2 = torch.load(model_path2)
        self.actor_critic2 = create_policy(self.observation_space,
                                           self.action_space,
                                           name='pomm',
                                           nn_kwargs={
                                               'batch_norm': True,
                                               'recurrent': False,
                                               'hidden_size': 512,
                                           },
                                           train=False)
        self.state_dict3, self.ob_rms3 = torch.load(model_path3)
        self.actor_critic3 = create_policy(self.observation_space,
                                           self.action_space,
                                           name='pomm',
                                           nn_kwargs={
                                               'batch_norm': True,
                                               'recurrent': False,
                                               'hidden_size': 512,
                                           },
                                           train=False)
        self.actor_critic0.load_state_dict(self.state_dict0)
        self.actor_critic0.to(device)
        self.actor_critic1.load_state_dict(self.state_dict1)
        self.actor_critic1.to(device)
        self.actor_critic2.load_state_dict(self.state_dict2)
        self.actor_critic2.to(device)
        self.actor_critic3.load_state_dict(self.state_dict3)
        self.actor_critic3.to(device)

    def act(self, obs, action_space):
        agent_id = self._get_id(obs)

        self.recent_states.append(featurize(obs, agent_id, DEFAULT_FEATURE_CONFIG))
        if len(self.recent_states) > self.list_len:
            del self.recent_states[0]
        stateset = self._toset(self.recent_states)

        obs = self._preprocess(obs, agent_id)
        #print('current id:', agent_id)
        #assert agent_id in [0, 1, 2, 3]
        with torch.no_grad():
            if agent_id == 0:
                #print('using policy 0')
                value, action, _ = self.actor_critic0.act(obs, deterministic=True)
            elif agent_id == 1:
                #print('using policy 1')
                value, action, _ = self.actor_critic1.act(obs, deterministic=True)
            elif agent_id == 2:
                #print('using policy 2')
                value, action, _ = self.actor_critic2.act(obs, deterministic=True)
            elif agent_id == 3:
                #print('using policy 3')
                value, action, _ = self.actor_critic3.act(obs, deterministic=True)
        #print(len(stateset))
        if len(self.recent_states) == self.list_len and len(stateset) <= 3:
            action = [[random.randint(1, 5)]]

        return int(np.array(action)[0][0])

    def _get_id(self, obs):
        teammate = obs["teammate"]
        teammate_value = teammate.value
        #assert teammate_value in [10, 11, 12, 13]
        teammate_id = teammate_value - 10
        if teammate_id == 0:
            return 2
        elif teammate_id == 1:
            return 3
        elif teammate_id == 2:
            return 0
        elif teammate_id == 3:
            return 1

    def _preprocess(self, obs, agent_id):
        return torch.from_numpy(np.array([featurize(obs, agent_id, DEFAULT_FEATURE_CONFIG)])).float()

    def _toset(self, list):
        s = set()
        for state in list:
            s.add(tuple(state))
        return s

