'''
dqn related functions
'''

import torch
from torch import nn
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TaxiFeaturesExtractor(BaseFeaturesExtractor):
    '''
    Taxi Features Extractor
    the features extractor will take all of what the agent observes and process with the following steps:
    1. get the map, fuel, passengers, and taxi position as tensors
    2. embed and flatten the map
    3. normalize the values
    4. add label encoding for the fuel, passengers, and taxi position
    5. add position encoding for the map
    6. run the input through the transformer encoder
    '''

    network: nn.Module

    def __init__(self, observation_space: dict, features_dim: int = 32):
        super().__init__(observation_space, features_dim)

        # TODO:add the network

    def preprocess_obs(self, obs):
        '''
        Preprocess the observation
        :param obs: (dict[Box, MultiDiscrete]) the input observation
        :return: list of the processed observations
        '''
        # get map as tensor

        # get fuel as tensor

        # get passengers as tensor

        # get taxi position as tensor

        # normalize the values

    def forward(self, observation) -> torch.Tensor:
        map_obs, fuel_obs, passengers_obs, taxi_pos_obs = self.preprocess_obs(observation)

        # flatten the map

        # position encoding of the map

        # embed the map
        
        # normalize the values

        # label encoding for the fuel, passengers, and taxi_pos

        # run transformer

class TaxiPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

policy_kwargs = dict(
    features_extractor_class=TaxiFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256)
)
