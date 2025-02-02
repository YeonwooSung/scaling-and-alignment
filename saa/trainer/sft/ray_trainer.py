"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from pprint import pprint
from typing import Type, Dict
import torch

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict

from saa import DataProto
from saa.protocol import pad_dataproto_to_divisor, unpad_dataproto
from saa.single_controller.base import Worker
from saa.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from saa.single_controller.ray.base import create_colocated_worker_cls
from saa.trainer.ppo import core_algos
from saa.trainer.core import Role, ResourcePoolManager
from saa.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from saa.utils.torch_functional import masked_mean

WorkerType = Type[Worker]


class RaySFTTrainer(object):
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None
    ):
        self.tokenizer = tokenizer
        self.config = config

        # sft trainer does not have reward_fn, val_reward_fn -> they should be None
        if reward_fn is not None:
            print("Warning: reward_fn is not None. It will be ignored.")
            reward_fn = None
        if val_reward_fn is not None:
            print("Warning: val_reward_fn is not None. It will be ignored.")
            val_reward_fn = None
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = False
        self.use_rm = False

        self.ray_worker_group_cls = ray_worker_group_cls

        self._create_dataloader()


    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from saa.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        #TODO
