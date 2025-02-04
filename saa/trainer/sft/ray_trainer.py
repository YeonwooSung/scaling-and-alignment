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
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict

# custom modules
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
        # TODO: we have to make sure the batch size is divisible by the dp size
        from saa.utils.dataset.sft_dataset import SFTDataset

        config = self.config

        # build dataset
        self.train_dataset = SFTDataset(
            parquet_files=config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=config.data.prompt_key,
            prompt_dict_keys=config.data.get('prompt_dict_keys', None),
            response_key=config.data.response_key,
            response_dict_keys=config.data.get('response_dict_keys', None),
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )
        self.val_dataset = SFTDataset(
            parquet_files=config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=config.data.prompt_key,
            prompt_dict_keys=config.data.get('prompt_dict_keys', None),
            response_key=config.data.response_key,
            response_dict_keys=config.data.get('response_dict_keys', None),
            max_length=config.data.max_length,
            truncation=config.data.truncation
        )

        #
        # build dataloader
        #

        # rank = self.device_mesh.get_rank()
        # world_size = self.device_mesh.size()
        rank = 0
        world_size = 1

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )


    def _validate(self):
        metric_dict = {}
        return metric_dict
