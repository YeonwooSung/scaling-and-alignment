import torch
import torch.nn as nn
import os
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import PreTrainedModelWrapper
from peft import PeftModel, PeftConfig
from safetensors import safe_open


class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        # get vhead config
        if hasattr(config, "vhead_layer_type"): # config from json first
            self.layer_type = config.vhead_layer_type
        else:
            self.layer_type = kwargs.pop("vhead_layer_type", 'mlp')
        if hasattr(config, 'vhead_num_neurons'):
            num_neurons = config.vhead_num_neurons
        else:
            num_neurons = kwargs.pop("vhead_num_neurons", 1024)
        if hasattr(config, 'vhead_num_layers'):
            num_layers = config.vhead_num_layers
        else:
            num_layers = kwargs.pop("vhead_num_layers", 1)

        if hasattr(config, 'vhead_num_output'):
            num_output = config.vhead_num_output
        else:
            num_output = kwargs.pop("vhead_num_output", 1)

        if self.layer_type == 'linear':
            self.summary = nn.Linear(hidden_size, num_output)
        else:
            module_lis = []
            input_neurons = hidden_size
            for i in range(num_layers):
                module_lis.extend([nn.Linear(input_neurons, num_neurons), nn.ReLU()])
                input_neurons = num_neurons
                
            module_lis.append(nn.Linear(num_neurons, num_output))
            self.summary = nn.Sequential(*module_lis)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if (self.layer_type == 'linear' and output.dtype != self.summary.weight.dtype):
            output = output.to(self.summary.weight.dtype)
        elif (self.layer_type != 'linear' and output.dtype != self.summary[0].weight.dtype):
            output = output.to(self.summary[0].weight.dtype)

        output = self.summary(output)
        return output


class AutoModelForCausalLMWithValueHead(PreTrainedModelWrapper):
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
        "vhead_layer_type",
        'vhead_num_neurons',
        'vhead_num_layers',
        'vhead_num_output',
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)
        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        initializer_range = kwargs.pop("v_head_initializer_range", 0.2)
        # random init by default
        init_strategy = kwargs.pop("v_head_init_strategy", None)
        if init_strategy is None:
            # do nothing
            pass
        elif init_strategy == "normal":
            self.v_head.summary.weight.data.normal_(mean=0.0, std=initializer_range)
            self.v_head.summary.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = base_model_output.loss

        if (hasattr(self.v_head.summary, 'weight') and last_hidden_state.device != self.v_head.summary.weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.summary.weight.device)
        elif not hasattr(self.v_head.summary, 'weight') and (last_hidden_state.device != self.v_head.summary[0].weight.device):
            last_hidden_state = last_hidden_state.to(self.v_head.summary[0].weight.device)
        
        # use the last token value as reward
        last_index = attention_mask.sum(dim=-1) - 1
        value = self.v_head(last_hidden_state).squeeze(-1)[torch.arange(len(last_hidden_state)), last_index]

        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()

        return (lm_logits, loss, value)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def push_to_hub(self, *args, **kwargs):
        setattr(self.pretrained_model, "v_head", self.v_head)
        return self.pretrained_model.push_to_hub(*args, **kwargs)

    

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the value head to the state dictionary of the wrapped model
        by prepending the key with `v_head.`. This function removes the `v_head.` prefix from the
        keys of the value head state dictionary.
        """
        for k in list(state_dict.keys()):
            if "v_head." in k:
                state_dict[k.replace("v_head.", "")] = state_dict.pop(k)
        self.v_head.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for ValueHead models."
                )

            first_device = list(set(self.pretrained_model.hf_device_map.values()))[0]

            self.v_head = self.v_head.to(first_device)

            def set_device_hook(module, input, outputs):
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(first_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)

            self.is_sequential_parallel = True
    
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class



def load_model_withhead(model_name, peft_name, tokenizer, device, \
        layer_type='linear', num_neurons=1024, num_layers=1, load_in_8bit=False):

    model_config = {
        'device_map': device,
        'vhead_layer_type': layer_type,
        'vhead_num_neurons': num_neurons,
        'vhead_num_layers': num_layers,
    }
    if load_in_8bit:
        model_config['load_in_8bit'] = True
    else:
        model_config['torch_dtype'] = torch.bfloat16

    if 'Mistral' not in model_name:
        model_config['attn_implementation'] = "flash_attention_2"
        
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **model_config)
    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if len(peft_name) and os.path.exists(peft_name):
        peft_config = PeftConfig.from_pretrained(peft_name)
        model = PeftModel(model, peft_config)
        loaded_state_dict = {}

        safetensor_files = sorted([f for f in os.listdir(peft_name) if f.endswith('.safetensors')])
        if len(safetensor_files):
            for safetensor_file in safetensor_files:
                safetensor_path = os.path.join(peft_name, safetensor_file)
                if os.path.exists(safetensor_path):
                    with safe_open(safetensor_path, framework="pt", device=device) as f:
                        for k in f.keys():
                            loaded_state_dict[k] = f.get_tensor(k)
        else:
            loaded_state_dict = torch.load(os.path.join(peft_name, "pytorch_model.bin"))
        missing, unexpected = model.base_model.model.pretrained_model.load_state_dict(loaded_state_dict, strict=False)
        missing, unexpected = model.base_model.model.load_state_dict(loaded_state_dict, strict=False)
    
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    return model

def model_withhead_forward(model, input_ids, attention_mask, device, forward_type='reward', labels=None):
    if forward_type == 'reward':
        _, _, reward_tensors = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    elif forward_type == 'dpo':
        res = model(input_ids.to(device), attention_mask=attention_mask.to(device))
        if len(res) == 3:
            logits, _, _ = res 
        else:
            logits = res.logits
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)
    else:
        raise NotImplementedError
    return reward_tensors