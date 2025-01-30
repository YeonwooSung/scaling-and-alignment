import torch
import torch.nn as nn
import os
from collections import OrderedDict
from transformers import AutoModelForCausalLM
from trl import PreTrainedModelWrapper
from peft import PeftModel, PeftConfig
from safetensors import safe_open


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def is_lora_model(model):
    for key in model.state_dict().keys():
        if 'lora' in key:
            return True
    return False

def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'pretrained_model.' in key:
                key = key.replace('pretrained_model.', '')
            save_dict[key] = state_dict[key]
    return save_dict

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(name)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )



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

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size

        self.layer_type = kwargs.pop("layer_type", 'linear')
        num_neurons = kwargs.pop("num_neurons", 1024)
        num_layers = kwargs.pop("num_layers", 1)
        if self.layer_type == 'linear':
            self.summary = nn.Linear(hidden_size, 1)
        else:
            module_lis = []
            input_neurons = hidden_size
            for i in range(num_layers):
                module_lis.extend([nn.Linear(input_neurons, num_neurons), nn.ReLU()])
                input_neurons = num_neurons
                
            module_lis.append(nn.Linear(num_neurons, 1))
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
    r"""
    An autoregressive model with a value head in addition to the language model head.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, `push_to_hub` and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the `ValueHead` class. Currently, the supported args are:
            - **summary_dropout_prob** (`float`, `optional`, defaults to `None`) -- The dropout probability for the
                `ValueHead` class.
            - **v_head_initializer_range** (`float`, `optional`, defaults to `0.2`) -- The initializer range for the
                `ValueHead` if a specific initialization strategy is selected.
            - **v_head_init_strategy** (`str`, `optional`, defaults to `None`) -- The initialization strategy for the
                `ValueHead`. Currently, the supported strategies are:
                - **`None`** -- Initializes the weights of the `ValueHead` with a random distribution. This is the default
                    strategy.
                - **"normal"** -- Initializes the weights of the `ValueHead` with a normal distribution.

    """
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    supported_args = (
        "summary_dropout_prob",
        "v_head_initializer_range",
        "v_head_init_strategy",
        "layer_type",
        'num_neurons',
        'num_layers',
    )

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class.
        """
        super().__init__(pretrained_model, **kwargs)
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        self.v_head = ValueHead(self.pretrained_model.config, **v_head_kwargs)

        self._init_weights(**v_head_kwargs)

    def _init_weights(self, **kwargs):
        r"""
        Initializes the weights of the value head. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `v_head_init_strategy` argument
        when calling `.from_pretrained`. Supported strategies are:
        - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `ValueHead` class. These arguments
                can contain the `v_head_init_strategy` argument as well as the `v_head_initializer_range`
                argument.
        """
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
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
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

    def generate(self, *args, **kwargs):
        r"""
        A simple wrapper around the `generate` method of the wrapped model.
        Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
        method of the wrapped model for more information about the supported arguments.

        Args:
            *args (`list`, *optional*):
                Positional arguments passed to the `generate` method of the wrapped model.
            **kwargs (`dict`, *optional*):
                Keyword arguments passed to the `generate` method of the wrapped model.
        """
        return self.pretrained_model.generate(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        ### return lora 
        pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs).copy()

        v_head_state_dict = self.v_head.state_dict(*args, **kwargs).copy()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        # print('pretrained_model_state_dict', pretrained_model_state_dict.keys())
        # input()
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



def load_model_withhead(model_name, peft_name, tokenizer, device, \
        layer_type='linear', num_neurons=1024, num_layers=1, load_in_8bit=False):

    model_config = {
        'device_map': device,
        'layer_type': layer_type,
        'num_neurons': num_neurons,
        'num_layers': num_layers,
    }
    if load_in_8bit:
        model_config['load_in_8bit'] = True
    else:
        model_config['torch_dtype'] = torch.bfloat16

    if 'Mistral' not in model_name:
        model_config['attn_implementation'] = "flash_attention_2"
        
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **model_config)
    model.config.pad_token_id = tokenizer.pad_token_id

    if len(peft_name) and os.path.exists(peft_name):
        peft_config = PeftConfig.from_pretrained(peft_name)
        model = PeftModel(model, peft_config)

        loaded_state_dict = {}
        if os.path.exists(os.path.join(peft_name, "model.safetensors")):
            with safe_open(os.path.join(peft_name, "model.safetensors"), framework="pt", device=device) as f:
                for k in f.keys():
                    loaded_state_dict[k] = f.get_tensor(k)
        else:
            loaded_state_dict = torch.load(os.path.join(peft_name, 'pytorch_model.bin'))
        # # Check if the loaded object is a state dict or a complete model
        # if isinstance(loaded_state_dict, dict):
        #     # Print the keys
        #     print(loaded_state_dict.keys())
        # else:
        #     # If it's a complete model, print the model parameters
        #     print([name for name, _ in loaded_state_dict.named_parameters()])

        missing, unexpected = model.base_model.model.pretrained_model.load_state_dict(loaded_state_dict, strict=False)
        missing, unexpected = model.base_model.model.load_state_dict(loaded_state_dict, strict=False)
    
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    return model

def model_withhead_forward(model, input_ids, attention_mask, device, forward_type='reward'):
    if forward_type == 'reward':
        _, _, reward_tensors = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    elif forward_type == 'dpo':
        _, reward_tensors, _ = model(input_ids.to(device), attention_mask=attention_mask.to(device))
    else:
        raise NotImplementedError
    return reward_tensors