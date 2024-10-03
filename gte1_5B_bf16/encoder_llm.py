import torch
import torch.nn as nn
from llm2vec import LLM2Vec
from torch import Tensor, device, nn
import torch.nn.functional as F

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model
from typing import Any, Dict, List, Optional, Tuple, Union



class EncodingModel_Stella(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen1.5-7B-instruct', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen1.5-7B-instruct', device_map="auto", trust_remote_code=True, bf16=True)

        self.model = self.initialize_peft(
            self.model,
        )
    
    
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
       

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, input_texts, is_des = False): # (b, max_length)
        batch_dict = self.tokenizer(input_texts, max_length=256, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
        

