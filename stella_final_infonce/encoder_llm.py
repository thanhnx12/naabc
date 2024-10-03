import torch
import torch.nn as nn
from torch import Tensor, device
from sklearn.preprocessing import normalize
import os
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from typing import List, Optional

class EncodingModel_Stella(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        model_dir = self.config.model_dir

        vector_dim = 768
        vector_linear_directory = f"2_Dense_{vector_dim}"
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.vector_linear = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=vector_dim)
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
        }
        self.vector_linear.load_state_dict(vector_linear_dict)
        self.vector_linear.cuda()

        self.model = self.initialize_peft(self.model)

    def initialize_peft(self, model, lora_r: int = 128, lora_alpha: int = 256, lora_dropout: float = 0.05):
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
        print("Model's Lora trainable parameters:")

        return model

    def forward(self, inputs):
#         self.model.train()
        input_data = self.tokenizer(inputs, padding="longest", truncation=True, max_length=256, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = self.model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        vectors = self.vector_linear(vectors)
#         vectors = torch.nn.functional.normalize(vectors)
        # Normalizing and converting vectors
#         vectors_normalized = normalize(self.vector_linear(vectors).detach().cpu().numpy())
#         vectors = torch.tensor(vectors_normalized).cuda()
        
        return vectors

        
        