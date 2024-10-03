from llm2vec import LLM2Vec
class EncodingModel_LLM2vec(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=512,
            skip_instruction = False,
            
        )
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
        vector_dim = 768
        self.vector_linear = nn.Sequential(
            nn.Linear(in_features=4096, out_features=vector_dim),
            nn.Tanh()
        ).to('cuda', dtype=torch.bfloat16)
        # set dtype of vector_linear to bfloat16
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

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

    def forward(self, inputs): # (b, max_length)
        batch_size = len(inputs)
        input_data = self.encoder.tokenize(inputs)
        input_data = {k: v.cuda() for k, v in input_data.items()}
        embeddings = self.encoder.forward(input_data)
        embeddings = self.vector_linear(embeddings)
        return embeddings
    def forward_mixup(self, inputs):
        """
        inputs: batch of [input1, input2]
        """
        batch_size = len(inputs)
        merged_inputs = []
        for i in range(batch_size):
            merged_inputs.append(inputs[i][0] + " " + inputs[i][1])
        input_data = self.tokenizer(merged_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        # take tokens from inputs[i][0] and inputs[i][1]
        outputs = self.encoder.model(**input_data)
        last_hidden_state = outputs.last_hidden_state
        temp = [inputs[i][0] for i in range(batch_size)]
        input_data_1 = self.tokenizer(temp)
        len_first = [len(input_data_1['input_ids'][i]) for i in range(batch_size)]

        last_token_indices = input_data['attention_mask'].sum(dim=1) - 1
        last_token_indices = last_token_indices.tolist()
        
        first = [last_hidden_state[i, :len_first[i], :].mean(dim = 0) for i in range(batch_size)]
        second = [last_hidden_state[i, len_first[i]:last_token_indices[i], :].mean(dim = 0) for i in range(batch_size)]
        first = torch.stack(first)
        second = torch.stack(second)
        # compress to bert size
        first = self.vector_linear(first)
        second = self.vector_linear(second)
        return first, second

    def compress_to_bert_size(self, vectors):
        return self.vector_linear(vectors)

    def save_lora_weights(self, save_path):
        """
        Save the LoRA weights of the model.
        
        Args:
            save_path (str): The path where the LoRA weights will be saved. (.../encoder)
        """
        # Ensure the encoder model is in evaluation mode
        self.encoder.model.eval()
        
        # Get the underlying PEFT model
        peft_model = self.encoder.model
        
        # Check if the model is a PeftModel
        if not isinstance(peft_model, PeftModel):
            raise ValueError("The model doesn't seem to be a PeftModel. LoRA weights cannot be saved.")
        
        # Save only the LoRA weights
        peft_model.save_pretrained(save_path)
        
        # Save the vector linear weights
        torch.save(self.vector_linear.state_dict(), os.path.join(save_path, "vector_linear.pth"))
        
        print(f"LoRA weights saved to {save_path}")
    
    def load_lora_weights(self, load_path):
        """
        Load the LoRA weights into the model.
        
        Args:
            load_path (str): The path where the LoRA weights are saved.
        """
        # Ensure the encoder model is in evaluation mode
        self.encoder.model.eval()
        
        # Get the underlying PEFT model
        peft_model = self.encoder.model
        
        # Check if the model is a PeftModel
        if not isinstance(peft_model, PeftModel):
            raise ValueError("The model doesn't seem to be a PeftModel. LoRA weights cannot be loaded.")
        
        # Load the LoRA weights
        peft_model.load_adapter(load_path, adapter_name="default")
        # Load the vector linear weights
        self.vector_linear.load_state_dict(torch.load(os.path.join(load_path, "vector_linear.pth")))
        print(f"LoRA weights loaded from {load_path}")
        
        # Optionally, you can set the loaded adapter as active
        peft_model.set_adapter("default")