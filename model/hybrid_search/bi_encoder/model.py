import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class BiEncoderPhoBERT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)

        # LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["encoder.layer.11.attention.output.dense"],  # hoặc mở rộng thêm
            bias="none"
        )
        self.encoder = get_peft_model(base_model, peft_config)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0]  # [CLS] vector
        return cls_output
