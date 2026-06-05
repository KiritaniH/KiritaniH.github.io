import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torchvision.models as models


class LLaMA_adapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.llama = GPT2LMHeadModel.from_pretrained("gpt2")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llama.config.pad_token_id = self.llama.config.eos_token_id

        base_cnn = models.resnet50(weights='IMAGENET1K_V1')
        modules = list(base_cnn.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        self.image_fc = nn.Linear(2048, 512) 

        self.proj = nn.Linear(512, self.llama.config.n_embd)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, image, text):
        with torch.no_grad():
            vis = self.image_encoder(image).squeeze(-1).squeeze(-1) 
        vis = self.image_fc(vis)          
        vis = self.proj(vis).unsqueeze(1) 
        tok = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        text_embeds = self.llama.transformer.wte(tok.input_ids)
        combined = torch.cat([vis, text_embeds], dim=1)
        outputs = self.llama(inputs_embeds=combined)
        return outputs

    def generate(self, image, prompt="Describe the image", max_len=64):
        self.eval()
        with torch.no_grad():
            vis = self.image_encoder(image).squeeze(-1).squeeze(-1)
            vis = self.image_fc(vis)
            vis = self.proj(vis).unsqueeze(1)

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            text_embeds = self.llama.transformer.wte(input_ids)
            combined = torch.cat([vis, text_embeds], dim=1)

            outputs = self.llama.generate(
                inputs_embeds=combined,
                max_new_tokens=max_len,
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)