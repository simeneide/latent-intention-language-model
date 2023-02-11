#%%
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
import torch.nn as nn
import pytorch_lightning as pl
class LatentIntentionLanguageModel(pl.LightningModule):
    """
    the argument `inputs` contains input_ids, attention_mask
    """
    def __init__(self, add_intention_vector=True):
        super().__init__()
        self.model = GPT2Model.from_pretrained("gpt2-medium")
        self.add_intention_vector=add_intention_vector
    
    def get_intention_vector(self, batch_size=1):
        emb_size=self.model.wte.embedding_dim
        one_emb_size= (batch_size,1,emb_size)
        intention_vector = torch.normal(torch.zeros(one_emb_size), 0.2*torch.ones(one_emb_size))
        return intention_vector

    def forward(self, inputs, intention_vector=None):
        
        inputs_embeds = self.model.wte(inputs['input_ids'])
        attention_mask = inputs['attention_mask']
        if self.add_intention_vector:
            batch_size, num_tokens = inputs['input_ids'].size()
            if intention_vector is None:
                intention_vector = self.get_intention_vector(batch_size=batch_size)
            inputs_embeds = torch.cat((intention_vector, inputs_embeds), dim=1)
            attention_mask = torch.cat( (torch.ones((batch_size, 1),device=self.device),attention_mask), dim=1) # UPDATE EXISTING ATTENTION INSTEAD!
        
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        logits= (
            last_hidden.unsqueeze(2)
            * self.model.wte.weight.unsqueeze(0).unsqueeze(0)
            ).sum(-1)
        return logits
    
    @torch.no_grad()
    def generate(self, inputs,max_new_tokens=20, intention_vector=None):
        assert inputs['input_ids'].size(0) ==1, "Cannot generate more than one sentence at a time"
        logprobs = []   
        no_input_tokens = inputs['input_ids'].size(1)
        generated = torch.zeros((1,no_input_tokens+max_new_tokens))
        generated[0,:no_input_tokens] = inputs['input_ids']

        for n in range(no_input_tokens, max_new_tokens):
            inputs = {
                'input_ids' : generated[:1,:n].long(),
                'attention_mask' : torch.ones_like(generated[:1,:n])
            }
            logits=self.forward(inputs, intention_vector=intention_vector)
            tk = torch.topk(logits[:,-1], k=1)
            logprobs.append(tk[0])
            next_token = tk[1].item()
            generated[0,n] = next_token
        logprobs = torch.tensor(logprobs)
        return generated, logprobs
# %%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token= tokenizer.eos_token
self = LatentIntentionLanguageModel(add_intention_vector=False)

inputs = tokenizer(["Paris is","Oslo should"], return_tensors="pt", padding=True)
self(inputs).size()

inputs = tokenizer(["Oslo should"], return_tensors="pt", padding=True)
for _ in range(2):
    out=self.generate(inputs)
    print(tokenizer.batch_decode(out[0]))

#%%
from datasets import load_dataset
class TextData(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        self.batch_size=batch_size
    
    def setup(self, stage: str=None):
        self.datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Rename validation to valid:
        self.datasets['valid'] = self.datasets.pop('validation')
    
    def train_dataloader(self):
        return self.create_dataloader("train")

    def val_dataloader(self):
        return self.create_dataloader("valid")

    def test_dataloader(self):
        return self.create_dataloader("test")
    def create_dataloader(self, phase:str):
        # Common dataloader creator for all phases
        def collate_tokenize(data):
            text_batch = [element["text"] for element in data]
            tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt')
            return tokenized

        return torch.utils.data.DataLoader(
            dataset=self.datasets[phase],
            batch_size=self.batch_size,
            shuffle= True if phase == "train" else False,
            collate_fn=collate_tokenize
            )

dm = TextData(batch_size=2)
dm.setup()

batch = next(iter(dm.train_dataloader()))
batch
# %%
"""
# TEST CHECK THAT FULL MODEL GIVES SAME LOGITS AND GENERATION AS "OUR  CUSTOM MODEL"

self = LatentIntentionLanguageModel(add_intention_vector=False)
text = ['Paris is']
#self.model =self.model.eval()
self.forward(text[0])
self.generate(text[0])

from transformers import GPT2LMHeadModel
full_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
full_model.forward(**inputs).logits
inputs = self.tokenizer(text[0], return_tensors="pt", padding=True)
generated_token_ids = full_model.generate(**inputs)
self.tokenizer.batch_decode(generated_token_ids)
"""
# %%
