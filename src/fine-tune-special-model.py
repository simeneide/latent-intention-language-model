#%%
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTConfig
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
import torch
from torch import optim, nn
import pytorch_lightning as pl
from datasets import load_dataset

#%% DATASET
class TextData(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=16, max_num_tokens=None):
        super().__init__()
        self.batch_size=batch_size
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        #self.prepare_data_per_node = False
    def prepare_data(self):
        self.datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
    def setup(self, stage: str=None):
        #self.datasets = load_dataset("wikitext", "wikitext-2-raw-v1")
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
            tokenized = self.tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt', max_length=self.max_num_tokens)
            return tokenized

        return torch.utils.data.DataLoader(
            dataset=self.datasets[phase],
            batch_size=self.batch_size,
            shuffle= True if phase == "train" else False,
            collate_fn=collate_tokenize
            )

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token= tokenizer.eos_token

dm = TextData(tokenizer=tokenizer, batch_size=2, max_num_tokens=5)
dm.prepare_data()
dm.setup()

batch = next(iter(dm.train_dataloader()))
batch
#%%
class LatentIntentionLanguageModel(pl.LightningModule):
    """
    the argument `batch` contains input_ids, attention_mask
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

    def forward(self, batch, intention_vector=None):       
        batch_embeds = self.model.wte(batch['input_ids'])
        attention_mask = batch['attention_mask']
        if self.add_intention_vector:
            batch_size, num_tokens = batch['input_ids'].size()
            if intention_vector is None:
                intention_vector = self.get_intention_vector(batch_size=batch_size)
            batch_embeds = torch.cat((intention_vector, batch_embeds), dim=1)
            attention_mask = torch.cat( (torch.ones((batch_size, 1),device=self.device),attention_mask), dim=1) # UPDATE EXISTING ATTENTION INSTEAD!
        
        output = self.model(inputs_embeds=batch_embeds, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        logits= (
            last_hidden.unsqueeze(2)
            * self.model.wte.weight.unsqueeze(0).unsqueeze(0)
            ).sum(-1)
        return logits
    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "valid")

    def step(self, batch, phase):
        logits = self.forward(batch)
        logits
        labels = batch['input_ids']
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        self.log(f"{phase}/loss",loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @torch.no_grad()
    def generate(self, batch,max_new_tokens=20, intention_vector=None):
        assert batch['input_ids'].size(0) ==1, "Cannot generate more than one sentence at a time"
        logprobs = []   
        no_input_tokens = batch['input_ids'].size(1)
        generated = torch.zeros((1,no_input_tokens+max_new_tokens))
        generated[0,:no_input_tokens] = batch['input_ids']

        for n in range(no_input_tokens, max_new_tokens):
            batch = {
                'input_ids' : generated[:1,:n].long(),
                'attention_mask' : torch.ones_like(generated[:1,:n])
            }
            logits=self.forward(batch, intention_vector=intention_vector)
            tk = torch.topk(logits[:,-1], k=1)
            logprobs.append(tk[0])
            next_token = tk[1].item()
            generated[0,n] = next_token
        logprobs = torch.tensor(logprobs)
        return generated, logprobs
# %%
model = LatentIntentionLanguageModel(add_intention_vector=False)

#batch = tokenizer(["Paris is","Oslo should"], return_tensors="pt", padding=True)
print(f"Size running forward(batch): {model(batch).size()}")
model.step(batch,"train")
batch_gen = tokenizer(["Oslo should"], return_tensors="pt", padding=True)
for _ in range(2):
    out=model.generate(batch_gen)
    print(tokenizer.batch_decode(out[0]))
#%%
trainer = pl.Trainer()
trainer.fit(model,dm)
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
