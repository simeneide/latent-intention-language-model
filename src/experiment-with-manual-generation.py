#%%
1
from transformers import gpt2
#%%
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel, GPT2LMHeadModel
import torch

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")
## MODEL SOURCE CODE:
# https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/models/openai/modeling_openai.py#L531

#%% AUTOMATIC GENERATION
prompt = "Hello, I am"
inputs = tokenizer(prompt, return_tensors="pt")
for _ in range(10):
    generated_tokens = model.generate(**inputs, do_sample=True, max_new_tokens=20)
    print(tokenizer.batch_decode(generated_tokens)[0])

#%% MANUAL GENERATION
prompt = "Hello, i am"
for _ in range(25):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    next_token = tokenizer.batch_decode(outputs.logits[:,-1].argmax().unsqueeze(0))[0]
    prompt += " " + next_token
print(prompt)

# %% USE THE BASE MODEL TO GENERATE THE SAME
model_base = OpenAIGPTModel.from_pretrained("openai-gpt")
#%%
prompt = "Hello, i am"
for _ in range(25):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs_embeds = model_base.tokens_embed(inputs.input_ids)

    output = model_base(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])
    last_hidden = output.last_hidden_state
    logits= (
        last_hidden.unsqueeze(2) 
        * model_base.tokens_embed.weight.unsqueeze(0).unsqueeze(0)
        ).sum(-1)
    next_token = tokenizer.batch_decode(logits[:,-1].argmax().unsqueeze(0))[0]
    prompt += " " + next_token

print(prompt)
# %%
import torch.nn as nn
class LatentIntentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
        self.model = OpenAIGPTModel.from_pretrained("openai-gpt")
        self.add_intention_vector=True
    
    def get_intention_vector(self, batch_size=1):
        emb_size=self.model.tokens_embed.embedding_dim
        one_emb_size= (batch_size,1,emb_size)
        intention_vector = torch.normal(torch.zeros(one_emb_size), 0.2*torch.ones(one_emb_size))
        return intention_vector

    def forward(self, text, intention_vector=None):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs_embeds = self.model.tokens_embed(inputs.input_ids)
        attention_mask = inputs['attention_mask']
        if self.add_intention_vector:
            batch_size, num_tokens = inputs.input_ids.size()
            if intention_vector is None:
                intention_vector = self.get_intention_vector(batch_size=batch_size)
            inputs_embeds = torch.cat((intention_vector, inputs_embeds), dim=1)
            attention_mask = torch.ones((batch_size,num_tokens+1))
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        logits= (
            last_hidden.unsqueeze(2)
            * model_base.tokens_embed.weight.unsqueeze(0).unsqueeze(0)
            ).sum(-1)
        return logits
    
    @torch.no_grad()
    def generate(self, prompt,max_new_tokens=20, intention_vector=None):
        logprobs = []
        for _ in range(max_new_tokens):
            logits=self.forward(prompt, intention_vector=intention_vector)
            tk = torch.topk(logits[:,-1], k=1)
            logprobs.append(tk[0])
            next_token = self.tokenizer.batch_decode(tk[1])[0]
            prompt += " " + next_token
        logprobs = torch.tensor(logprobs)
        return prompt, logprobs
# %%
self = LatentIntentionLanguageModel()
# %%
text = "Paris is a major railway, highway, and air-transport hub"
tokenizer(text)
# served by two international airports: Charles de Gaulle Airport (the second-busiest airport in Europe) and Orly Airport.  Opened in 1900, the city's subway system, the Paris Métro, serves 5.23 million passengers daily."

generated_text, logprobs = self.generate(text)
logprobs.exp()
#%%
from torch.nn import CrossEntropyLoss
ivec = (0.1*self.get_intention_vector()).detach().clone().requires_grad_(True)
#%%
for it in range(150):
    logits = self(text, intention_vector = ivec)[:,1:,]
    inputs = self.tokenizer(text, return_tensors="pt")
    labels = inputs['input_ids']
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if it %10==0:
            print(f"iter {it} \t loss: {loss:.3f}")
        loss.backward()
    with torch.no_grad():
        ivec -= 0.05*ivec.grad
    self.zero_grad()
    ivec.grad=None
# %%

# %% COMPUTE LOG PROBS OF SEQUENCE
from torch.distributions import Categorical
def compute_likelihood_of_words_given_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    labels = inputs['input_ids']
    bs, num_tokens = inputs['input_ids'].size()
    embed_size = 40478
    logits = torch.zeros((bs,num_tokens, embed_size ))
    for n in range(num_tokens):
        attention_mask = torch.zeros_like(inputs['attention_mask'])
        attention_mask[... ,:n] = 1
        logits[:,n,:] = model(input_ids=inputs['input_ids'], attention_mask=attention_mask).logits[:,n,:]

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    per_word_prob = Categorical(logits=shift_logits.view(-1, shift_logits.size(-1))).log_prob(shift_labels.view(-1)).exp().detach().numpy()
    return per_word_prob

#%%
inputs = tokenizer("Hello", return_tensors="pt")
generated_text_lm = tokenizer.batch_decode(model.generate(**inputs, max_new_tokens=20, num_beams=25,early_stopping=True))[0]
generated_text_lm
generated_text_lm_sampled = tokenizer.batch_decode(model.generate(**inputs, max_new_tokens=20, do_sample=True))[0]
generated_text_lm_sampled

generated_text_lilm = self.generate("Hello", intention_vector=ivec, max_new_tokens=20)
generated_text_lilm
# %%
generated_probs_lilm = compute_likelihood_of_words_given_text(generated_text_lilm)
generated_probs_lm = compute_likelihood_of_words_given_text(generated_text_lm)
generated_probs_lm_sampled = compute_likelihood_of_words_given_text(generated_text_lm_sampled)
#%%
import matplotlib.pyplot as plt
plt.plot(generated_probs_lilm,"-o")
plt.plot(generated_probs_lm,"-o")
plt.plot(generated_probs_lm_sampled,"-o")


# %%
