from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 1. Load your tokenizer & model
model_name = "/lustre/fsw/portfolios/coreai/users/ageifman/code/checkpoints/qwen_1_5b_llm_tokens/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Define your new tokens and the vectors you want for each
new_tokens =  [f"<new_word{i}>" for i in range(100)]
# new_tokens.append("<think>")
# new_tokens.append("</think>")
# print(new_tokens)
# hidden_size = model.config.hidden_size
# new_vectors = [torch.randn(hidden_size)]*len(new_tokens)
input_embeddings = model.get_input_embeddings()
embedding_weight = input_embeddings.weight.data  # shape: (vocab_size, hidden_size)

# 4. Compute statistics
mean = embedding_weight.mean(dim=0)
std = embedding_weight.std(dim=0)
avg_norm = embedding_weight.norm(dim=1).mean()
# 5. Sample a single normalized vector and replicate it
# single_vector = torch.normal(mean=mean, std=std)
vec = torch.normal(mean=mean, std=std)
vec = vec / vec.norm() * avg_norm

new_vectors = [vec]*len(new_tokens)
# 3. Add the tokens to the tokenizer
num_added = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added} tokens. Vocabulary size is now {len(tokenizer)}.")

# 4. Resize model embeddings so they have room for the new tokens
model.resize_token_embeddings(len(tokenizer))

# 5. Assign your custom vectors to the newly added slots
inp_emb = model.get_input_embeddings().weight.data
out_emb = None
if model.config.tie_word_embeddings is False:
    # Only fetch output embeddings if they are not tied to input
    out_emb = model.get_output_embeddings().weight.data

for tok, vec in zip(new_tokens, new_vectors):
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    inp_emb[tok_id] = vec
    if out_emb is not None:
        out_emb[tok_id] = vec

print("Custom embeddings assigned!")

# 6. Save the updated tokenizer & model
save_dir = "/lustre/fsw/portfolios/coreai/users/ageifman/code/checkpoints/qwen_1_5b_llm_tokens/updated_model"
os.makedirs(save_dir, exist_ok=True)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"Model and tokenizer saved to: {save_dir}")
