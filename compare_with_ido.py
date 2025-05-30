import json
from transformers import AutoTokenizer

# get the tokenizer from kmodel nvidia/Llama-3.1-Nemotron-Nano-8B-v1
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")

test_nir_filename = "debug_aime.txt"
test_ido_filename = "ido_galil_aime_debug.txt"

# readl test_ido_filename into dictionary using json load
with open(test_ido_filename, 'r') as f:
    test_ido = json.load(f)

# readl test_nir_filename line by line, each line a json object, into a list
test_nir = []
with open(test_nir_filename, 'r') as f:
    for line in f:
        test_nir.append(json.loads(line))

test_nir_short = []
test_ido_short = []

for el in test_nir:
    generated_text = el['message_log'][1]['content']
    num_generated_tokens = len(tokenizer.encode(generated_text))
    test_nir_short.append({
        'num_generated_tokens' : el['num_generated_tokens'],
        'num_generated_tokens_verified' : num_generated_tokens,
        'short_prompt' :  el['message_log'][0]['content'][:200]
    })

for el in test_ido['results']:
    generated_text = el['generated_text']
    # count number of tokens in generated_text, using a tokenizer that was defined earlier
    num_generated_tokens = len(tokenizer.encode(generated_text))
    test_ido_short.append({
        'num_generated_tokens' : el['total_generated_tokens'],
        "num_generated_tokens_verified" : num_generated_tokens,
        'short_prompt' : el['prompt'][:200]
    })
    

# sort lists by the short_prompt key
test_nir_short.sort(key=lambda x: x['short_prompt'])
test_ido_short.sort(key=lambda x: x['short_prompt'])

# pprint test_nir_short to test_nir_short.txt
with open('test_nir_short.txt', 'w') as f:
    f.write(json.dumps(test_nir_short, indent=4))

# pprint test_ido_short to test_ido_short.txt
with open('test_ido_short.txt', 'w') as f:
    f.write(json.dumps(test_ido_short, indent=4))

assert len(test_nir_short) == len(test_ido_short)

for result in zip(test_nir_short, test_ido_short):
    print (result[0]['short_prompt'])
    print (result[1]['short_prompt'])
    print (f"num_generated_tokens: {result[0]['num_generated_tokens']} ({result[0]['num_generated_tokens_verified']})")
    print (f"num_generated_tokens: {result[1]['num_generated_tokens']} ({result[1]['num_generated_tokens_verified']})")
    print ("\n")

# compute average of num_generated_tokens from test_nir_short
# compute average of num_generated_tokens_verified from test_ido_short

nir_avg = sum([result['num_generated_tokens'] for result in test_nir_short]) / len(test_nir_short)
ido_avg = sum([result['num_generated_tokens'] for result in test_ido_short]) / len(test_ido_short)

nir_avg_verified = sum([result['num_generated_tokens_verified'] for result in test_nir_short]) / len(test_nir_short)
ido_avg_verified = sum([result['num_generated_tokens_verified'] for result in test_ido_short]) / len(test_ido_short)

print (f"nir_avg: {nir_avg}")
print (f"ido_avg: {ido_avg}")
print (f"nir_avg_verified: {nir_avg_verified}")
print (f"ido_avg_verified: {ido_avg_verified}")



