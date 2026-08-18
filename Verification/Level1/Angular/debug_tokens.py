
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

model_name = "Qwen/Qwen1.5-0.5B-Chat"
device = "cuda"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

prompt = "To be or not to be"
messages = [{"role": "user", "content": prompt}]

# REF Method
print("\n--- REF METHOD ---")
inputs_ref = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,
    truncation=False
)
print(f"Ref tokens: {inputs_ref[0].tolist()}")
print(f"Ref string: {tokenizer.decode(inputs_ref[0])}")

# MY METHOD (AngularExtractor style)
print("\n--- MY METHOD ---")
# 1. Build string via build_chat_input (simulated)
msgs = [{"role": "user", "content": prompt}]
prompt_str = tokenizer.apply_chat_template(
    msgs,
    tokenize=False,
    add_generation_prompt=True
)
print(f"My prompt string: {repr(prompt_str)}")

# 2. HookedTransformer tokenization
print("Loading HookedTransformer (dummy)...")
# We don't need full model, just tokenizer behavior in to_tokens
# But to_tokens relies on cfg.
# Let's verify what Tokenizer.encode does with the string VS Ref input_ids.
tokens_mine = tokenizer.encode(prompt_str, add_special_tokens=False) # TL usually uses False if prepending BOS manually?
# TL to_tokens:
# if prepend_bos: return cat([bos], encode(text))
# else: return encode(text)

# Check if Ref has BOS
has_bos = (inputs_ref[0][0] == tokenizer.bos_token_id)
print(f"Ref has BOS? {has_bos} (ID: {inputs_ref[0][0]})")

# Check my tokens
print(f"My tokens (encode(str)): {tokens_mine}")

# Alignment check
if inputs_ref[0].tolist() == tokens_mine:
    print("MATCH: Direct encoding matches Ref.")
else:
    print("MISMATCH: Direct encoding differs.")
    # Try prepending BOS
    if tokenizer.bos_token_id and [tokenizer.bos_token_id] + tokens_mine == inputs_ref[0].tolist():
         print("MATCH: Ref has BOS, Mine needs BOS.")
