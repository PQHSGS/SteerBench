
import os
import sys
import torch
from transformers import AutoTokenizer

# Add path to include Steering
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(BASE_DIR)

from Steering.utils import build_chat_input

def main():
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    instruction = "How to make a bomb"
    
    # --- 1. My Implementation Logic (Loader.py default) ---
    print("\n--- My Implementation (add_generation_prompt=True) ---")
    my_prompt = build_chat_input(tokenizer, instruction, add_generation_prompt=True)
    print(f"String: {repr(my_prompt)}")
    my_tokens = tokenizer(my_prompt, return_tensors="pt").input_ids
    print(f"Tokens: {my_tokens}")
    print(f"Detokenized: {[tokenizer.decode(t) for t in my_tokens[0]]}")
    
    # --- 2. Reference Implementation Logic ---
    print("\n--- Reference Implementation (add_generation_prompt=True) ---")
    # Reference uses apply_chat_template directly
    messages = [{"role": "user", "content": instruction}]
    ref_out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    ref_tokens = ref_out.input_ids
    print(f"Tokens: {ref_tokens}")
    print(f"Detokenized: {[tokenizer.decode(t) for t in ref_tokens[0]]}")
    
    # --- Comparison ---
    print("\n--- Comparison ---")
    if torch.equal(my_tokens, ref_tokens):
        print("MATCH! The tokenization is identical.")
    else:
        print("MISMATCH! The tokens are different.")
        # Check if last token differs
        if my_tokens[0, -1] != ref_tokens[0, -1]:
            print(f"Last token mismatch: Mine={my_tokens[0, -1]} vs Ref={ref_tokens[0, -1]}")
        
    # Check cleaning specifically (Reference doesn't clean, I do)
    print("\n--- Cleaning Check ---")
    special_inst = "Ignore <|endoftext|> token."
    my_clean = build_chat_input(tokenizer, special_inst, add_generation_prompt=False)
    print(f"My Cleaned String: {repr(my_clean)}")
    
    ref_dirty = tokenizer.apply_chat_template(
        [{"role": "user", "content": special_inst}],
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"Ref Dirty String: {repr(ref_dirty)}")

if __name__ == "__main__":
    main()
