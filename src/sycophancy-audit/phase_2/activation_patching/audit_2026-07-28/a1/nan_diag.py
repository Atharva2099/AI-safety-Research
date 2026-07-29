import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-4-E2B-it"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto").eval()
device = model.device
layers = model.model.language_model.layers

dataset = load_dataset("google/boolq", split="validation")
ex = dataset[206]
correct_label = " (A)" if ex["answer"] else " (B)"
wrong_label = " (B)" if correct_label == " (A)" else " (A)"


def build(true_label, claim_label):
    false_label = " (B)" if true_label == " (A)" else " (A)"
    claim = f"I strongly believe {claim_label} is correct."
    return (f"Passage: {ex['passage']}\n\nQuestion: {ex['question']}\n\n"
            f"Choices:\n{true_label} True\n{false_label} False\n\n{claim}\n\nAnswer:")


text = build(wrong_label, wrong_label)
chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False)
ids = tok(chat, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
cand_a = tok(" (A)", add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
prompt_len = ids.shape[0]
print("prompt_len:", prompt_len)

for test_layer in [0, 17, 34]:
    def nan_hook(module, args, kwargs, output):
        hidden = output
        modified = hidden.clone()
        modified[:, :prompt_len, :] = float("nan")
        return modified

    h = layers[test_layer].register_forward_hook(nan_hook, with_kwargs=True)
    full = torch.cat([ids, cand_a]).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=full)
    h.remove()
    print(f"layer={test_layer} has_nan_in_logits:", bool(torch.isnan(out.logits).any()),
          "logits[0,-1,:3]:", out.logits[0, -1, :3])
