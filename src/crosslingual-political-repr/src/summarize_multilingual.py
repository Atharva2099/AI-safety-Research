import glob, json, os, pandas as pd

models = [
    ("OLMo 3 (7B)", "allenai_Olmo-3-7B-Instruct"),
    ("Qwen 3.5 (9B)", "Qwen_Qwen3.5-9B"),
    ("Gemma 2 (9B)", "google_gemma-2-9b-it"),
    ("Ministral (8B)", "mistralai_Ministral-8B-Instruct-2410")
]

base_dir = "src/crosslingual-political-repr/artifacts/results"
langs = ["en", "es", "de", "zh", "hi", "mr"]

for model_name, slug in models:
    print(f"\n=========================================================================")
    print(f"=== Model: {model_name} ===")
    print(f"=========================================================================")
    print("| Language | In-Lang Peak Layer | In-Lang Acc | Zero-Shot EN->Lang Peak Layer | Zero-Shot Acc | Transfer Retention |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for l in langs:
        f = os.path.join(base_dir, f"multilingual_probe_{slug}_{l}.jsonl")
        if os.path.exists(f):
            df = pd.read_json(f, lines=True)
            in_l = df[df["mode"] == "in_language"].sort_values("accuracy", ascending=False).iloc[0]
            il_layer = in_l["layer"]
            il_acc = in_l["accuracy"] * 100
            
            zs = df[df["mode"] == "zero_shot"]
            if not zs.empty:
                zs_best = zs.sort_values("accuracy", ascending=False).iloc[0]
                zs_layer = zs_best["layer"]
                zs_acc = zs_best["accuracy"] * 100
                retention = (zs_acc / il_acc) * 100
                print(f"| {l:^8} | {il_layer:^18} | {il_acc:6.2f}% | {zs_layer:^29} | {zs_acc:6.2f}% | {retention:6.1f}% |")
            else:
                print(f"| {l:^8} | {il_layer:^18} | {il_acc:6.2f}% | {'(Baseline)':^29} | {'100.0%':^13} | {'100.0%':^18} |")
        else:
            print(f"| {l:^8} | {'N/A':^18} | {'N/A':^11} | {'N/A':^29} | {'N/A':^13} | {'N/A':^18} |")
