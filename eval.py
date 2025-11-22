import os
import json
import re
from tqdm import tqdm

import torch
from PIL import Image
import numpy as np

from transformers import AutoProcessor, LlavaForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# === setting ===
#MODEL_PATH = "./LLava_huggingface"
MODEL_PATH = "./output/v8-20251121-235817/checkpoint-505-merged"
DATA_ROOT = "./converted_vqa_rad/test"
TEST_JSONL = os.path.join(DATA_ROOT, "test.jsonl")
MAX_NEW_TOKENS = 32


# === functions ===
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}") from e
    return items

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def compute_bleu_scores(preds, gts):
    smoothie = SmoothingFunction().method4
    scores = []
    for p, g in zip(preds, gts):
        p_n = normalize_text(p)
        g_n = normalize_text(g)
        if not g_n.strip():
            continue
        score = sentence_bleu(
            [g_n.split()],
            p_n.split(),
            weights=(0.5, 0.5),
            smoothing_function=smoothie,
        )
        scores.append(score)
    if not scores:
        return 0.0, 0.0, 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)

def extract_qa(item):
    # support for both message and conversations format
    if "messages" in item:
        msgs = item["messages"]
        user = msgs[0]
        assistant = msgs[1]

        user_content = user.get("content", "")
        if isinstance(user_content, list):
            q_parts = []
            for c in user_content:
                if isinstance(c, dict) and c.get("type") == "text":
                    q_parts.append(c.get("text", ""))
            q = " ".join(q_parts).strip()
        else:
            q = str(user_content).replace("<image>", "").strip()

        assistant_content = assistant.get("content", "")
        if isinstance(assistant_content, list):
            gt_parts = []
            for c in assistant_content:
                if isinstance(c, dict) and c.get("type") == "text":
                    gt_parts.append(c.get("text", ""))
            gt = " ".join(gt_parts).strip()
        else:
            gt = str(assistant_content).strip()
        return q, gt

    if "conversations" in item:
        convs = item["conversations"]
        human = convs[0]["value"]
        gt = convs[1]["value"].strip()
        if "<image>" in human:
            q = human.split("<image>")[-1].strip()
        else:
            q = human.strip()
        return q, gt

    raise ValueError("Unknown data format: no 'messages' or 'conversations'.")


def resolve_image_path(item):
    img_path = item.get("image")
    if img_path is None:
        images = item.get("images") or []
        if images:
            img_path = images[0]

    if img_path is None:
        raise ValueError("Unknown data format: no image path found.")

    if os.path.isabs(img_path):
        return img_path
    if os.path.exists(img_path):
        return img_path
    return os.path.join(DATA_ROOT, img_path)


# === load model ===
def load_model_and_processor():
    print("Loading processor & model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print("Model loaded on", device)
    return model, processor, device


# === main ===
def main():
    data = load_jsonl(TEST_JSONL)

    model, processor, device = load_model_and_processor()

    all_preds = []
    all_gts = []

    for item in tqdm(data, desc="Evaluating on test.jsonl"):
        img_path = resolve_image_path(item)
        image = Image.open(img_path).convert("RGB")

        q, gt = extract_qa(item)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ],
            }
        ]
        

        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        pred = processor.decode(gen_ids, skip_special_tokens=True)
        all_preds.append(pred)
        all_gts.append(gt)
        

    mean_bleu, std_bleu, n = compute_bleu_scores(all_preds, all_gts)

    print("======== VQA-RAD test set BLEU ========")
    print(f"Samples: {n}")
    print(f"Mean BLEU-2: {mean_bleu:.4f}")
    print(f"Std  BLEU-2: {std_bleu:.4f}")

if __name__ == "__main__":
    main()
