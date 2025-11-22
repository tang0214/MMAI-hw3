# ✅ 正確寫法 (改用這個)
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
model_path = "./LLava_huggingface"  # 你剛剛下載的路徑

# 1. 載入模型 (使用 LlavaForConditionalGeneration)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 建議使用 float16 以節省顯存
    device_map="auto"           # 自動分配到 GPU
)

# 2. 載入處理器 (LLaVA 需要 Processor 來同時處理圖片和文字)
processor = AutoProcessor.from_pretrained(model_path)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "What brand is the car?"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

image_file = "./photos/toyota.jpg"
raw_image = Image.open(image_file)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
