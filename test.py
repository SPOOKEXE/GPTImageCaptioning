from transformers import AutoModel, AutoProcessor
from PIL import Image

import torch

model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

prompt = "Question or Instruction"
image = Image.open("C:\\Users\\Declan\\Desktop\\Test\\img\\15_AnjyuKouzuki\\b_st2_natsusyoujyo01_kouzuki_a07_001.jpg")

inputs = processor(text=[prompt], images=[image], return_tensors="pt")
with torch.inference_mode():
	output = model.generate(
		**inputs,
		do_sample=False,
		use_cache=True,
		max_new_tokens=256,
		eos_token_id=151645,
		pad_token_id=processor.tokenizer.pad_token_id
	)

prompt_len = inputs["input_ids"].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
print(decoded_text)
