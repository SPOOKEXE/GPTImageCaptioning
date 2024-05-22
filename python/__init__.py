
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import torch
import os

# available models
model : str = "deepseek-ai/deepseek-vl-7b-chat" # "deepseek-ai/deepseek-vl-7b-base"
base_prompt : str = """You are the best captioning bot in the world. CLIP caption the below image and include their ethnicity, estimated age, estimated height, hair color, eye color and such, all in one sentence using commas. Keep it succinct and emotionless.
<image_placeholder>"""

# models and processors
print("Loading Chat Processor")
vl_chat_processor : VLChatProcessor = VLChatProcessor.from_pretrained(model)
tokenizer : AutoTokenizer = vl_chat_processor.tokenizer

print("Loading Model")
vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# caption function
def caption_image( image_filepath : str ) -> str:
	if os.path.exists(image_filepath) is False:
		print("Given image does not exist.")
		return None
	conversation : list[dict] = [
		{ "role": "User", "content": base_prompt, "images": [image_filepath] },
		{ "role": "Assistant", "content": "" }
	]
	# load images
	pil_images = load_pil_images( conversation )
	pil_images = [ image.resize((1024,1024)) for image in pil_images ]
	# prepare inputs
	prepare_inputs = vl_chat_processor( conversations=conversation, images=pil_images, force_batchify=True ).to(vl_gpt.device)
	del pil_images
	# run image encoder to get the image embeddings
	inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
	att_mask = prepare_inputs.attention_mask
	del prepare_inputs
	# run the model to get the response
	outputs = vl_gpt.language_model.generate(
		inputs_embeds=inputs_embeds,
		attention_mask=att_mask,
		pad_token_id=tokenizer.eos_token_id,
		bos_token_id=tokenizer.bos_token_id,
		eos_token_id=tokenizer.eos_token_id,
		max_new_tokens=512,
		do_sample=False,
		use_cache=True
	)
	del att_mask
	decoded = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
	del inputs_embeds
	return decoded

# ask to caption a image
if __name__ == '__main__':

	while True:
		print("Enter the filepath of the image you want to caption.")
		filepath : str = input("")
		answer = caption_image( filepath )
		print(answer)

# C:\Users\Declan\Desktop\test.jpg
