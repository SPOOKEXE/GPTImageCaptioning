
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from typing import Literal

import torch
import os

# available models
model : str = "deepseek-ai/deepseek-vl-7b-chat" # "deepseek-ai/deepseek-vl-7b-base"
base_prompt : str = """<image_placeholder>You are the best captioning bot in the world. Caption this image in the style of CLIP. Include worn clothing, pose, estimated age, estimated ethnicity, if its explicit, quesitonable or explicit. Exclude background objects like furniture but include big-picture subjects like 'pools' or 'bedroom'. Only caption the first image. If there is no person in the image, simply output 'No person in image.'"""

# models and processors
vl_chat_processor : VLChatProcessor = VLChatProcessor.from_pretrained(model)
tokenizer : AutoTokenizer = vl_chat_processor.tokenizer

vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# caption function
def caption_image( image_filepath : str ) -> str:
	conversation : list[dict] = [
		{
			"role": "User",
			"content": base_prompt,
			"images": [image_filepath]
		},
		{
			"role": "Assistant",
			"content": ""
		}
	]
	# load images and prepare for inputs
	pil_images = load_pil_images( conversation )
	prepare_inputs = vl_chat_processor( conversations=conversation, images=pil_images, force_batchify=True ).to(vl_gpt.device)
	# run image encoder to get the image embeddings
	inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
	# run the model to get the response
	outputs = vl_gpt.language_model.generate(
		inputs_embeds=inputs_embeds,
		attention_mask=prepare_inputs.attention_mask,
		pad_token_id=tokenizer.eos_token_id,
		bos_token_id=tokenizer.bos_token_id,
		eos_token_id=tokenizer.eos_token_id,
		max_new_tokens=512,
		do_sample=False,
		use_cache=True
	)

	answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
	return f"{prepare_inputs['sft_format'][0]}", answer

# ask to caption a image
if __name__ == '__main__':
	print("Enter the filepath of the image you want to caption.")
	filepath : str = input("")

	if os.path.exists(filepath) is False:
		exit()

	formatt, answer = caption_image( filepath )
	print(formatt)
	print("=======")
	print(answer)
