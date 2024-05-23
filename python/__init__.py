
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageFile

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import torch
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# available models
model : str = "deepseek-ai/deepseek-vl-7b-chat" # "deepseek-ai/deepseek-vl-7b-base"

# models and processors
print("Loading Chat Processor")
vl_chat_processor : VLChatProcessor = VLChatProcessor.from_pretrained(model)
tokenizer : AutoTokenizer = vl_chat_processor.tokenizer

print("Loading Model")
vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# caption function
def CLIP_VL_image_caption( image_filepath : str ) -> str:
	if os.path.exists(image_filepath) is False:
		print("Given image does not exist.")
		return None
	with open('python/prompt.txt', 'r') as file:
		base_prompt : str = file.read()
	conversation : list[dict] = [
		{ "role": "User", "content": base_prompt, "images": [image_filepath] },
		{ "role": "Assistant", "content": "" }
	]
	# load images
	pil_images = load_pil_images( conversation )
	pil_images = [ image.resize((512,512)) for image in pil_images ]
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

def clip_directory( directory : str, tag_ext : str = "txt", overwrite_captions : bool = True ) -> None:
	for filename in os.listdir(directory):
		filepath : str = os.path.join(directory, filename)
		if os.path.isfile(filepath) is False:
			continue
		raw_name, _ = os.path.splitext(filename)
		caption_filepath : str = f"{directory}/{raw_name}.{tag_ext}"
		if overwrite_captions is False and os.path.exists(caption_filepath) is True:
			print(f"Caption already exists for {raw_name} under { os.path.split(directory)[-1] } ")
			continue
		try:
			img = Image.open(filepath)
			del img
		except:
			continue
		print(f'Captioning image file {filename} under { os.path.split(directory)[-1] }')
		caption : str = CLIP_VL_image_caption( filepath )
		with open(caption_filepath, 'w') as file:
			file.write(caption.strip())

def clip_subdirs( directory : str, tag_ext : str = "txt" ) -> None:
	for dirname in os.listdir( directory ):
		dirpath : str = os.path.join( directory, dirname )
		if os.path.isdir(dirpath) is False:
			continue
		if os.path.exists(dirpath) is False:
			continue
		print(f"CLIP captioning {dirname}")
		clip_directory(dirpath, tag_ext=tag_ext)

# ask to caption a image
def ask_for_image() -> None:
	while True:
		print("Enter the filepath of the image you want to caption.")
		filepath : str = input("")
		answer = CLIP_VL_image_caption( filepath )
		print(answer)

def caption_directory() -> None:
	while True:
		print("Enter the directory that you want to CLIP caption using a VL-LLM.")
		target_directory : str = input("")
		if os.path.isdir(target_directory) is False:
			print("Invalid directory path.")
			continue
		if os.path.exists(target_directory) is False:
			print("Directory path does not exist.")
			continue
		clip_directory(target_directory, tag_ext="txt")

def caption_subdirs() -> None:
	while True:
		print("Enter the root directory that you want to CLIP caption using a VL-LLM.")
		target_directory : str = input("")
		if os.path.isdir(target_directory) is False:
			print("Invalid directory path.")
			continue
		if os.path.exists(target_directory) is False:
			print("Directory path does not exist.")
			continue
		clip_subdirs(target_directory, tag_ext="txt")

if __name__ == '__main__':
	caption_directory()
	# caption_subdirs()

# C:\Users\Declan\Desktop\test
# C:\Users\Declan\Desktop\Training
