
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from PIL import Image, ImageFile

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import time
import torch
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# # deepseek-ai/deepseek-vl-7b-chat
# model : str = "deepseek-ai/deepseek-vl-7b-chat"
# print("Loading Chat Processor")
# vl_chat_processor : VLChatProcessor = VLChatProcessor.from_pretrained(model)
# tokenizer : AutoTokenizer = vl_chat_processor.tokenizer
# print("Loading Model")
# vl_gpt : MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
# vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
# def CLIP_VL_image_caption( image_filepath : str ) -> str:
# 	if os.path.exists(image_filepath) is False:
# 		print("Given image does not exist.")
# 		return None
# 	with open('python/prompt.txt', 'r') as file:
# 		base_prompt : str = file.read()
# 	conversation : list[dict] = [
# 		{ "role": "User", "content": base_prompt, "images": [image_filepath] },
# 		{ "role": "Assistant", "content": "" }
# 	]
# 	# load images
# 	pil_images = load_pil_images( conversation )
# 	pil_images = [ image.resize((1024,1024)) for image in pil_images ]
# 	# prepare inputs
# 	prepare_inputs = vl_chat_processor( conversations=conversation, images=pil_images, force_batchify=True ).to(vl_gpt.device)
# 	del pil_images
# 	# run image encoder to get the image embeddings
# 	inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
# 	att_mask = prepare_inputs.attention_mask
# 	del prepare_inputs
# 	# run the model to get the response
# 	with torch.inference_mode():
# 		outputs = vl_gpt.language_model.generate(
# 			inputs_embeds=inputs_embeds,
# 			attention_mask=att_mask,
# 			pad_token_id=tokenizer.eos_token_id,
# 			bos_token_id=tokenizer.bos_token_id,
# 			eos_token_id=tokenizer.eos_token_id,
# 			max_new_tokens=512,
# 			do_sample=False,
# 			use_cache=True
# 		)
# 	del att_mask
# 	decoded = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# 	del inputs_embeds
# 	return decoded

# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) #, load_in_4bit=True)
# model.to("cuda:0")
# def CLIP_VL_image_caption( image_filepath : str ) -> str:
# 	with open('python/prompt.txt', 'r') as file:
# 		prompt : str = file.read()
# 		prompt = prompt.replace("<image_placeholder>", "<image>")
# 		prompt = f"[INST] {prompt} [/INST]"
# 	image = Image.open(image_filepath)
# 	inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
# 	with torch.inference_mode():
# 		output = model.generate(**inputs, max_new_tokens=100)
# 	del inputs
# 	return processor.decode(output[0], skip_special_tokens=True)

model_name = "Salesforce/instructblip-vicuna-7b" # "Salesforce/instructblip-flan-t5-xl"
processor = InstructBlipProcessor.from_pretrained(model_name)
model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def caption_raw_image(
	image: Image.Image,
	prompt: str,
	text_decoding_method: str = "Beam search",
	num_beams: int = 16,
	max_length: int = 256,
	min_length: int = 10,
	top_p: float = 1,
	repetition_penalty: float = 1.5,
	length_penalty: float = 1.0,
	temperature: float = 1,
) -> str:
	h, w = image.size
	scale = 1024 / max(h, w)
	if scale < 1:
		new_w = int(w * scale)
		new_h = int(h * scale)
		image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
	inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda", torch.float16)
	with torch.inference_mode():
		generated_ids = model.generate(
			**inputs,
			do_sample=text_decoding_method=="Nucleus sampling",
			num_beams=num_beams,
			max_length=max_length,
			min_length=min_length,
			top_p=top_p,
			repetition_penalty=repetition_penalty,
			length_penalty=length_penalty,
			temperature=temperature,
		)
		return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def CLIP_VL_image_caption(image_filepath: str) -> str:
	return caption_raw_image( Image.open(image_filepath), "" )

def clip_directory( directory : str, tag_ext : str = "txt", overwrite_captions : bool = True ) -> None:
	items = os.listdir(directory)
	last : float = time.time()
	for index, filename in enumerate(items):
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
		print(f'Captioning image {index+1}/{len(items)} - {filename} under { os.path.split(directory)[-1] }')
		caption : str = CLIP_VL_image_caption( filepath )
		with open(caption_filepath, 'w') as file:
			file.write(caption.strip())
		now : float = time.time()
		print(f"Estimating { round((now-last) * (len(items) - index), 1) } seconds left.")
		last = now

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
