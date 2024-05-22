@echo off
git clone https://github.com/deepseek-ai/DeepSeek-VL
cd DeepSeek-VL
pip install -r requirements.txt
echo Move deepseek_vl into your codebase. Press any key to continue afterwards.
read
cd ..