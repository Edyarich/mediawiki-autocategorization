#!/bin/sh
python3.9 -m venv .venv39
.venv39/bin/python -m  pip install  --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt 
.venv39/bin/python -m  pip install bigartm10-0.10.1-cp39-cp39-linux_x86_64.whl

curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma2

