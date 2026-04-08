
#!/bin/bash
source ~/.config/secrets.env
source /home/mauceric/Secretarius/lora_local/llenv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root 
