# discord-vicuna

Chat with Vicuna-13b in Discord.

Model I use (put in ```/models```):\
https://huggingface.co/jeremy-costello/vicuna-13b-v1.1-4bit-128g

Install:
- ```conda create -n discord-vicuna python=3.11```
- ```conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia```
- ```pip install -r requirements.txt```

How to use:
- Create a Discord bot account
  - https://docs.pycord.dev/en/stable/discord.html
- Put your bot token in a file called ```discord_token.txt```

What it can do:
- Chat history automatically added to context
- Input custom context
- Load context from Wikipedia

Commands:
- ```--load_model``` to load the model (requires ~12GB of VRAM)
- ```--request $INPUT``` to chat
- more can be found in ```vicuna_bot.py```
