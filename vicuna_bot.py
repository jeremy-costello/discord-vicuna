import time
import torch
from transformers import AutoTokenizer

import discord
from discord.ext import commands

from gptq_for_llama.datautils import *
from gptq_for_llama.modelutils import *
from gptq_for_llama.llama_inference import load_quant

'''
TODO
- lock model
- add context
'''


DISCORD_TOKEN = "MTA5NzE3MTE2NzA1NzU1MTQ1MA.GjR2P_.EPe8L_yA_r6q2WFaFE7ERkZQfl3o4xVow2dkXI"

command_prefix = "--"

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True


class LLaMaBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def on_ready(self):
        print("Logged in!")


class LLaMaModel:
    def __init__(self):
        model_name = "vicuna-13b-v1.1-4bit-128g"
        model_extension = "pt"
        
        self.model = None
        self.tokenizer = None
        self.variables = {
            "model_folder": f"models/{model_name}",
            "load": f"models/{model_name}/{model_name}.{model_extension}",
            "wbits": 4,
            "groupsize": 128,
            "device": 0,
            "min_length": 10,  # 10
            "max_length": 512,  # 50
            "top_p": 0.95,
            "temperature": 0.8
        }
        
        self.locked = False
        self.context = None
        self.history = None


bot = LLaMaBot(command_prefix=command_prefix,
               intents=intents)

llama_model = LLaMaModel()

    
@bot.command()
async def load_model(ctx):
    if llama_model.model is None and llama_model.tokenizer is None:
        await ctx.send("Loading model!")
        start_time = time.time()
        llama_model.model = load_quant(llama_model.variables["model_folder"],
                                       llama_model.variables["load"],
                                       llama_model.variables["wbits"],
                                       llama_model.variables["groupsize"],
                                       llama_model.variables["device"])
        llama_model.model.to(DEV)
        
        await ctx.send("Loading tokenizer!")
        llama_model.tokenizer = AutoTokenizer.from_pretrained(llama_model.variables["model_folder"],
                                                              use_fast=False)
        
        run_time = round(time.time() - start_time, 1)
        await ctx.send(f"Model loaded in {run_time} seconds!")
    else:
        await ctx.send("Model is already loaded!")

@bot.command()
async def unload_model(ctx):
    llama_model.model = None
    llama_model.tokenizer = None
    
    await ctx.send("Model unloaded!")

# convert to correct type
'''
@bot.command()
async def update_variable(ctx, key, value):
    if key in llama_model.variables.keys():
        llama_model.variables[key] = value
        msg = f"Set {key} to {value}."
    else:
        msg = f"Invalid key: {key}."
    await ctx.send(msg)
'''

@bot.command()
async def request(ctx, *user_input):
    if llama_model.model is None and llama_model.tokenizer is None:
        await ctx.send("Model is not loaded!")
    elif llama_model.locked:
        await ctx.send("Model is currently in use, please wait.")
    else:     
        llama_model.locked = True
        user_input = " ".join(list(user_input))
        text = f"Human: {user_input}</s>Assistant:"
        
        if llama_model.history is not None:
            text = llama_model.history + "\n" + text
        elif llama_model.context is not None:
            text = llama_model.context + "\n" + text
        
        input_ids = llama_model.tokenizer.encode(text, return_tensors="pt").to(DEV)

        with torch.no_grad():
            generated_ids = llama_model.model.generate(
                input_ids,
                do_sample=True,
                min_length=llama_model.variables["min_length"],
                max_length=llama_model.variables["max_length"],
                top_p=llama_model.variables["top_p"],
                temperature=llama_model.variables["temperature"]
            )

        response = llama_model.tokenizer.decode([el.item() for el in generated_ids[0]])
        llama_model.history = response.replace("<s>", "").strip()
        
        response_strip = response.split("</s>")[-2].strip().lstrip("Assistant:").strip()
        await ctx.send(response_strip)
        
        llama_model.locked = False

@bot.command()
async def custom_context(ctx, *context):
    llama_model.context = " ".join(list(context))
    await ctx.send(f"Context added:\n\n{llama_model.context}")

@bot.command()
async def wikipedia_context(ctx):
    pass

@bot.command()
async def clear_context(ctx):
    llama_model.context = None
    await ctx.send("Context removed!")

@bot.command()
async def print_history(ctx):
    await ctx.send(llama_model.history)

@bot.command()
async def clear_history(ctx):
    llama_model.history = None
    await ctx.send("History cleared!")


bot.run(DISCORD_TOKEN)
