from openai import OpenAI
import sys
import os

import json
import datetime

from Generator_Agent.template import *
import random


client=OpenAI(api_key=api_key)


def create_prompt_system_prompt(Tg:float, Er:float, Group1:str, Group2:str):
    if Tg != None and Er != None:
        system_prompt = property_focused_system_prompts[0]
        
    elif Group1 != None and Group2 != None:
           
        if group1 == "vinyl(C=C)" and group2 == "vinyl(C=C)":
            system_prompt = vinyl_system_prompts[0]
        elif group1 == "epoxy(C1OC1)" and group2 == "imine(NC)":
            system_prompt = epoxy_imine_system_prompts[0]
        elif group1 == "imine(NC)" and group2 == "epoxy(C1OC1)":
            system_prompt = epoxy_imine_system_prompts[0]
        elif group1 == "vinyl(C=C)" and group2 == "thiol(CCS)":
            system_prompt = thiol_ene_system_prompts[0]
        elif group1 == "thiol(CCS)" and group2 == "vinyl(C=C)":
            system_prompt = thiol_ene_system_prompts[0]
        elif group1 == "vinyl(C=C)" and group2 == "hydroxyl(=O)":
            system_prompt = hydroxyl_system_prompts[0]
        elif group1 == "hydroxyl(=O)" and group2 == "vinyl(C=C)":
            system_prompt = hydroxyl_system_prompts[0]
        elif group1 == "acrylate(C=C(C=O))" and group2 == "vinyl(C=C)":
            system_prompt = acrylate_vinyl_system_prompts[0]
        elif group1 == "vinyl(C=C)" and group2 == "acrylate(C=C(C=O))":
            system_prompt = acrylate_vinyl_system_prompts[0]
        else:
            system_prompt = mixed_functionality_system_prompts[0]
        
    elif Tg != None and Er != None and Group1 != None and Group2 != None:
        system_prompt = mixed_functionality_system_prompts[0]
        
    else:
        system_prompt = property_focused_system_prompts[0]
    return system_prompt



  

def generate_samples(Tg:float, Er:float, Group1:str, Group2:str):
    query = "Generate a TSMP with Tg = {Tg}°C, Er = {Er} MPa with Group1 = {Group1} in monomer1, Group2 = {Group2} in monomer2"
    system_prompt = create_prompt_system_prompt(Tg, Er, Group1, Group2)
    user_prompt = query.format(Tg=Tg, Er=Er, Group1=Group1, Group2=Group2)
    
    messages = []
    messages.append({"role":"system","content":system_prompt})
    messages.append({"role":"user","content":user_prompt})
    temperatures =[0.3,0.5,0.7,0.9,1.0]
    temperature = random.choice(temperatures)
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=temperature,
        max_tokens=300,
        n=1)
    output_content = completion.choices[0].message.content
    return output_content





    







