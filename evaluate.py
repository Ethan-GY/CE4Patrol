# -*- coding: utf-8 -*-
import os
import requests
import base64
from dotenv import load_dotenv
from PIL import Image
import json
import io

load_dotenv()
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
siliconflow_api_URL = "https://api.siliconflow.cn/v1/chat/completions"

if not siliconflow_api_key:
    raise ValueError("SILICONFLOW_API_KEY not found in environment variables")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_vlm_api(model_name, prompt, image_path):
    print(f"\n---Calling model:{model_name}")
    base64_image = encode_image(image_path)
    if not base64_image:
        raise None

    headers = {
        "Authorization": f"Bearer {siliconflow_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages":[
        { 
        "role": "system", 
        "content": "你是一个图片描述小助手,帮助用户理解图片,回答问题" 
        },
        {
        "role": "user",
        "content":[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"                }

            }       
            ]
        }
        ],
        "max_tokens":300,
        "temperature":0.1
    }
    response = requests.post(siliconflow_api_URL, headers=headers, json=payload)
    return response.json()
   

if __name__ == "__main__":
    models_to_test = [
        "THUDM/GLM-4.1V-9B-Thinking",
        "Pro/Qwen/Qwen2.5-VL-7B-Instruct"
    ]
    img_path = "assets/test_fire.jpg"
    test_prompt = "图中有什么？有没有火灾？请详细描述"
    
    for model in models_to_test:
        print(f"Testing model: {model}")
        response = call_vlm_api(model, test_prompt, img_path)
        
        if response:
            print("API Response:")
            content = response.get('choices', [{}])[0].get('message', {}).get('content', 'No content in response')
            print(content)
            print("-"*20)
        else:
            print(f"Failed to get response from {model}")
        