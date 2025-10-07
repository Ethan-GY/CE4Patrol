import base64
import requests
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_qwen_vl_api(api_key, image_path, text_prompt):
    """调用通义千问VL API的函数"""
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "qwen-vl-plus", # 或 qwen-vl-max
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{base64_image}"},
                        {"text": text_prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "result_format": "message"
        }
    }
    
    try:
        response = requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation", headers=headers, json=payload)
        response.raise_for_status()
        # 解析返回的JSON以提取模型输出文本
        result_text = response.json()['output']['choices'][0]['message']['content']
        # 提取JSON部分
        json_part = result_text[result_text.find('{'):result_text.rfind('}')+1]
        return json.loads(json_part)
    except Exception as e:
        print(f"API call failed for {image_path}: {e}")
        return None