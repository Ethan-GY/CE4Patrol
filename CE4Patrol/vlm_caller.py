"""
VLM caller module with caching and retry mechanisms.
"""
import os
import json
import logging
import base64
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

class VLMCaller:
    """视觉语言模型调用器 (带缓存和重试)"""
    def __init__(self, cache_dir: str = ".vlm_cache"):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("CE4_API_KEY")
        self.api_base = os.getenv("CE4_API_BASE")
        self.model = os.getenv("CE4_MODEL")

        if not all([self.api_key, self.api_base, self.model]):
            raise ValueError("Missing required env vars: CE4_API_KEY, CE4_API_BASE, CE4_MODEL")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger.info(f"VLM cache enabled at: {self.cache_dir.absolute()}")

    def _get_cache_key(self, prompt: str, image_path: str) -> str:
        """生成唯一的缓存键"""
        image_bytes = Path(image_path).read_bytes()
        hasher = hashlib.sha256()
        hasher.update(prompt.encode('utf-8'))
        hasher.update(image_bytes)
        return hasher.hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_vlm(self, prompt: str, image_path: str) -> Dict[str, Any]:
        """调用VLM，优先从缓存读取"""
        cache_key = self._get_cache_key(prompt, image_path)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            self.logger.info(f"Cache hit for image {Path(image_path).name}. Loading from cache.")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        self.logger.info(f"Cache miss for image {Path(image_path).name}. Calling VLM API...")
        try:
            if not Path(image_path).exists():
                return {"error": f"Image file not found: {image_path}"}
            
            image_data = self._encode_image(image_path)
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1, max_tokens=2000
            )
            content = response.choices[0].message.content
            parsed_response = self._parse_json_response(content)
            
            result = parsed_response if parsed_response else {"raw_response": content, "error": "Failed to parse JSON response"}
            
            # 写入缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return result
        except Exception as e:
            self.logger.error(f"VLM call failed after retries: {e}")
            return {"error": str(e)}

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """采纳您强大的多策略JSON解析器"""
        try:
            return json.loads(content)
        except json.JSONDecodeError: pass
        try:
            start_idx, end_idx = content.find('{'), content.rfind('}')
            if start_idx != -1 and end_idx != -1: return json.loads(content[start_idx:end_idx + 1])
        except json.JSONDecodeError: pass
        try:
            if '```json' in content:
                json_str = content.split('```json')[1].split('```')[0].strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError): pass
        return None