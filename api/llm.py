import json
import typing as t
from litellm import completion
class VertexLLM:
    def __init__(self):
        with open(r"..\account.json", 'r') as f:
            self.credentials = json.load(f)
    
    def generate(self, messages:t.List[t.Dict], model):
        response = completion(
            model=model,
            messages=messages,
            vertex_credentials=self.credentials,
            max_tokens=8192
        )
        return response.choices[0].message.content
    
    async def stream_generate(self, messages:t.List[t.Dict], model):
        response = completion(
            model=model,
            messages=messages,
            vertex_credentials=self.credentials,
            stream=True
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield json.dumps({
                "chunk":content
                })