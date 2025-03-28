from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Annotated
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import time
from llm import VertexLLM
from prompt import SYSTEM_PROMPT, INSTRUCTION_PROMPT
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
import typing as t
import logging
import asyncio
from concurrent.futures import ProcessPoolExecutor
logger = logging.getLogger(__name__)
load_dotenv()

app = FastAPI()
llm = VertexLLM()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

executor = ProcessPoolExecutor()

class Request(BaseModel):
    messages: t.List[t.Dict]

def tavily_tool(query):
    response = TavilySearchResults(max_results=8).invoke(query)  
    
    if not isinstance(response, list):
        return []

    lst = []
    for item in response:
        if isinstance(item, tuple) and len(item) == 2:
            item = {"url": item[0], "score": item[1]}

        if isinstance(item, dict) and "score" in item and "url" in item:
            if item["score"] > 0.5:
                lst.append(item["url"])

    return lst

def clean_html(html_content):
    """Removes scripts, styles, and extracts visible text."""
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    return soup.get_text(separator=" ", strip=True)

def get_web_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        return f"Error fetching content: {e}"
    return f"Mock content from {url}"

def get_facebook_content(url, headless=True):
    """Extracts content from Facebook using Selenium."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)

        page_source = driver.page_source
        driver.quit()

        soup = BeautifulSoup(page_source, "html.parser")
        texts = [div.get_text(separator=" ", strip=True) for div in soup.find_all('div')]
        return " ".join(texts)
    except Exception as e:
        return f"Failed to fetch Facebook content: {e}"
    
def extract_info_tool(url: Annotated[str, "The URL to extract information from."]):
    """Extracts text content from a given URL."""
    if "facebook.com" in url or "m.facebook.com" in url:
        return get_facebook_content(url)
    return get_web_content(url)

async def async_extract_info_tool(url: Annotated[str, "The URL to extract information from."]):
    """Extracts text content from a given URL."""
    if "facebook.com" in url or "m.facebook.com" in url:
        return get_facebook_content(url)
    return get_web_content(url)

def extract_info_sync(url: str) -> str:
    """Hàm đồng bộ chọn phương thức phù hợp để trích xuất nội dung."""
    if "facebook.com" in url or "m.facebook.com" in url:
        return get_facebook_content(url)
    return get_web_content(url)


async def async_extract_info_tool_multiprocess(url: str):
    """Chạy extract_info_sync trong một process khác."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, extract_info_sync, url)

@app.post("/generate")
def generate(request: Request):
    messages = request.messages
    query = messages[-1]['content']
    urls: list = tavily_tool(query)
    contents = [extract_info_tool(url) for url in urls]
    print(contents)
    prompt = INSTRUCTION_PROMPT.format(content="/n".join(contents), query=query)
    messages[-1]['content'] = prompt
    response = llm.generate(messages, os.environ["MODEL"])
    return {
        "content": response
    }

@app.post("/stream_generate")
async def generate(request: Request):
    messages = request.messages
    query = messages[-1]['content']
    urls: list = tavily_tool(query)
    contents = await asyncio.gather(*[async_extract_info_tool_multiprocess(url) for url in urls])
    print(contents)
    prompt = INSTRUCTION_PROMPT.format(content="/n".join(contents), query=query)
    messages[-1]['content'] = prompt
    return StreamingResponse(llm.stream_generate(messages, os.environ["MODEL"]), media_type="text/plain")



@app.get("/")
def home():
    return {"message": "Welcome to the Semantic Search API!"}