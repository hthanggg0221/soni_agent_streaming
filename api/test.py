import httpx

def test_streaming():
    url = "http://localhost:8000/stream_generate"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "Sự kiện Innovation Day 2025 do câu lạc bộ nào tổ chức"
            }
        ]
    }

    with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        print("Streaming response:\n")
        for line in response.iter_text():
            print(line, end="", flush=True)

test_streaming()