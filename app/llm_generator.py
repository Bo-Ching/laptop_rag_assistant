import json
import time
import requests

LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

GENERATION_PROMPT = """
你是一個筆電規格問答助手。
你只能根據提供的檢索內容回答。
如果檢索內容不足以回答，就回答：我找不到足夠資訊回答這個問題。
不要自行補充未提供的規格資訊。
"""


def build_messages(query: str, context: str):
    return [
        {"role": "system", "content": GENERATION_PROMPT},
        {
            "role": "user",
            "content": f"使用者問題：\n{query}\n\n檢索內容：\n{context}\n\n請用繁體中文回答。"
        }
    ]


def stream_answer(query: str, context: str) -> tuple[str, dict]:
    payload = {
        "messages": build_messages(query, context),
        "max_tokens": 200,
        "temperature": 0.1,
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    generated_text = []
    token_count = 0

    with requests.post(LLAMA_URL, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")

            if not line.startswith("data: "):
                continue

            data_str = line[len("data: "):].strip()

            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            delta = data["choices"][0].get("delta", {})
            content = delta.get("content", "")

            if content:
                # 🔥 TTFT 計算
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                print(content, end="", flush=True)
                generated_text.append(content)
                token_count += 1

    end_time = time.perf_counter()

    answer = "".join(generated_text).strip()

    #Metrics
    ttft = first_token_time - start_time if first_token_time else None
    generation_time = end_time - first_token_time if first_token_time else None
    tps = token_count / generation_time if generation_time and token_count > 0 else None

    metrics = {
        "ttft": ttft,
        "tps": tps,
        "tokens": token_count,
        "generation_time": generation_time
    }

    return answer, metrics


if __name__ == "__main__":
    test_query = "BXH 的 CPU 是什麼？"
    test_context = """Product: AORUS MASTER 16 BXH
    Specification: CPU / 處理器
    Value:
    Intel® Core™ Ultra 9 Processor 275HX (36MB cache, up to 5.4 GHz, 24 cores, 24 threads)
    """

    print("Streaming Answer:\n")
    answer, metrics = stream_answer(test_query, test_context)

    print("\n\nMetrics:")
    print(metrics)