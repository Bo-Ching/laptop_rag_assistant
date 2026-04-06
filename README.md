# 消費級筆電AI助手
本專案為一個輕量級的Retrieval-Augmented Generation系統，
目標是從筆電產品的規格頁中，建立可準確回答規格問題的問答系統。


## 專案目標
建立一個系統，功能包含: 
- 精準回答筆電規格問題，如: CPU, GPU, Wi-Fi等
- 支援中英文問答
- 能夠在低資源環境下運行(4GB VRAM)
- 不依賴LangChain, LlamaIndex等高階框架
  
## 專案設定
- Embedding Model: paraphrase-multilingual-MiniLM-L12-v2
- Retrieval: FAISS + SQLite filtering
- Generation
  - Model: Qwen2.5-1.5B-Instruct
  - Quantization: Q4_K_M (4-bit GGUF)
  - Inference Backend: llama.cpp Server

## 模型選擇理由
### 模型選擇考量
本系統任務為產品規格問答，因問題類型固定、高度結構化以及不需要複雜推理等原因，因此採取以下策略:
- 以Retrieval為核心，採用FAISS向量搜尋以及SQLite結構化過濾，縮小正確答案範圍，生成模型的任務僅為從檢索結果中抽取關鍵資訊，並依照使用者問題進行語句重組及回答。
- Embedding Model採用paraphrase-multilingual-MiniLM-L12-v2，保持輕量化及支援中英文查詢
- Generation Model採用Qwen2.5-1.5B-Instruct，因其具備基本語言理解及摘要能力。並採用4-bit量化，在不影響任務正確率的前提下，大幅降低記憶體需求。
  
### 推估VRAM使用量分析

| 組件 | 說明 | VRAM 使用量 |
|------|------|------------|
| 模型權重（Q4_K_M） | 1.5B 量化模型載入 | 約 0.8 ~ 1.2 GB |
| KV Cache | 推理過程中儲存 token 狀態 | 約 0.5 ~ 1.0 GB |
| 推理框架 | llama.cpp runtime buffer | 約 0.3 ~ 0.8 GB |
| **總計（峰值）** | - | **約 1.6 ~ 3.0 GB** |


## 評測分析
| Query | Top1 Score | Answer | TTFT (s) | TPS |
|------|-----------|--------|----------|-----|
| AORUS MASTER 16 BXH 的 CPU 是什麼？ | 0.8326 | AORUS MASTER 16 BXH 的 CPU 是 Intel® Core™ Ultra 9 Processor 275HX。 | 1.52 | 13.76 |
| BXH 的顯卡是什麼？ | 0.2945 | BXH 的顯卡是 NVIDIA® GeForce RTX™ 5070 Ti Laptop GPU，具備 12GB GDDR7 與 140W 功率。 | 0.46 | 11.64 |
| Does BXH support Wi-Fi 7? | 0.5515 | 我找不到足夠資訊回答這個問題。 | 5.28 | 12.86 |
| What is the weight of BXH? | 0.6546 | BXH 的重量約為 2.5 公斤。 | 5.71 | 7.62 |
| BXH 有 Thunderbolt 5 嗎？ | 0.4500 | BXH 有 Thunderbolt 5。 | 3.82 | 12.94 |
-  TTFT 約落在 0.4s ~ 5.7s
-  TPS 約為 7 ~ 13 tokens/sec

## 啟動步驟
1. clone專案
2. 安裝uv
    ```bash
    pip install uv
    ```
3. 安裝相依套件
    ```bash
    uv sync
    source .venv/bin/activate
    ```
4. 下載模型並放入models資料夾
5. 安裝並啟動llama.cpp
   1. clone llama.cpp
        ```bash
        git clone https://github.com/ggerganov/llama.cpp
        cd llama.cpp
        ```
    2. 編譯
        ```bash
        make
        ```
    3. 啟動LLM server
        ```bash
        ./server -m ../models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf -c 2048
        ```
6. 執行RAG系統
    ```python
    uv run python rag_query.py
    ```

