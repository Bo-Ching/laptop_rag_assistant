import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.llm_generator import stream_answer


FAISS_INDEX_PATH = "data/faiss.index"
SQLITE_DB_PATH = "data/specs.db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


SPEC_QUERY_MAP = {
    "cpu": "cpu",
    "processor": "cpu",
    "處理器": "cpu",
    "中央處理器": "cpu",

    "gpu": "gpu",
    "graphics": "gpu",
    "video graphics": "gpu",
    "顯示卡": "gpu",
    "顯卡": "gpu",
    "圖形處理器": "gpu",

    "display": "display",
    "screen": "display",
    "panel": "display",
    "螢幕": "display",
    "顯示器": "display",
    "面板": "display",

    "memory": "memory",
    "ram": "memory",
    "記憶體": "memory",
    "系統記憶體": "memory",

    "storage": "storage",
    "ssd": "storage",
    "disk": "storage",
    "儲存": "storage",
    "硬碟": "storage",

    "keyboard": "keyboard",
    "鍵盤": "keyboard",

    "port": "ports",
    "ports": "ports",
    "io": "ports",
    "interface": "ports",
    "連接埠": "ports",
    "接口": "ports",
    "thunderbolt": "ports",
    "usb": "ports",
    "hdmi": "ports",

    "audio": "audio",
    "speaker": "audio",
    "sound": "audio",
    "音訊": "audio",
    "喇叭": "audio",

    "wifi": "communications",
    "wi-fi": "communications",
    "bluetooth": "communications",
    "lan": "communications",
    "network": "communications",
    "通訊": "communications",
    "網路": "communications",
    "藍牙": "communications",
    "無線網路": "communications",

    "webcam": "webcam",
    "camera": "webcam",
    "鏡頭": "webcam",
    "視訊鏡頭": "webcam",

    "security": "security",
    "tpm": "security",
    "安全性": "security",

    "battery": "battery",
    "電池": "battery",

    "adapter": "adapter",
    "charger": "adapter",
    "變壓器": "adapter",
    "充電器": "adapter",

    "dimensions": "dimensions",
    "size": "dimensions",
    "尺寸": "dimensions",
    "長寬高": "dimensions",

    "weight": "weight",
    "重量": "weight",

    "color": "color",
    "colour": "color",
    "顏色": "color",

    "os": "os",
    "operating system": "os",
    "作業系統": "os",
    "系統": "os"
}


# 全域只載入一次模型與索引
print("[INIT] start loading sentence transformer...")
model_encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("[INIT] sentence transformer loaded")

print("[INIT] start loading faiss index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
print("[INIT] faiss index loaded")

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(SQLITE_DB_PATH)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_model_from_query(query: str) -> Optional[str]:
    m = re.search(r"\b([A-Z]{3,4})\b", query.upper())
    if m:
        token = m.group(1).upper()
        if token not in {"CPU", "GPU", "SSD", "RAM", "OS", "LAN", "WIFI", "USB", "HDMI"}:
            return token
    return None


def extract_spec_category(query: str) -> Optional[str]:
    q = normalize_text(query)
    matches = []

    for key, value in SPEC_QUERY_MAP.items():
        if normalize_text(key) in q:
            matches.append((len(key), value))

    if not matches:
        return None

    matches.sort(reverse=True)
    return matches[0][1]


def row_to_doc(row: tuple) -> Dict[str, Any]:
    return {
        "row_id": row[0],
        "doc_id": row[1],
        "product_name": row[2],
        "series": row[3],
        "model": row[4],
        "spec_key_en": row[5],
        "spec_key_zh": row[6],
        "spec_category": row[7],
        "spec_aliases": json.loads(row[8]) if row[8] else [],
        "value_raw": row[9],
        "embedding_text": row[10]
    }


def load_candidate_rows(
    conn: sqlite3.Connection,
    model: Optional[str],
    spec_category: Optional[str]
) -> List[Dict[str, Any]]:
    sql = """
    SELECT row_id, doc_id, product_name, series, model,
           spec_key_en, spec_key_zh, spec_category,
           spec_aliases_json, value_raw, embedding_text
    FROM documents
    """
    conditions = []
    params = []

    if model:
        conditions.append("UPPER(model) = ?")
        params.append(model.upper())

    if spec_category:
        conditions.append("spec_category = ?")
        params.append(spec_category)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    cur = conn.cursor()
    rows = cur.execute(sql, params).fetchall()
    return [row_to_doc(row) for row in rows]


def build_context(results: List[Tuple[float, Dict[str, Any]]]) -> str:
    chunks = []

    for rank, (score, doc) in enumerate(results, start=1):
        chunk = (
            f"[Top {rank} | score={score:.4f}]\n"
            f"Product Name: {doc['product_name']}\n"
            f"Series: {doc['series']}\n"
            f"Model: {doc['model']}\n"
            f"Spec Category: {doc['spec_category']}\n"
            f"Spec Key EN: {doc['spec_key_en']}\n"
            f"Spec Key ZH: {doc['spec_key_zh']}\n"
            f"Value: {doc['value_raw']}\n"
        )
        chunks.append(chunk)

    return "\n" + ("-" * 80 + "\n").join(chunks)


def search_with_faiss_subset(
    model_encoder: SentenceTransformer,
    faiss_index: faiss.Index,
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5
) -> List[Tuple[float, Dict[str, Any]]]:
    print('searching docs')
    detected_model = extract_model_from_query(query)
    detected_spec = extract_spec_category(query)

    print(f"[DEBUG] detected_model={detected_model}, detected_spec={detected_spec}")

    candidates = load_candidate_rows(conn, detected_model, detected_spec)

    if not candidates and detected_model:
        candidates = load_candidate_rows(conn, detected_model, None)

    if not candidates and detected_spec:
        candidates = load_candidate_rows(conn, None, detected_spec)

    if not candidates:
        candidates = load_candidate_rows(conn, None, None)

    query_vec = model_encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    total_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # 候選等於全資料庫時，直接用 FAISS 搜
    if len(candidates) == total_count:
        scores, indices = faiss_index.search(query_vec, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            row = conn.execute("""
                SELECT row_id, doc_id, product_name, series, model,
                       spec_key_en, spec_key_zh, spec_category,
                       spec_aliases_json, value_raw, embedding_text
                FROM documents
                WHERE row_id = ?
            """, (int(idx),)).fetchone()

            if row is None:
                continue

            results.append((float(score), row_to_doc(row)))

        return results

    # subset 時直接算 cosine similarity
    candidate_texts = [doc["embedding_text"] for doc in candidates]
    candidate_vecs = model_encoder.encode(
        candidate_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores = np.dot(candidate_vecs, query_vec[0])
    ranked_idx = np.argsort(-scores)[:top_k]

    return [(float(scores[i]), candidates[i]) for i in ranked_idx]


def answer_query(query: str, top_k: int = 3) -> None:
    print("[ANSWER] answer_query entered")
    conn = get_connection()

    try:
        print("[RAG] before retrieval")
        results = search_with_faiss_subset(
            model_encoder=model_encoder,
            faiss_index=faiss_index,
            conn=conn,
            query=query,
            top_k=top_k
        )
        print("[RAG] retrieval done")

        print("=" * 100)
        print(f"Query: {query}")
        print("=" * 100)

        if not results:
            print("找不到相關規格。")
            return

        for rank, (score, doc) in enumerate(results, start=1):
            print(f"[Top {rank}] score={score:.4f}")
            print(f"Product   : {doc['product_name']}")
            print(f"Model     : {doc['model']}")
            print(f"Spec EN   : {doc['spec_key_en']}")
            print(f"Spec ZH   : {doc['spec_key_zh']}")
            print("Value:")
            print(doc["value_raw"])
            print("-" * 100)

        context = build_context(results)

        print("Answer:")
        print("[RAG] before generation")
        answer, metrics = stream_answer(query, context)
        

        print("\n")
        print("TTFT:", metrics["ttft"])
        print("TPS:", metrics["tps"])

    finally:
        conn.close()


if __name__ == "__main__":
    print("[MAIN] script started")
    test_queries = [
        "AORUS MASTER 16 BXH 的 CPU 是什麼？",
        "BXH 的顯卡是什麼？",
        "Does BXH support Wi-Fi 7?",
        "What is the weight of BXH?",
        "BXH 有 Thunderbolt 5 嗎？"
    ]

    for q in test_queries:
        answer_query(q, top_k=3)
        print()