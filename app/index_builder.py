import os
import re
import json
import sqlite3
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_JSON = "data/specs_parsed.json"
OUTPUT_DOCS_JSON = "data/vector_docs.json"
OUTPUT_VECTORS_NPY = "data/vectors.npy"
OUTPUT_FAISS_INDEX = "data/faiss.index"
OUTPUT_SQLITE_DB = "data/specs.db"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


SPEC_ZH = {
    "OS": "作業系統",
    "CPU": "處理器",
    "Video Graphics": "顯示卡",
    "Display": "螢幕",
    "System Memory": "記憶體",
    "Storage": "儲存",
    "Keyboard Type": "鍵盤",
    "I/O Port": "連接埠",
    "Audio": "音訊",
    "Communications": "通訊",
    "Webcam": "視訊鏡頭",
    "Security": "安全性",
    "Battery": "電池",
    "Adapter": "變壓器",
    "Dimensions (W x D x H)": "尺寸",
    "Weight": "重量",
    "Color": "顏色"
}

SPEC_ALIASES = {
    "OS": ["os", "operating system", "作業系統", "系統"],
    "CPU": ["cpu", "processor", "處理器", "中央處理器"],
    "Video Graphics": ["gpu", "graphics", "video graphics", "顯示卡", "顯卡", "圖形處理器"],
    "Display": ["display", "screen", "panel", "螢幕", "顯示器", "面板"],
    "System Memory": ["memory", "ram", "system memory", "記憶體", "系統記憶體"],
    "Storage": ["storage", "ssd", "disk", "儲存", "硬碟"],
    "Keyboard Type": ["keyboard", "鍵盤"],
    "I/O Port": ["io", "port", "ports", "interface", "連接埠", "接口"],
    "Audio": ["audio", "speaker", "sound", "音訊", "喇叭"],
    "Communications": ["wifi", "wi-fi", "bluetooth", "lan", "network", "通訊", "網路", "無線網路", "藍牙"],
    "Webcam": ["webcam", "camera", "視訊鏡頭", "鏡頭"],
    "Security": ["security", "tpm", "安全性"],
    "Battery": ["battery", "電池"],
    "Adapter": ["adapter", "charger", "變壓器", "充電器"],
    "Dimensions (W x D x H)": ["dimensions", "size", "尺寸", "長寬高"],
    "Weight": ["weight", "重量"],
    "Color": ["color", "colour", "顏色"]
}

SPEC_CATEGORY = {
    "OS": "os",
    "CPU": "cpu",
    "Video Graphics": "gpu",
    "Display": "display",
    "System Memory": "memory",
    "Storage": "storage",
    "Keyboard Type": "keyboard",
    "I/O Port": "ports",
    "Audio": "audio",
    "Communications": "communications",
    "Webcam": "webcam",
    "Security": "security",
    "Battery": "battery",
    "Adapter": "adapter",
    "Dimensions (W x D x H)": "dimensions",
    "Weight": "weight",
    "Color": "color"
}


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def clean_text(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def split_product_name(product_name: str) -> Dict[str, str]:
    parts = product_name.strip().split()
    if len(parts) >= 2:
        return {
            "series": " ".join(parts[:-1]),
            "model": parts[-1]
        }
    return {
        "series": product_name,
        "model": product_name
    }


def build_embedding_text(product_name: str, spec_key_en: str, value_raw: str) -> str:
    spec_key_zh = SPEC_ZH.get(spec_key_en, spec_key_en)
    aliases = SPEC_ALIASES.get(spec_key_en, [])
    alias_text = ", ".join(aliases)

    return (
        f"Product Name: {product_name}\n"
        f"Specification: {spec_key_en} / {spec_key_zh}\n"
        f"Aliases: {alias_text}\n"
        f"Value:\n{value_raw}"
    )


def convert_products_to_docs(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    docs = []

    for product in products:
        product_name = product["product_name"]
        split_info = split_product_name(product_name)
        series = split_info["series"]
        model = split_info["model"]

        for spec_key_en, value_raw in product["specs"].items():
            value_raw = clean_text(value_raw)
            spec_key_zh = SPEC_ZH.get(spec_key_en, spec_key_en)
            spec_category = SPEC_CATEGORY.get(spec_key_en, slugify(spec_key_en))
            aliases = SPEC_ALIASES.get(spec_key_en, [])

            doc = {
                "id": f"{slugify(product_name)}__{slugify(spec_category)}",
                "product_name": product_name,
                "series": series,
                "model": model,
                "spec_key_en": spec_key_en,
                "spec_key_zh": spec_key_zh,
                "spec_category": spec_category,
                "spec_aliases": aliases,
                "value_raw": value_raw,
                "embedding_text": build_embedding_text(product_name, spec_key_en, value_raw)
            }
            docs.append(doc)

    return docs


def build_embeddings(model: SentenceTransformer, docs: List[Dict[str, Any]]) -> np.ndarray:
    texts = [doc["embedding_text"] for doc in docs]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


def save_sqlite(docs: List[Dict[str, Any]], db_path: str) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE documents (
        row_id INTEGER PRIMARY KEY,
        doc_id TEXT NOT NULL,
        product_name TEXT NOT NULL,
        series TEXT,
        model TEXT,
        spec_key_en TEXT NOT NULL,
        spec_key_zh TEXT,
        spec_category TEXT,
        spec_aliases_json TEXT,
        value_raw TEXT NOT NULL,
        embedding_text TEXT NOT NULL
    )
    """)

    cur.execute("CREATE INDEX idx_product_name ON documents(product_name)")
    cur.execute("CREATE INDEX idx_model ON documents(model)")
    cur.execute("CREATE INDEX idx_spec_category ON documents(spec_category)")

    rows = []
    for idx, doc in enumerate(docs):
        rows.append((
            idx,
            doc["id"],
            doc["product_name"],
            doc["series"],
            doc["model"],
            doc["spec_key_en"],
            doc["spec_key_zh"],
            doc["spec_category"],
            json.dumps(doc["spec_aliases"], ensure_ascii=False),
            doc["value_raw"],
            doc["embedding_text"]
        ))

    cur.executemany("""
    INSERT INTO documents (
        row_id, doc_id, product_name, series, model,
        spec_key_en, spec_key_zh, spec_category,
        spec_aliases_json, value_raw, embedding_text
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()


def build_faiss_index(vectors: np.ndarray, index_path: str) -> None:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, index_path)


def main():
    os.makedirs("data", exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)

    docs = convert_products_to_docs(products)

    with open(OUTPUT_DOCS_JSON, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Converted {len(docs)} spec documents.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = build_embeddings(model, docs)

    np.save(OUTPUT_VECTORS_NPY, vectors)
    print(f"[INFO] Saved vectors to {OUTPUT_VECTORS_NPY}, shape={vectors.shape}")

    build_faiss_index(vectors, OUTPUT_FAISS_INDEX)
    print(f"[INFO] Saved FAISS index to {OUTPUT_FAISS_INDEX}")

    save_sqlite(docs, OUTPUT_SQLITE_DB)
    print(f"[INFO] Saved SQLite metadata DB to {OUTPUT_SQLITE_DB}")


if __name__ == "__main__":
    main()