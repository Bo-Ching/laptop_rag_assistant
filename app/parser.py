from bs4 import BeautifulSoup
from pathlib import Path
import json
import re


INPUT_PATH = Path("data/view-source_https___www.gigabyte.com_Laptop_AORUS-MASTER-16-AM6H_sp.html")
OUTPUT_PATH = Path("data/specs_parsed.json")

def clean_text(text: str) -> str:
    """整理文字，去掉多餘空白與空行。"""
    if not text:
        return ""

    # 換行後逐行清理
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    # 合併連續空白
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def reconstruct_raw_html(view_source_html: str) -> str:
    """
    這個檔案是 Chrome/瀏覽器的 view-source 頁面，
    真正原始碼在 <td class="line-content"> 裡。
    這裡先把每一行 source 還原回真正 HTML。
    """
    soup = BeautifulSoup(view_source_html, "html.parser")
    line_cells = soup.select("td.line-content")

    if not line_cells:
        raise ValueError("找不到 td.line-content，這份檔案看起來不是 view-source 格式。")

    raw_lines = []
    for cell in line_cells:
        # 這裡用 get_text() 是對的，因為 line-content 裡顯示的就是原始碼文字
        raw_lines.append(cell.get_text("", strip=False))

    return "\n".join(raw_lines)


def extract_model_names(parsed_source: BeautifulSoup) -> list[str]:
    """
    從頁面上方 subtitle 抓型號順序，例如：
    AORUS MASTER 16 BZH / AORUS MASTER 16 BYH / AORUS MASTER 16 BXH
    """
    subtitle_el = parsed_source.select_one(".model-base-info-subtitle")
    if subtitle_el:
        subtitle_text = clean_text(subtitle_el.get_text(" ", strip=True))
        if "/" in subtitle_text:
            names = [part.strip() for part in subtitle_text.split("/") if part.strip()]
            if names:
                return names

    # fallback：如果抓不到，就用 product_1, product_2...
    slides = parsed_source.select("div.multiple-content-swiper .swiper-slide")
    return [f"product_{i+1}" for i in range(len(slides))]


def extract_spec_titles(parsed_source: BeautifulSoup) -> list[str]:
    """
    左側欄位標題，例如 OS / CPU / Video Graphics / Display ...
    """
    titles = []
    for ul in parsed_source.select("ul.spec-item-list"):
        title_el = ul.select_one(".spec-title")
        if title_el:
            title = clean_text(title_el.get_text(" ", strip=True))
            if title:
                titles.append(title)
    return titles


def extract_product_specs(parsed_source: BeautifulSoup, spec_titles: list[str], model_names: list[str]) -> list[dict]:
    """
    每個 swiper-slide 對應一個產品型號。
    slide 裡的 div.spec-item-list[data-spec-row] 對應每一列規格值。
    """
    slides = parsed_source.select("div.multiple-content-swiper .swiper-slide")
    results = []

    for idx, slide in enumerate(slides):
        product_name = model_names[idx] if idx < len(model_names) else f"product_{idx+1}"

        spec_items = slide.select("div.spec-item-list[data-spec-row]")
        specs = {}

        # 依 data-spec-row 排序，確保和左側 title 對齊
        spec_items_sorted = sorted(
            spec_items,
            key=lambda x: int(x.get("data-spec-row", "9999"))
        )

        for row_idx, item in enumerate(spec_items_sorted):
            value = clean_text(item.get_text("\n", strip=True))
            if not value:
                continue

            if row_idx < len(spec_titles):
                key = spec_titles[row_idx]
            else:
                # 保底，避免網站改版後 row 數量不一致
                key = f"row_{row_idx}"

            specs[key] = value

        results.append({
            "product_name": product_name,
            "specs": specs
        })

    return results


def main():
    input_path = INPUT_PATH
    output_path = OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    view_source_html = input_path.read_text(encoding="utf-8")

    # Step 1: 還原真正 HTML
    raw_html = reconstruct_raw_html(view_source_html)

    # Step 2: 再 parse 一次真正 HTML
    parsed_source = BeautifulSoup(raw_html, "html.parser")

    # Step 3: 抓左側規格標題
    spec_titles = extract_spec_titles(parsed_source)

    # Step 4: 抓型號名稱順序
    model_names = extract_model_names(parsed_source)

    # Step 5: 抓每個型號的規格
    products = extract_product_specs(parsed_source, spec_titles, model_names)

    # 額外輸出 debug 資訊
    print(f"Found {len(spec_titles)} spec titles")
    print(f"Found {len(model_names)} model names: {model_names}")
    print(f"Found {len(products)} products")

    for product in products:
        print("=" * 80)
        print(product["product_name"])
        for k, v in product["specs"].items():
            preview = v.replace("\n", " | ")
            print(f"{k}: {preview[:120]}")

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"\nSaved parsed specs to: {output_path}")


if __name__ == "__main__":
    main()