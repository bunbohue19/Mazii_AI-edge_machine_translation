import httpx
import json
from tqdm import tqdm
import re
import os

os.makedirs("/workspace/Mazii_AI-edge_machine_translation/data/test/output/jsonl", exist_ok=True)

# Cấu hình
API_URL = "http://localhost:8888/v1/translate"
INPUT_FILE = "/workspace/Mazii_AI-edge_machine_translation/data/test/input/vi_ja.jsonl"
OUTPUT_FILE = "/workspace/Mazii_AI-edge_machine_translation/data/test/output/jsonl/result_vi_ja.jsonl"
TARGET_LANG_CODE = "Japanese"

def run_translation():
    # 1. Kiểm tra tiến độ đã hoàn thành (Resume logic)
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
    
    print(f"Đã tìm thấy {processed_count} câu đã dịch. Sẽ bắt đầu từ câu {processed_count + 1}...")

    # 2. Đếm tổng số câu của file gốc
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 3. Mở file ghi (append) và file đọc
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as out_f, \
         open(INPUT_FILE, 'r', encoding='utf-8') as in_f:
        
        with httpx.Client(timeout=120.0) as client:
            # Dùng enumerate để biết index hiện tại
            for i, line in enumerate(tqdm(in_f, total=total_lines, desc="Dịch")):
                # Skip những câu đã dịch ở lần chạy trước
                if i < processed_count:
                    continue
                
                line = line.strip()
                if not line:
                    continue

                new_item = {}
                try:
                    item = json.loads(line)
                    original_text = item.get("text", "")
                    
                    payload = {
                        "text": original_text,
                        "target_language_code": TARGET_LANG_CODE,
                        "temperature": 0.2
                    }
                    
                    response = client.post(API_URL, json=payload)
                    response.raise_for_status()
                    
                    result_data = response.json()
                    translated_text = result_data.get("translated_text", "")
                    
                    new_item["Bản gốc"] = original_text
                    new_item["Bản dịch"] = translated_text
                    
                except Exception as e:
                    print(f"\nLỗi tại dòng {i+1}: {e}")
                    new_item["Bản gốc"] = original_text if 'original_text' in locals() else "Unknown"
                    new_item["Bản dịch"] = None
                    new_item["error"] = str(e)

                # Ghi và flush ngay lập tức
                out_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                out_f.flush() 

    print(f"\n--- Hoàn thành! File lưu tại: {OUTPUT_FILE} ---")

if __name__ == "__main__":
    run_translation()