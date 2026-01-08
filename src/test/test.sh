curl -X POST "http://localhost:8888/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "まずはイタリアの計画について説明しましょう。イタリアは主にドイツと日本の「弟分」でした。",
    "target_language_code": "Vietnamese",
    "temperature": 0.20
  }'