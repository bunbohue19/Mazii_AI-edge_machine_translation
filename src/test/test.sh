curl -X POST "http://localhost:8888/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "11時29分頃、デモ隊は英国政府に向かい、トラファルガー広場を通り過ぎて、ストランド街沿いにアルドウィックのそばを通り抜け、キングスウェイをホルボーンに向かって進みましたが、そこでは保守党がグランドコンノートルームズホテルで春季フォーラムを開催していました。",
    "temperature": 0.20,
    "target_language_code": "Vietnamese"
  }'