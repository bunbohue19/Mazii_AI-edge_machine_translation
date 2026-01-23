# Send a translation request

# For not TranslateGemma
# curl -X POST "http://localhost:8888/v1/translate" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "text": "11時29分頃、デモ隊は英国政府に向かい、トラファルガー広場を通り過ぎて、ストランド街沿いにアルドウィックのそばを通り抜け、キングスウェイをホルボーンに向かって進みましたが、そこでは保守党がグランドコンノートルームズホテルで春季フォーラムを開催していました。",
#     "temperature": 0.20,
#     "target_lang_code": "Vietnamese"
#   }'

# TranslateGemma
# curl -X POST "http://localhost:8888/v1/translate" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "text": "800 dặm của Hệ thống đường ống xuyên Alaska đã bị đóng sau sự cố tràn hàng ngàn thùng dầu thô ở phía nam Fairbanks Alaska.",
#     "temperature": 0.20,
#     "source_lang_code": "vi",
#     "target_lang_code": "ja"
#   }'

# python 1_infer.py
python 2_convert.py