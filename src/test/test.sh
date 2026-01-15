curl -X POST "http://localhost:8888/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Người được ủy quyền nộp tiền bảo lãnh, nếu được chấp thuận, và hợp thức hóa các khoản phí được cảnh sát thực hiện bắt giữ đệ trình lên. Các khoản phí này sau đó được nhập vào hệ thống máy tính của bang nơi vụ án được theo dõi.",
    "thinking_budget": 128,
    "temperature": 0.20,
    "target_language_code": "Japanese"
  }'