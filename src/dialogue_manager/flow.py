import os
import pickle
from underthesea import word_tokenize
from src.nlu.extractor import EntityExtractor

class DialogueManager:
    def __init__(self, model_path="src/nlu/intent_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.extractor = EntityExtractor()

    def parse(self, text: str) -> dict:
        """Nhận diện Intent và trích xuất Entity"""
        tokenized_text = word_tokenize(text.lower(), format="text")
        intent = self.model.predict([tokenized_text])[0]
        entities = self.extractor.extract(text)
        return {
            "text": text,
            "intent": intent,
            "entities": entities
        }

    def determine_action(self, parsed: dict) -> dict:
        """Xác định hành động tiếp theo và áp dụng quy tắc Slot-filling / Fallback"""
        intent = parsed["intent"]
        entities = parsed["entities"]

        # 1. Fallback nếu rơi vào khong_hieu
        if intent == "khong_hieu":
            return {
                "action": "fallback",
                "message": "Xin lỗi, em chưa hiểu rõ yêu cầu. Bạn có thể chọn tìm phim theo thể loại, tra cứu lịch chiếu hoặc xem bảng giá vé nhé!",
                "data": None
            }

        # 2. Xử lý hỏi lịch chiếu (Slot-filling)
        if intent == "hoi_lich_chieu":
            if not entities["movie"] and not entities["cinema"]:
                return {
                    "action": "ask_slot",
                    "missing_slot": "movie_or_cinema",
                    "message": "Bạn muốn tra cứu lịch chiếu của phim nào hoặc tại cụm rạp nào ạ?",
                    "data": None
                }
            return {
                "action": "lookup_showtimes",
                "message": None,
                "data": entities
            }

        # 3. Xử lý tìm phim
        if intent == "tim_phim":
            return {
                "action": "recommend_movie",
                "message": None,
                "data": entities
            }

        # 4. Các intent thông tin tĩnh
        if intent == "chao_hoi":
            return {
                "action": "greet",
                "message": "Xin chào! Em là trợ lý CGV Cinemas. Em có thể giúp gì cho bạn hôm nay?",
                "data": None
            }

        if intent == "hoi_gia_ve":
            return {
                "action": "show_pricing",
                "message": None,
                "data": entities
            }

        if intent == "dat_ve":
            return {
                "action": "guide_booking",
                "message": "Để đặt vé giữ chỗ, bạn có thể thao tác trực tiếp qua app CGV Cinemas hoặc website cgv.vn nhé!",
                "data": None
            }

        return {"action": "fallback", "message": "Em chưa xử lý được yêu cầu này.", "data": None}

if __name__ == "__main__":
    dm = DialogueManager()
    tests = [
        "Xin chào bot",
        "Có phim hoạt hình nào đang chiếu không?",
        "Cho mình xem lịch chiếu phim Mai ở CGV Bà Triệu",
        "Lịch chiếu hôm nay thế nào?",
        "Giá vé phòng IMAX bao nhiêu?",
        "Hôm nay trời nhiều mây quá"
    ]
    print("--- KIỂM TRA ĐIỀU HƯỚNG HỘI THOẠI (DIALOGUE MANAGER) ---")
    for t in tests:
        res = dm.parse(t)
        action = dm.determine_action(res)
        print(f"Câu hỏi: {t}")
        print(f"-> Intent: {res['intent']} | Action: {action['action']}")
        if action['message']:
            print(f"   Phản hồi: {action['message']}")
        print()