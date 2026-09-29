import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataset():
    # Nạp danh sách phim và rạp có sẵn để câu mẫu chân thực
    movies_df = pd.read_csv("data/processed/movies.csv")
    sample_movies = movies_df["title"].dropna().tolist()[:25]
    
    genres = [
        "hành động", "kinh dị", "hài hước", "tình cảm", 
        "hoạt hình", "viễn tưởng", "tâm lý", "phiêu lưu", "chiếu rạp"
    ]
    cinemas = [
        "CGV Bà Triệu", "CGV Long Biên", "CGV Landmark 81", 
        "CGV Crescent Mall", "CGV IPH", "CGV Mipec"
    ]

    dataset = []

    # 1. Intent: chao_hoi
    greetings = [
        "Xin chào", "Hello bot", "Hi bạn", "Chào buổi tối", "Chào CGV",
        "Có ai ở đây không", "Tư vấn giúp mình với", "Hello", "Chào ad nhé",
        "Bot ơi giúp mình với", "Chào bạn, mình cần hỗ trợ", "Alo bot"
    ]
    for text in greetings * 10:
        dataset.append({"text": text, "intent": "chao_hoi", "entities": []})

    # 2. Intent: tim_phim
    for g in genres:
        templates = [
            f"Gợi ý cho mình vài phim {g} với",
            f"Có phim {g} nào đang chiếu rạp hay không?",
            f"Tối nay muốn xem phim thể loại {g}, giới thiệu giùm mình",
            f"Tư vấn phim {g} hot nhất hiện nay",
            f"Mình thích xem {g}, có bộ nào điểm cao không?",
            f"Đang chiếu những phim {g} nào vậy bot?",
            f"Hôm nay rạp có phim {g} nào đáng xem không?"
        ]
        for t in templates * 3:
            start = t.find(g)
            dataset.append({
                "text": t,
                "intent": "tim_phim",
                "entities": [{"entity": "genre", "value": g, "start": start, "end": start + len(g)}]
            })

    # Thêm câu hỏi tìm phim chung chung (không có genre)
    general_movie_queries = [
        "Rạp đang chiếu những phim gì hot nhất?",
        "Tối nay có phim gì hay đáng xem không?",
        "Gợi ý cho mình top phim ăn khách hiện tại",
        "Cuối tuần này nên đi xem phim gì vậy bot?",
        "Cho mình danh sách các phim đang chiếu rạp"
    ]
    for text in general_movie_queries * 6:
        dataset.append({"text": text, "intent": "tim_phim", "entities": []})

    # 3. Intent: hoi_lich_chieu
    for m in sample_movies:
        c = random.choice(cinemas)
        templates = [
            f"Lịch chiếu phim {m} ở {c} hôm nay thế nào?",
            f"{m} có suất chiếu nào ở {c} tối nay không?",
            f"Cho mình xem lịch chiếu {m} tại rạp {c}",
            f"Mấy giờ có suất chiếu phim {m} ở {c} vậy?",
            f"Tra cứu suất chiếu của {m} tại {c}"
        ]
        for t in templates:
            entities = []
            sm = t.find(m)
            if sm != -1:
                entities.append({"entity": "movie", "value": m, "start": sm, "end": sm + len(m)})
            sc = t.find(c)
            if sc != -1:
                entities.append({"entity": "cinema", "value": c, "start": sc, "end": sc + len(c)})
            dataset.append({"text": t, "intent": "hoi_lich_chieu", "entities": entities})

    # 4. Intent: hoi_gia_ve
    for c in cinemas:
        templates = [
            f"Giá vé xem phim ở {c} là bao nhiêu?",
            f"Vé rạp {c} cuối tuần giá thế nào bạn?",
            f"Cho mình hỏi bảng giá vé sinh viên tại {c}",
            f"Xem phim ở {c} hết bao nhiêu tiền một vé?"
        ]
        for t in templates * 4:
            dataset.append({"text": t, "intent": "hoi_gia_ve", "entities": []})

    general_pricing = [
        "Giá vé xem phim 2D bao nhiêu tiền vậy?",
        "Xem phòng chiếu IMAX giá vé bao nhiêu một người?",
        "Bảng giá vé CGV hiện tại",
        "Vé buổi tối sau 18h giá bao nhiêu?",
        "Giá vé trẻ em và học sinh sinh viên thế nào?"
    ]
    for text in general_pricing * 8:
        dataset.append({"text": text, "intent": "hoi_gia_ve", "entities": []})

    # 5. Intent: dat_ve
    book_texts = [
        "Tôi muốn đặt 2 vé xem phim",
        "Hướng dẫn mình cách đặt vé online",
        "Làm sao để book vé xem phim vậy bot?",
        "Đặt vé suất 19h tối nay giúp mình",
        "Mình muốn giữ chỗ 2 ghế hàng VIP",
        "Mua vé xem phim qua chatbot được không?",
        "Cho mình link đặt vé với",
        "Cách mua vé trên app CGV như thế nào?"
    ]
    for text in book_texts * 10:
        dataset.append({"text": text, "intent": "dat_ve", "entities": []})

    # 6. Intent: khong_hieu (Fallback)
    fallbacks = [
        "Hôm nay trời đẹp quá", "Rạp có bán trà sữa trân châu không?",
        "Bạn bao nhiêu tuổi rồi", "Cổ phiếu hôm nay tăng hay giảm?",
        "Viết cho tôi một bài thơ tình", "Chỉ đường đi đến bờ hồ",
        "Thủ đô của nước Pháp là gì?", "Bạn có người yêu chưa?",
        "Dự báo thời tiết ngày mai ra sao", "Một cộng một bằng mấy?"
    ]
    for text in fallbacks * 8:
        dataset.append({"text": text, "intent": "khong_hieu", "entities": []})

    # Trộn ngẫu nhiên dữ liệu
    random.seed(42)
    random.shuffle(dataset)

    # Lưu dialogues.json tổng hợp
    os.makedirs("data/dialogues", exist_ok=True)
    with open("data/dialogues/dialogues.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Chia tập theo tỷ lệ 80 - 10 - 10 cân bằng theo nhãn intent
    intents = [item["intent"] for item in dataset]
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42, stratify=intents)
    
    temp_intents = [item["intent"] for item in temp_data]
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_intents)

    with open("data/dialogues/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open("data/dialogues/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open("data/dialogues/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"-> Tổng số câu hội thoại: {len(dataset)}")
    print(f"-> Train: {len(train_data)} câu (80%)")
    print(f"-> Validation: {len(val_data)} câu (10%)")
    print(f"-> Test: {len(test_data)} câu (10%)")

if __name__ == "__main__":
    create_dataset()