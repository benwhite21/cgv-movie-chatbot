import os
import re
import pandas as pd

class EntityExtractor:
    def __init__(self, movies_path="data/processed/movies.csv", showtimes_path="data/processed/showtimes.csv"):
        # 1. Tải danh sách tên phim từ dữ liệu thật
        self.movie_titles = []
        if os.path.exists(movies_path):
            df_m = pd.read_csv(movies_path)
            titles = df_m["title"].dropna().unique().tolist()
            # Sắp xếp tên phim dài trước để ưu tiên khớp cụm từ đầy đủ
            self.movie_titles = sorted(titles, key=len, reverse=True)

        # 2. Từ điển thể loại phim
        self.genres = [
            "hành động", "kinh dị", "hài hước", "hài", "tình cảm", "lãng mạn", 
            "hoạt hình", "viễn tưởng", "tâm lý", "phiêu lưu", "chiến tranh",
            "gia đình", "tội phạm", "bí ẩn", "âm nhạc"
        ]

        # 3. Từ điển rạp và bí danh (alias)
        self.cinema_aliases = {
            "bà triệu": "CGV Vincom Bà Triệu",
            "vincom bà triệu": "CGV Vincom Bà Triệu",
            "long biên": "CGV Aeon Mall Long Biên",
            "aeon long biên": "CGV Aeon Mall Long Biên",
            "iph": "CGV Indochina Plaza Hà Nội (IPH)",
            "indochina": "CGV Indochina Plaza Hà Nội (IPH)",
            "mipec": "CGV Mipec Tower",
            "landmark 81": "CGV Vincom Center Landmark 81",
            "landmark": "CGV Vincom Center Landmark 81",
            "crescent mall": "CGV Crescent Mall"
        }

    def extract(self, text: str) -> dict:
        text_lower = text.lower()
        extracted = {
            "genre": None,
            "movie": None,
            "cinema": None
        }

        # 1. Bóc tách thể loại trước
        for g in self.genres:
            if g in text_lower:
                extracted["genre"] = g
                break

        # 2. Bóc tách tên rạp
        for alias, full_name in self.cinema_aliases.items():
            if alias in text_lower:
                extracted["cinema"] = full_name
                break

        # 3. Bóc tách tên phim: Khớp trực tiếp với danh sách kho phim đã cào
        for m in self.movie_titles:
            m_clean = m.strip().lower()
            if len(m_clean) >= 2 and m_clean in text_lower:
                extracted["movie"] = m
                break

        # 4. Trợ lực Regex: CHỈ tìm khi CHƯA có tên phim VÀ CHƯA có thể loại
        # (Để tránh bắt nhầm 'phim hành động', 'phim hoạt hình' thành tên phim)
        if not extracted["movie"] and not extracted["genre"]:
            pattern = r"(?:phim|suất chiếu)\s+([a-zA-Z0-9\sÀ-ỹ]+?)(?:\s+(?:ở|tại|hôm nay|ngày mai|tối nay|chiều nay|lúc)|\?|$)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 1:
                    extracted["movie"] = candidate

        return extracted

if __name__ == "__main__":
    extractor = EntityExtractor()
    test_queries = [
        "Có phim hành động nào hay không bạn?",
        "Lịch chiếu phim Mai ở rạp CGV Bà Triệu tối nay thế nào?",
        "Giá vé ở rạp Landmark 81 bao nhiêu tiền?",
        "Tư vấn phim hoạt hình cho trẻ em"
    ]
    print("--- KIỂM TRA THỬ BÓC TÁCH THỰC THỂ (NER ĐÃ CẬP NHẬT) ---")
    for q in test_queries:
        print(f"Câu: '{q}'")
        print(f"-> Thực thể bóc tách được: {extractor.extract(q)}\n")