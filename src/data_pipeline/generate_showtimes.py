import os
import random
from datetime import datetime, timedelta
import pandas as pd

def generate_mock_showtimes():
    movies_path = "data/processed/movies.csv"
    if not os.path.exists(movies_path):
        print(f"Lỗi: Không tìm thấy file {movies_path}")
        return

    movies_df = pd.read_csv(movies_path)
    # Lấy 35 phim đầu tiên làm danh mục phim đang có suất chiếu tại rạp
    active_movies = movies_df.head(35)[["movie_id", "title"]].to_dict("records")

    cinemas = [
        "CGV Vincom Bà Triệu",
        "CGV Aeon Mall Long Biên",
        "CGV Indochina Plaza Hà Nội (IPH)",
        "CGV Mipec Tower",
        "CGV Vincom Center Landmark 81",
        "CGV Crescent Mall"
    ]

    formats = ["2D Phụ đề", "2D Lồng tiếng", "IMAX 2D"]
    time_slots = ["09:15", "11:30", "13:45", "15:20", "17:40", "19:15", "20:30", "21:45", "23:00"]

    records = []
    today = datetime.now()

    # Sinh lịch chiếu cho 3 ngày liên tiếp (hôm nay, ngày mai, ngày kia)
    for day_offset in range(3):
        current_date = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        
        for cinema in cinemas:
            # Mỗi rạp chọn ngẫu nhiên 8-14 phim chiếu trong ngày
            daily_movies = random.sample(active_movies, k=random.randint(8, 14))
            
            for m in daily_movies:
                # Mỗi phim có từ 2 đến 4 suất chiếu tại rạp đó
                slots = random.sample(time_slots, k=random.randint(2, 4))
                slots.sort()
                
                for slot in slots:
                    fmt = random.choices(formats, weights=[0.6, 0.25, 0.15])[0]
                    # Giá vé: IMAX giá cao, buổi tối sau 17h giá cao hơn ban ngày
                    if fmt == "IMAX 2D":
                        price = 160000
                    elif slot >= "17:00":
                        price = 110000
                    else:
                        price = 85000

                    records.append({
                        "movie_id": m["movie_id"],
                        "title": m["title"],
                        "cinema": cinema,
                        "date": current_date,
                        "time": slot,
                        "format": fmt,
                        "price": price
                    })

    df_showtimes = pd.DataFrame(records)
    output_path = "data/processed/showtimes.csv"
    df_showtimes.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"-> Đã tạo thành công {len(df_showtimes)} suất chiếu giả lập tại {output_path}!")

if __name__ == "__main__":
    generate_mock_showtimes()