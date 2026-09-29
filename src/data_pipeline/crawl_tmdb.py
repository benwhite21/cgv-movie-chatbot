import os
import requests
import pandas as pd
import time

# Dán API key (v3 auth) của bạn vào giữa 2 dấu ngoặc kép
API_KEY = "8584125935dd5dfdfd093be5d567d7d8"
BASE_URL = "https://api.themoviedb.org/3"

def get_genre_mapping():
    """Lấy danh mục thể loại dịch sang tiếng Việt"""
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=vi-VN"
    try:
        res = requests.get(url, timeout=10).json()
        return {g['id']: g['name'] for g in res.get('genres', [])}
    except Exception as e:
        print(f"Lỗi lấy bảng thể loại: {e}")
        return {}

def fetch_movies(total_pages=5):
    genre_map = get_genre_mapping()
    movies_list = []
    seen_ids = set()

    # Thu thập từ 4 endpoint phổ biến để gom đủ 150 - 300 phim
    endpoints = ["now_playing", "popular", "upcoming", "top_rated"]

    print("Đang tiến hành thu thập dữ liệu phim từ TMDB...")
    for ep in endpoints:
        for page in range(1, total_pages + 1):
            url = f"{BASE_URL}/movie/{ep}?api_key={API_KEY}&language=vi-VN&page={page}"
            try:
                data = requests.get(url, timeout=10).json()
            except Exception as e:
                print(f"Lỗi trang {page} mục {ep}: {e}")
                continue

            results = data.get("results", [])
            for m in results:
                m_id = m.get("id")
                if not m_id or m_id in seen_ids:
                    continue
                seen_ids.add(m_id)

                # Gọi chi tiết để lấy credits (đạo diễn, diễn viên) và runtime
                detail_url = f"{BASE_URL}/movie/{m_id}?api_key={API_KEY}&language=vi-VN&append_to_response=credits"
                try:
                    det = requests.get(detail_url, timeout=10).json()
                except Exception:
                    continue

                # Lấy đạo diễn và top 3 diễn viên chính
                credits = det.get("credits", {})
                directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
                cast = [c["name"] for c in credits.get("cast", [])[:3]]

                # Lấy tóm tắt nội dung (overview)
                overview = det.get("overview", "").strip()
                if not overview:
                    overview = det.get("title", "")

                # Thể loại (genres)
                genres = [g["name"] for g in det.get("genres", [])]
                if not genres:
                    genres = [genre_map.get(gid, "") for gid in m.get("genre_ids", []) if gid in genre_map]
                genres = [g for g in genres if g]

                if not genres:
                    continue

                movies_list.append({
                    "movie_id": m_id,
                    "title": det.get("title") or m.get("title"),
                    "overview": overview,
                    "genres": ", ".join(genres),
                    "release_date": det.get("release_date", ""),
                    "rating": det.get("vote_average", 0.0),
                    "runtime": det.get("runtime", 0),
                    "director": ", ".join(directors),
                    "cast": ", ".join(cast),
                    "poster_path": det.get("poster_path", "")
                })

                # Nghỉ nhẹ giữa các request để tránh rate limit
                time.sleep(0.04)

    df = pd.DataFrame(movies_list)
    output_path = "data/processed/movies.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"-> Thu thập hoàn tất: {len(df)} phim đã lưu vào {output_path}!")

if __name__ == "__main__":
    fetch_movies(total_pages=5)