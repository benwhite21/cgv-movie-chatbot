import os
import sys

# Đảm bảo đường dẫn import không bị lỗi
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize

class MovieRecommender:
    def __init__(self, movies_path="data/processed/movies.csv", showtimes_path="data/processed/showtimes.csv"):
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Không tìm thấy file: {movies_path}")

        self.df = pd.read_csv(movies_path)
        self.df["overview"] = self.df["overview"].fillna("")
        self.df["genres"] = self.df["genres"].fillna("")
        self.df["director"] = self.df["director"].fillna("")
        self.df["cast"] = self.df["cast"].fillna("")
        self.df["rating"] = self.df["rating"].fillna(0.0)

        # Lấy danh sách ID các phim đang có suất chiếu tại rạp
        self.showing_movie_ids = set()
        if os.path.exists(showtimes_path):
            df_st = pd.read_csv(showtimes_path)
            self.showing_movie_ids = set(df_st["movie_id"].dropna().unique().tolist())

        # Tạo Feature Soup và tách từ tiếng Việt
        self.df["soup"] = (
            self.df["title"] + " " +
            (self.df["genres"] + " ") * 2 +  # Tăng trọng số thể loại
            self.df["director"] + " " +
            self.df["cast"] + " " +
            self.df["overview"]
        ).apply(lambda x: word_tokenize(x.lower(), format="text"))

        # Vector hóa đặc trưng TF-IDF
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["soup"])

    def recommend(self, query: str, genre_filter: str = None, top_k: int = 3, only_now_showing: bool = False) -> list:
        """
        Gợi ý Top-K phim dựa trên văn bản truy vấn và bộ lọc thể loại/đang chiếu
        """
        tokenized_query = word_tokenize(query.lower(), format="text")
        query_vec = self.vectorizer.transform([tokenized_query])

        # 1. Tính độ tương đồng Cosine giữa query và toàn bộ phim
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 2. Chuẩn hóa điểm rating về thang [0, 1]
        max_rating = self.df["rating"].max() or 10.0
        normalized_ratings = (self.df["rating"] / max_rating).to_numpy()

        # 3. Tính điểm xếp hạng kết hợp:
        # Nếu phim đang có suất chiếu tại rạp, cộng thêm điểm ưu tiên (bonus +0.25)
        showing_bonus = np.array([0.25 if m_id in self.showing_movie_ids else 0.0 for m_id in self.df["movie_id"]])
        final_scores = (0.65 * sim_scores) + (0.20 * normalized_ratings) + (0.15 * showing_bonus)
        # 4. Sắp xếp thứ tự giảm dần
        ranked_indices = final_scores.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            row = self.df.iloc[idx]
            movie_id = int(row["movie_id"])

            # Áp dụng bộ lọc "chỉ phim đang chiếu" nếu có yêu cầu
            if only_now_showing and self.showing_movie_ids and (movie_id not in self.showing_movie_ids):
                continue

            # Áp dụng bộ lọc thể loại nếu người dùng chỉ định rõ
            if genre_filter and (genre_filter.lower() not in str(row["genres"]).lower()):
                continue

            # Đảm bảo đường link poster hợp lệ
            poster = ""
            if pd.notnull(row["poster_path"]) and str(row["poster_path"]).strip():
                poster = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"

            overview_short = str(row["overview"])
            if len(overview_short) > 130:
                overview_short = overview_short[:130] + "..."

            results.append({
                "movie_id": movie_id,
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "rating": float(row["rating"]),
                "runtime": int(row["runtime"]) if pd.notnull(row["runtime"]) else 0,
                "director": str(row["director"]),
                "cast": str(row["cast"]),
                "poster_url": poster,
                "overview": overview_short,
                "similarity_score": round(float(sim_scores[idx]), 4),
                "is_showing": movie_id in self.showing_movie_ids
            })

            if len(results) >= top_k:
                break

        return results

    def get_top_rated(self, top_k: int = 3, only_now_showing: bool = True) -> list:
        """Lấy danh sách các phim hot có điểm đánh giá cao nhất"""
        filtered_df = self.df.copy()
        if only_now_showing and self.showing_movie_ids:
            filtered_df = filtered_df[filtered_df["movie_id"].isin(self.showing_movie_ids)]

        top_df = filtered_df.sort_values(by="rating", ascending=False).head(top_k)
        results = []
        for _, row in top_df.iterrows():
            poster = f"https://image.tmdb.org/t/p/w500{row['poster_path']}" if pd.notnull(row["poster_path"]) else ""
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": str(row["title"]),
                "genres": str(row["genres"]),
                "rating": float(row["rating"]),
                "poster_url": poster,
                "overview": str(row["overview"])[:130] + "...",
                "is_showing": True
            })
        return results

if __name__ == "__main__":
    recommender = MovieRecommender()
    print("--- KIỂM TRA THỬ HỆ THỐNG GỢI Ý PHIM (RECOMMENDER) ---")
    
    test_cases = [
        {"query": "Tôi muốn tìm phim hoạt hình vui nhộn hài hước cho gia đình", "genre": "hoạt hình"},
        {"query": "Phim kinh dị rùng rợn giật gân", "genre": "kinh dị"},
        {"query": "Phim khoa học viễn tưởng kỹ xảo hoành tráng", "genre": "viễn tưởng"}
    ]

    for tc in test_cases:
        print(f"\n[Yêu cầu]: '{tc['query']}' (Bộ lọc: {tc['genre']})")
        recs = recommender.recommend(tc['query'], genre_filter=tc['genre'], top_k=2)
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']} | Điểm: ★ {r['rating']} | Thể loại: {r['genres']}")
            print(f"     Tương đồng: {r['similarity_score']} | Đang chiếu: {r['is_showing']}")