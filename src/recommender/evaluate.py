import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from src.recommender.engine import MovieRecommender

def evaluate_recommendations():
    recommender = MovieRecommender()

    # Bộ ca kiểm thử đại diện cho các nhu cầu tìm phim đa dạng
    test_queries = [
        {"q": "Tôi muốn tìm phim hành động đánh nhau gay cấn", "target_genre": "hành động"},
        {"q": "Có phim hoạt hình vui nhộn cho gia đình không", "target_genre": "hoạt hình"},
        {"q": "Giới thiệu phim kinh dị ma quỷ rùng rợn", "target_genre": "kinh dị"},
        {"q": "Tìm phim tình cảm lãng mạn lứa đôi ngọt ngào", "target_genre": "lãng mạn"},
        {"q": "Phim viễn tưởng vũ trụ công nghệ tương lai", "target_genre": "khoa học viễn tưởng"},
        {"q": "Có phim hài hước xem giải trí cuối tuần không", "target_genre": "hài"},
        {"q": "Phim tài liệu chiến tranh lịch sử", "target_genre": "chiến tranh"},
        {"q": "Gợi ý phim phiêu lưu khám phá mạo hiểm", "target_genre": "phiêu lưu"},
        {"q": "Phim tội phạm điều tra phá án ly kỳ", "target_genre": "tội phạm"},
        {"q": "Phim chính kịch tâm lý xã hội sâu sắc", "target_genre": "chính kịch"}
    ]

    K = 3
    precision_list = []
    
    print("=" * 60)
    print(f"ĐÁNH GIÁ CHỈ SỐ HỆ THỐNG GỢI Ý (TOP-{K}) - GIAI ĐOẠN 3")
    print("=" * 60)

    for tc in test_queries:
        recs = recommender.recommend(tc["q"], top_k=K)
        # Đếm số phim trong Top-K có chứa đúng thể loại mục tiêu
        hits = 0
        for r in recs:
            if tc["target_genre"].lower() in str(r["genres"]).lower():
                hits += 1

        prec = hits / K
        precision_list.append(prec)
        print(f"Query: '{tc['q'][:35]}...'")
        print(f"-> Thể loại kỳ vọng: {tc['target_genre']} | Số phim khớp: {hits}/{K} (Precision@{K}: {prec:.2f})\n")

    mean_precision = np.mean(precision_list)
    print("=" * 60)
    print(f"KẾT QUẢ TỔNG HỢP: Mean Precision@{K} = {mean_precision * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_recommendations()