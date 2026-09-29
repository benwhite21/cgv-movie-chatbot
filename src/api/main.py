import os
import sys

# Đảm bảo đường dẫn thư mục gốc dự án
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from src.dialogue_manager.flow import DialogueManager
from src.recommender.engine import MovieRecommender

app = FastAPI(
    title="CGV Movie Chatbot API",
    description="Hệ thống API tư vấn chọn phim và tra cứu lịch chiếu cụm rạp CGV",
    version="1.0.0"
)

# Cấu hình CORS để Frontend (HTML/JS/React) gọi API từ bất kỳ cổng nào mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nạp các mô hình lõi và dữ liệu vào bộ nhớ khi khởi động server
print("Đang khởi tạo hệ thống NLU, Recommender và Database...")
dm = DialogueManager()
recommender = MovieRecommender()

showtimes_path = "data/processed/showtimes.csv"
if os.path.exists(showtimes_path):
    showtimes_df = pd.read_csv(showtimes_path)
else:
    showtimes_df = pd.DataFrame()

# Định nghĩa cấu trúc dữ liệu Request & Response chuẩn Pydantic
class ChatRequest(BaseModel):
    message: str

class MovieCard(BaseModel):
    movie_id: int
    title: str
    genres: str
    rating: float
    poster_url: str
    overview: str
    runtime: Optional[int] = 0
    director: Optional[str] = ""
    cast: Optional[str] = ""

class ShowtimeItem(BaseModel):
    title: str
    cinema: str
    date: str
    time: str
    format: str
    price: int

class ChatResponse(BaseModel):
    reply_text: str
    intent: str
    data_type: str  # "text", "movie_carousel", "showtimes_list"
    movies: List[MovieCard] = []
    showtimes: List[ShowtimeItem] = []
    quick_replies: List[str] = []

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "CGV Chatbot API Backend",
        "docs_url": "/docs"
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_handler(req: ChatRequest):
    user_msg = req.message.strip()
    if not user_msg:
        return ChatResponse(
            reply_text="Bạn hãy nhập câu hỏi để em hỗ trợ nhé!",
            intent="chao_hoi",
            data_type="text",
            quick_replies=["Phim hot đang chiếu", "Gợi ý phim hành động", "Bảng giá vé"]
        )

    # 1. Phân tích NLU & Điều hướng hội thoại
    parsed = dm.parse(user_msg)
    action_info = dm.determine_action(parsed)
    intent = parsed["intent"]
    entities = parsed["entities"]

    # 2. Xử lý theo từng hành động
    if action_info["action"] == "greet":
        # Chào hỏi: Gợi ý kèm Top 3 phim có điểm đánh giá cao nhất đang chiếu
        top_movies = recommender.get_top_rated(top_k=3, only_now_showing=True)
        return ChatResponse(
            reply_text=action_info["message"],
            intent=intent,
            data_type="movie_carousel",
            movies=[MovieCard(**m) for m in top_movies],
            quick_replies=["Phim kinh dị", "Phim hoạt hình", "Giá vé CGV", "Lịch chiếu hôm nay"]
        )

    elif action_info["action"] == "recommend_movie":
        # Tìm phim: Ưu tiên lọc theo thể loại đã bóc tách được từ NER
        genre = entities.get("genre")
        recs = recommender.recommend(user_msg, genre_filter=genre, top_k=3)
        
        if genre:
            reply = f"Dưới đây là một số phim thể loại {genre} nổi bật dành cho bạn:"
        else:
            reply = "Em gợi ý cho bạn một số phim đang được quan tâm nhất hiện nay:"

        return ChatResponse(
            reply_text=reply,
            intent=intent,
            data_type="movie_carousel",
            movies=[MovieCard(**m) for m in recs],
            quick_replies=["Xem lịch chiếu", "Tìm phim khác", "Bảng giá vé"]
        )

    elif action_info["action"] == "lookup_showtimes":
        # Tra cứu lịch chiếu từ showtimes.csv
        movie = entities.get("movie")
        cinema = entities.get("cinema")

        filtered = showtimes_df.copy()
        if movie and not filtered.empty:
            filtered = filtered[filtered["title"].str.lower().str.contains(movie.lower())]
        if cinema and not filtered.empty:
            filtered = filtered[filtered["cinema"].str.lower().str.contains(cinema.lower())]

        if not filtered.empty:
            # Lấy tối đa 5 suất chiếu gần nhất
            matched_records = filtered.head(5).to_dict("records")
            info_text = f"Suất chiếu cho"
            if movie:
                info_text += f" phim '{movie}'"
            if cinema:
                info_text += f" tại {cinema}"
            info_text += ":"

            return ChatResponse(
                reply_text=info_text,
                intent=intent,
                data_type="showtimes_list",
                showtimes=[ShowtimeItem(**s) for s in matched_records],
                quick_replies=["Đặt vé ngay", "Tìm phim khác", "Giá vé IMAX"]
            )
        else:
            return ChatResponse(
                reply_text=f"Rất tiếc hiện chưa có suất chiếu phù hợp cho yêu cầu này. Bạn hãy thử chọn ngày khác hoặc cụm rạp CGV khác nhé!",
                intent=intent,
                data_type="text",
                quick_replies=["Lịch chiếu hôm nay", "Phim hot đang chiếu"]
            )

    elif action_info["action"] == "ask_slot":
        # Thiếu thông tin slot phim/rạp
        return ChatResponse(
            reply_text=action_info["message"],
            intent=intent,
            data_type="text",
            quick_replies=["CGV Bà Triệu", "CGV Long Biên", "CGV Landmark 81"]
        )

    elif action_info["action"] == "show_pricing":
        pricing_text = (
            "🎫 BẢNG GIÁ VÉ TIÊU CHUẨN CGV:\n\n"
            "• Vé 2D (Thứ 2 - Thứ 5 trước 17h): 85.000 VNĐ\n"
            "• Vé 2D (Buổi tối sau 17h & Cuối tuần): 110.000 VNĐ\n"
            "• Phòng chiếu cao cấp IMAX 2D: 160.000 VNĐ\n"
            "• Học sinh, sinh viên (U22 có thẻ): Đồng giá 75.000 VNĐ (áp dụng 2D ngày thường)."
        )
        return ChatResponse(
            reply_text=pricing_text,
            intent=intent,
            data_type="text",
            quick_replies=["Xem phim đang chiếu", "Lịch chiếu hôm nay"]
        )

    elif action_info["action"] == "guide_booking":
        return ChatResponse(
            reply_text=action_info["message"],
            intent=intent,
            data_type="text",
            quick_replies=["Xem phim hot", "Giá vé 2D"]
        )

    else:
        # Fallback
        return ChatResponse(
            reply_text=action_info["message"],
            intent=intent,
            data_type="text",
            quick_replies=["Phim hot đang chiếu", "Gợi ý phim hài", "Bảng giá vé"]
        )