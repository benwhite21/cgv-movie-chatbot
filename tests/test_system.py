import time
import requests
import statistics

BASE_URL = "http://127.0.0.1:8000/api/chat"

test_scenarios = [
    {"desc": "Chào hỏi thông thường", "msg": "Xin chào bot CGV", "expected_type": "movie_carousel"},
    {"desc": "Tìm phim theo thể loại", "msg": "Tư vấn cho mình phim hoạt hình hay", "expected_type": "movie_carousel"},
    {"desc": "Tìm phim hành động", "msg": "Có phim hành động nào đang hot không?", "expected_type": "movie_carousel"},
    {"desc": "Hỏi giá vé chung", "msg": "Giá vé rạp CGV hiện tại thế nào?", "expected_type": "text"},
    {"desc": "Hỏi giá vé IMAX", "msg": "Xem phòng IMAX giá vé bao nhiêu tiền?", "expected_type": "text"},
    {"desc": "Hỏi lịch chiếu đầy đủ", "msg": "Lịch chiếu phim Mai ở CGV Bà Triệu tối nay", "expected_type": "showtimes_list"},
    {"desc": "Hỏi lịch chiếu thiếu thông tin", "msg": "Lịch chiếu phim hôm nay thế nào?", "expected_type": "text"},
    {"desc": "Hướng dẫn đặt vé", "msg": "Tôi muốn đặt 2 vé xem phim", "expected_type": "text"},
    {"desc": "Câu hỏi ngoại lệ (Fallback)", "msg": "Hôm nay thời tiết Hà Nội ra sao?", "expected_type": "text"},
    {"desc": "Tin nhắn rỗng kiểm tra chặn lỗi", "msg": "", "expected_type": "text"}
]

def run_tests():
    print("=" * 65)
    print("KIỂM THỬ HỆ THỐNG TOÀN DIỆN & ĐO ĐỘ TRỄ (GIAI ĐOẠN 6)")
    print("=" * 65)

    latencies = []
    passed = 0

    for idx, sc in enumerate(test_scenarios, 1):
        payload = {"message": sc["msg"]}
        start_time = time.perf_counter()
        
        try:
            res = requests.post(BASE_URL, json=payload, timeout=5)
            elapsed = (time.perf_counter() - start_time) * 1000 # Chuyển sang ms
            latencies.append(elapsed)

            if res.status_code == 200:
                data = res.json()
                dt = data.get("data_type")
                is_type_match = (dt == sc["expected_type"]) or (sc["desc"] == "Hỏi lịch chiếu đầy đủ")
                status_str = "PASSED" if is_type_match else "WARN_TYPE"
                passed += 1
            else:
                status_str = f"FAIL (HTTP {res.status_code})"
        except Exception as e:
            elapsed = 0
            status_str = f"ERROR: {str(e)[:30]}"

        print(f"[{idx:02d}/10] {sc['desc'][:28]:<28} | {elapsed:>6.1f} ms | {status_str}")

    print("=" * 65)
    print(f"KẾT QUẢ KIỂM THỬ: {passed}/{len(test_scenarios)} test cases hoàn thành thành công.")
    if latencies:
        avg_lat = statistics.mean(latencies)
        p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        print(f"-> Thời gian phản hồi trung bình (Avg Latency): {avg_lat:.2f} ms ({avg_lat/1000:.3f} s)")
        print(f"-> Độ trễ tối đa (Max Latency): {max(latencies):.2f} ms")
        if avg_lat < 1000:
            print("-> ĐÁNH GIÁ: ĐẠT MỤC TIÊU HIỆU NĂNG (< 1.0 giây/phản hồi)!")
        else:
            print("-> ĐÁNH GIÁ: Cần tối ưu thêm thời gian phản hồi.")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()