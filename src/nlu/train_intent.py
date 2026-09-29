import os
import json
import pickle
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def tokenize_text(text: str) -> str:
    """Tách từ tiếng Việt và chuyển chữ thường"""
    return word_tokenize(text.lower(), format="text")

def train_and_evaluate():
    train_path = "data/dialogues/train.json"
    test_path = "data/dialogues/test.json"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Lỗi: Không tìm thấy file dữ liệu train.json hoặc test.json!")
        return

    # 1. Đọc dữ liệu
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 2. Tiền xử lý
    print("Đang tiền xử lý dữ liệu và tách từ tiếng Việt...")
    X_train = [tokenize_text(item["text"]) for item in train_data]
    y_train = [item["intent"] for item in train_data]

    X_test = [tokenize_text(item["text"]) for item in test_data]
    y_test = [item["intent"] for item in test_data]

    # 3. Xây dựng Pipeline: TF-IDF + LinearSVC
    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LinearSVC(C=1.0, random_state=42))
    ])

    print("Đang huấn luyện mô hình Intent Classification...")
    model_pipeline.fit(X_train, y_train)

    # 4. Đánh giá mô hình trên tập Test độc lập
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (GIAI ĐOẠN 2)")
    print("="*50)
    print(f"Độ chính xác toàn diện (Accuracy): {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, digits=4))
    print("="*50)

    # 5. Lưu mô hình đã huấn luyện
    output_model_path = "src/nlu/intent_model.pkl"
    with open(output_model_path, "wb") as f:
        pickle.dump(model_pipeline, f)
    print(f"-> Đã lưu mô hình NLU thành công tại: {output_model_path}")

if __name__ == "__main__":
    train_and_evaluate()