# Spam Classification Application

Ứng dụng phân loại thư rác bằng Python sử dụng các mô hình học máy như Naive Bayes, SVM và Logistic Regression. Ứng dụng này giúp xác định email nào là thư rác và email nào là hợp lệ dựa trên nội dung văn bản của email.

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Cách triển khai ứng dụng web](#cách-triển-khai-ứng-dụng-web)
- [Kết quả và đánh giá mô hình](#kết-quả-và-đánh-giá-mô-hình)

## Giới thiệu

Ứng dụng sử dụng mô hình học máy để phân loại các email thành hai loại: _thư rác (spam)_ và _hợp lệ (ham)_. Các mô hình đã huấn luyện bao gồm Naive Bayes, SVM và Logistic Regression, với mục tiêu lựa chọn mô hình có hiệu suất tốt nhất để triển khai.

## Tính năng

- Huấn luyện và đánh giá nhiều mô hình khác nhau.
- Lựa chọn mô hình tốt nhất để triển khai.
- Ứng dụng web đơn giản cho phép người dùng nhập nội dung email và nhận kết quả phân loại.
- Lưu trữ mô hình đã huấn luyện và vectorizer để tái sử dụng.

## Cấu trúc dự án

```plaintext
Email-app/
├── model_training.py         # File huấn luyện mô hình
├── app.py                    # Ứng dụng web Flask
├── best_model.pkl            # Mô hình có hiệu suất tốt nhất đã lưu
├── tfidf_vectorizer.pkl      # Bộ vectorizer TF-IDF đã lưu
├── templates/
│   └── index.html            # Template HTML cho ứng dụng web
├── spam_ham_dataset.csv      # Dữ liệu huấn luyện
└── README.md                 # Hướng dẫn dự án
```

## Cài đặt

### Yêu Cầu

Để chạy ứng dụng, bạn cần cài đặt Python 3.x và các thư viện sau:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `Flask`
- `nltk`
- `matplotlib`
- `seaborn`

### Hướng Dẫn Cài Đặt

1. **Clone hoặc tải dự án về máy của bạn**:
   ```bash
   git clone <link-to-your-repo>
   cd Email-app
   ```
2. **Cài đặt các thư viện yêu cầu từ file requirements.txt**:
   ```bash
   pip install -r requirements.txt or python -m pip install -r requirements.txt
   ```
3. **Tải dữ liệu NLTK cần thiết**:
   ```bash
   import nltk
   nltk.download('stopwords')
   ```

### Sử dụng

- Huấn luyện mô hình
- Chạy model_training.py để huấn luyện các mô hình, đánh giá hiệu suất và lưu mô hình có hiệu suất tốt nhất cùng với vectorizer TF-IDF.
  ```bash
  python model_training.py
  ```

### Cách triển khai ứng dụng web

1. Chạy ứng dụng web bằng Flask:
   ```bash
   python app.py
   ```
2. Mở trình duyệt và truy cập địa chỉ http://127.0.0.1:5000 để sử dụng ứng dụng.
3. Nhập nội dung email vào ô nhập liệu và nhấn "Phân loại" để kiểm tra xem email là thư rác hay hợp lệ.

## Kết quả và đánh giá mô hình

Dưới đây là độ chính xác của từng mô hình trong quá trình huấn luyện:

- Naive Bayes: Độ chính xác 0.92
- SVM: Độ chính xác 0.99
- Logistic Regression: Độ chính xác 0.98
- Mô hình có độ chính xác cao nhất sẽ được lưu và sử dụng trong ứng dụng.

```

```
