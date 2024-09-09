# FROM python:3.8-slim

# RUN pip config set global.index-url https://pypi.org/simple

FROM tensorflow/tensorflow:2.17.0

# 작업 디렉토리 설정
WORKDIR /flask-ml

# 필요한 패키지를 설치하기 위한 requirements.txt 복사
COPY requirements.txt .

# pip으로 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션과 모델 파일을 복사
COPY . .

# Flask 앱 실행
CMD ["python", "app.py"]