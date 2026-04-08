FROM python:3.9-slim

# 보안 및 성능 최적화 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 변경 빈도가 낮은 요구사항(requirements) 먼저 복사하여 빌드 캐시 활용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 진짜 다중 객체 인식을 위해 탐지 전용 모델(OwlViT)을 캐시합니다.
RUN python -c "from transformers import pipeline; pipeline('zero-shot-object-detection', model='google/owlvit-base-patch32')"

# 나머지 프로젝트 소스 코드 복사
COPY . .

# FastAPI 서버가 실행되는 포트 오픈
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
