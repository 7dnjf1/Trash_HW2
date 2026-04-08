from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from schemas import WasteResponse, WasteClassificationResult
from model import classify_waste

app = FastAPI(
    title="지능형 분리수거 도우미 API 서버",
    description="2026년 한국 분리배출 기준을 따르는 AI 쓰레기 이미지 무가 추론 및 정보 반환 시스템",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to AI Waste Classification API Server.",
        "docs_url": "/docs"
    }

@app.post("/api/v1/classify", response_model=WasteResponse)
async def classify_image(file: UploadFile = File(...)):
    """
    쓰레기 사진을 업로드하면, AI 모델을 통해 폐기물의 종류를 분석하고 
    2026년 기준 분리배출 요령과 과태료 안내를 반환합니다.
    """
    # 기본 이미지 확장자 및 타입 체크
    valid_content_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in valid_content_types:
        raise HTTPException(
            status_code=400, 
            detail="지원되지 않는 파일 형식입니다. JPEG, PNG, WEBP 파일만 가능합니다."
        )
    
    try:
        # 파일 바이너리 읽기
        contents = await file.read()
        
        # 모델 추론 수행 (내부에서 PIL 변환 -> transformers 파이프라인 추론 -> 매핑 진행)
        result = classify_waste(contents)
        
        return WasteResponse(
            status="success",
            message="이미지 분류 완료",
            data=WasteClassificationResult(
                category=result["category"],
                confidence=result["confidence"],
                disposal_method=result["disposal_method"],
                fine_info=result["fine_info"]
            )
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"이미지 모델 분석 중 오류 발생: {str(e)}"
        )

@app.get("/healthz")
def health_check():
    """
    CI/CD 파이프라인의 헬스체크를 위한 엔드포인트입니다.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # 로컬 개발 및 테스트 시 직접 python main.py 로 구동하기 위한 세팅
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
