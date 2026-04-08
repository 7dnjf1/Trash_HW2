from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from schemas import WasteResponseV2
from app.services.waste_predictor import predict_and_annotate

app = FastAPI(
    title="지능형 분리수거 도우미 API 서버",
    description="2026년 한국 분리배출 기준을 따르는 AI 쓰레기 이미지 무가 추론 및 정보 반환 시스템",
    version="1.0.0"
)

@app.get("/")
def read_root():
    # 이제 JSON 대신 사용자를 위한 예쁜 웹 인터페이스를 반환합니다.
    return FileResponse("templates/index.html")

@app.post("/api/v1/classify", response_model=WasteResponseV2)
async def classify_image(file: UploadFile = File(...)):
    """
    쓰레기 사진을 업로드하면, AI 모델을 통해 폐기물의 종류를 다중 분석하고 
    2026년 기준 분리배출 요령과 Bounding Box가 포함된 이미지를 반환합니다.
    """
    valid_content_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in valid_content_types:
        raise HTTPException(
            status_code=400, 
            detail="지원되지 않는 파일 형식입니다. JPEG, PNG, WEBP 파일만 가능합니다."
        )
    
    try:
        contents = await file.read()
        
        # V2 고도화 서비스 호출 (Object Detection + Visualization)
        result = predict_and_annotate(contents)
        
        return WasteResponseV2(
            status="success",
            message="객체 탐지 및 분석 완료",
            items=result["items"],
            compliance_report=result["compliance_report"],
            annotated_image_base64=result["annotated_image_base64"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"AI 모델 분석 중 오류 발생: {str(e)}"
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
