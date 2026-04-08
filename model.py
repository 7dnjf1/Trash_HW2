import io
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile
app = FastAPI(title="Waste Classification API", description="이미지 업로드 시 폐기물 분류를 반환합니다.", version="1.0.0")
# Zero-shot Image Classification 파이프라인 초기화 (CLIP 기반)
# openai/clip-vit-base-patch32는 가볍고 빠르게 일반 사물 분류가 가능합니다.
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# 분류 대상 카테고리 레이블 프롬프트
CANDIDATE_LABELS = [
    "plastic bottle, plastic container", 
    "glass bottle", 
    "aluminum can", 
    "paper, cardboard", 
    "food waste", 
    "general waste, trash", 
    "battery"
]

# 2026년 기준 분리배출 방법 및 과태료 정보 매핑
WASTE_MAPPING = {
    "plastic bottle, plastic container": {
        "category": "플라스틱 (Plastic)",
        "disposal_method": "내용물을 비우고 물로 헹군 뒤, 상표(라벨)와 뚜껑을 제거하고 찌그러트려 부피를 최소화한 후 배출합니다.",
        "fine_info": "분리배출 위반 및 혼합 배출 시 10만원 이하의 과태료가 부과될 수 있습니다."
    },
    "glass bottle": {
        "category": "유리 (Glass)",
        "disposal_method": "내용물을 비우고 투명, 녹색, 갈색 등 색상별로 분리하여 지정된 전용 수거함에 배출합니다. 깨진 유리는 특수 규격 마대(불연성)에 담아 배출해야 합니다.",
        "fine_info": "분리배출 위치 위반 및 무단 투기 시 10만원 이하의 과태료가 부과될 수 있습니다."
    },
    "aluminum can": {
        "category": "캔 (Can)",
        "disposal_method": "내용물이나 이물질을 비우고 압착하여 부피를 줄인 뒤 배출합니다. 부탄가스 통 등은 완전히 비우고 구멍을 뚫어 화재 위험을 제거해야 합니다.",
        "fine_info": "분리배출 위반 시 10만원 이하의 과태료가 부과될 수 있습니다."
    },
    "paper, cardboard": {
        "category": "종이 (Paper)",
        "disposal_method": "물기나 기름, 이물질이 묻지 않도록 주의합니다. 택배 박스는 테이프와 철핀을 완전히 제거한 후 평평하게 펴서 끈으로 묶어 배출해야 합니다.",
        "fine_info": "음식물 등 이물질 혼합 배출 시 과태료 10만원 이하 부과 대상이 될 수 있습니다."
    },
    "food waste": {
        "category": "음식물 (Food)",
        "disposal_method": "물기를 최대한 짜서 제거한 후 전용 종량제 봉투나 음식물 쓰레기 수거함에 버려야 합니다. 단단한 뼈, 조개 껍데기, 차/한약 찌꺼기 등은 일반 쓰레기로 분리하세요.",
        "fine_info": "일반 쓰레기 종량제 봉투에 혼합하여 버리면 10만원 이하의 과태료가 부과됩니다."
    },
    "general waste, trash": {
        "category": "일반쓰레기 (General)",
        "disposal_method": "재활용이 불가능한 모든 생활 쓰레기는 규격에 맞는 종량제 봉투에 담아 배출해야 합니다.",
        "fine_info": "무단 투기 적발 시 20만원~100만원, 비규격 봉투 사용 시 20만원의 과태료가 부과될 수 있습니다."
    },
    "battery": {
        "category": "폐배터리 (Battery)",
        "disposal_method": "일반쓰레기에 버리면 화재나 폭발 위험이 매우 큽니다. 반드시 주민센터나 아파트, 편의점 등에 비치된 전용 폐건전지 수거함에 분리 배출해야 합니다.",
        "fine_info": "종량제 봉투에 혼합 배출 시 10만원 이하 과태료가 부과되며, 배터리로 인한 화재 발생 시 법적 책임이 크게 따를 수 있습니다."
    }
}

def classify_waste(image_bytes: bytes):
    # 바이트 데이터를 PIL 이미지로 변환
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Zero-shot 추론 실행
    results = classifier(image, candidate_labels=CANDIDATE_LABELS)
    
    # 가장 높은 확률의 결과 선택
    top_result = results[0]
    best_label = top_result["label"]
    confidence = top_result["score"]
    
    # 매핑 데이터 추출 (가끔 알 수 없는 경우가 생기면 일반쓰레기로 기본화)
    mapping_info = WASTE_MAPPING.get(best_label, WASTE_MAPPING["general waste, trash"])
        
    return {
        "category": mapping_info["category"],
        "confidence": round(confidence, 4),
        "disposal_method": mapping_info["disposal_method"],
        "fine_info": mapping_info["fine_info"]
    }


@app.post("/classify", summary="폐기물 분류", description="이미지 파일을 업로드하면 분류 결과를 반환합니다.")
async def classify_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    result = classify_waste(content)
    return result
