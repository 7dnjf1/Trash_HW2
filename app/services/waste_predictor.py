import io
import base64
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import torch

# Zero-shot Object Detection 모델 초기화 (OwlViT - OpenAI CLIP 기반의 강력한 객체 탐지 모델)
# google/owlvit-base-patch32 를 사용하여 텍스트 쿼리만으로 객체를 탐지합니다.
detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")

# 카테고리별 컬러 및 법규 정보 매핑
COLOR_MAP = {
    "plastic": "#00FF00",  # 초록
    "glass": "#0000FF",    # 파랑
    "metal": "#FF0000",    # 빨강
    "paper": "#FFFF00",    # 노랑
    "food": "#8B4513",     # 갈색
    "general": "#808080",  # 회색
    "special": "#800080"   # 보라 (폐배터리 등)
}

# 2026년 기준 상세 법규 및 지침
LAW_2026 = {
    "PET bottle": {
        "category": "plastic",
        "guideline": "rPET 10% 이상 사용 권장 + 라벨 제거 필수 (2026년부터 QR코드로 대체됨)",
        "epr_target": True
    },
    "plastic toy": {
        "category": "plastic",
        "guideline": "2026년부터 EPR 대상! 플라스틱류와 동일하게 분리배출 하세요.",
        "epr_target": True
    },
    "glass": {"category": "glass", "guideline": "색상별 분리배출, 깨진 유리는 불연성 마대 사용"},
    "can": {"category": "metal", "guideline": "내용물 비우고 압착, 가스통은 구멍 뚫기"},
    "paper": {"category": "paper", "guideline": "테이프 제거 후 평평하게 펴서 배출"},
    "food": {"category": "food", "guideline": "물기 제거 후 전용 수거함 배출"},
    "trash": {"category": "general", "guideline": "종량제 봉투에 담아 배출"},
    "battery": {"category": "special", "guideline": "반드시 전용 수거함에 배출 (화재 위험)"}
}

# 모델이 인식할 텍스트 쿼리 (레이아웃 최적화)
CANDIDATE_QUERIES = [
    "PET bottle", "plastic toy", "plastic bottle", "glass bottle", 
    "aluminum can", "cardboard box", "paper", "food waste", 
    "battery", "trash bag"
]

def predict_and_annotate(image_bytes: bytes):
    # 이미지 로드
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # 객체 탐지 실행
    results = detector(image, candidate_labels=CANDIDATE_QUERIES, threshold=0.1)

    items = []
    draw = ImageDraw.Draw(image)
    
    # 폰트 설정 (기본 폰트 사용, 필요시 경로 지정 필요)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    compliance_summary = {"total_detected": len(results), "epr_items": 0}

    for res in results:
        label = res["label"]
        score = res["score"]
        box = res["box"]  # {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}

        # 2026 법규 정보 및 카테고리 매핑
        law_info = LAW_2026.get(label, {"category": "general", "guideline": "일반 분리배출 기준 준수"})
        category = law_info.get("category", "general")
        color = COLOR_MAP.get(category, "#808080")
        
        if law_info.get("epr_target"):
            compliance_summary["epr_items"] += 1

        # 시각화 (Bounding Box)
        # OwlViT의 box는 0~1 사이의 정규화된 값이 아닌 픽셀 값으로 반환될 수 있으므로 확인 필요 (Transformers 버전에 따라 다름)
        # 보통 transformers pipeline은 픽셀 값을 반환함
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
        draw.text((xmin, ymin - 25), f"{label} ({int(score*100)}%)", fill=color, font=font)

        items.append({
            "label": label,
            "category": category,
            "confidence": round(score, 4),
            "guideline": law_info["guideline"],
            "box": [xmin, ymin, xmax, ymax]
        })

    # 어노테이션된 이미지를 Base64로 변환
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "items": items,
        "compliance_report": compliance_summary,
        "annotated_image_base64": img_str
    }
