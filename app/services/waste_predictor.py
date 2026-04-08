import io
import base64
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# 검증된 CLIP 기반 Zero-shot Image Classification 모델 유지
# (OwlViT object detection은 파인튜닝 없이 실사용 신뢰도가 낮아 CLIP으로 대체)
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

# 카테고리별 색상 맵 (RGB)
COLOR_MAP = {
    "plastic": (0, 200, 80),    # 초록
    "glass":   (30, 120, 255),  # 파랑
    "metal":   (220, 50, 50),   # 빨강
    "paper":   (230, 200, 0),   # 노랑
    "food":    (139, 90, 43),   # 갈색
    "general": (130, 130, 130), # 회색
    "special": (150, 50, 200)   # 보라
}

# 2026년 기준 법규 + 카테고리 + 배출 지침 통합 매핑
WASTE_DATABASE = {
    "plastic bottle, plastic container": {
        "label": "플라스틱 / PET병",
        "category": "plastic",
        "guideline": (
            "내용물을 비우고 물로 헹군 뒤 라벨을 제거하고 납작하게 찌그러트려 배출하세요.\n"
            "【2026 신규】 PET병은 rPET 10% 이상 사용이 권장되며, "
            "라벨은 QR코드로 대체됩니다. 라벨 제거가 필수입니다."
        ),
        "epr_target": False
    },
    "plastic toy": {
        "label": "플라스틱 장난감",
        "category": "plastic",
        "guideline": (
            "플라스틱류와 동일하게 분리배출합니다.\n"
            "【2026 신규】 플라스틱 장난감은 EPR(생산자책임재활용) 대상 품목으로 "
            "새롭게 지정되었습니다. 반드시 플라스틱 수거함에 배출하세요."
        ),
        "epr_target": True
    },
    "glass bottle": {
        "label": "유리병",
        "category": "glass",
        "guideline": "내용물을 비우고 색상별(투명·녹색·갈색)로 분리해 전용 수거함에 배출하세요.",
        "epr_target": False
    },
    "aluminum can": {
        "label": "캔 / 금속",
        "category": "metal",
        "guideline": "내용물을 비우고 압착해 부피를 줄인 뒤 배출하세요. 가스통은 구멍을 뚫어야 합니다.",
        "epr_target": False
    },
    "paper, cardboard": {
        "label": "종이 / 박스",
        "category": "paper",
        "guideline": "테이프와 철핀을 제거한 뒤 평평하게 접어 끈으로 묶어 배출하세요.",
        "epr_target": False
    },
    "food waste": {
        "label": "음식물 쓰레기",
        "category": "food",
        "guideline": "물기를 최대한 제거한 뒤 전용 음식물 쓰레기 봉투 또는 수거함에 배출하세요.",
        "epr_target": False
    },
    "battery": {
        "label": "폐배터리",
        "category": "special",
        "guideline": "절대 일반쓰레기에 버리지 마세요! 편의점·주민센터의 전용 수거함에만 배출합니다.",
        "epr_target": False
    },
    "general waste, trash": {
        "label": "일반쓰레기",
        "category": "general",
        "guideline": "재활용이 불가한 쓰레기는 규격 종량제 봉투에 담아 배출하세요.",
        "epr_target": False
    }
}

CANDIDATE_LABELS = list(WASTE_DATABASE.keys())


def predict_and_annotate(image_bytes: bytes) -> dict:
    """CLIP 분류 후 결과를 V2 포맷(items, compliance_report, annotated_image)으로 반환."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # CLIP 추론
    raw_results = classifier(image, candidate_labels=CANDIDATE_LABELS)

    # 신뢰도 10% 이상인 결과만 취함 (최대 3개, 최소 1개 보장)
    filtered = [r for r in raw_results if r["score"] >= 0.10]
    if not filtered:
        filtered = raw_results[:1]  # 최소 1개는 반환

    # Pillow 드로잉 준비
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", max(16, width // 30))
    except Exception:
        font = ImageFont.load_default()

    items = []
    epr_count = 0
    box_margin = 0.06  # 박스를 이미지 경계에서 6% 안쪽으로

    for idx, res in enumerate(filtered[:3]):
        label_key = res["label"]
        score = res["score"]
        info = WASTE_DATABASE.get(label_key, WASTE_DATABASE["general waste, trash"])

        cat = info["category"]
        color = COLOR_MAP.get(cat, (130, 130, 130))

        if info.get("epr_target"):
            epr_count += 1

        # 여러 결과가 있을 경우 이미지를 위아래로 나눠 박스를 그림
        if len(filtered) == 1:
            x0 = int(width * box_margin)
            y0 = int(height * box_margin)
            x1 = int(width * (1 - box_margin))
            y1 = int(height * (1 - box_margin))
        else:
            # 이미지를 세로로 N등분
            n = min(len(filtered), 3)
            segment_h = height // n
            x0 = int(width * box_margin)
            y0 = idx * segment_h + int(segment_h * 0.05)
            x1 = int(width * (1 - box_margin))
            y1 = (idx + 1) * segment_h - int(segment_h * 0.05)

        # 박스 그리기
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        tag = f"{info['label']} ({int(score * 100)}%)"
        # 태그 배경
        bbox = font.getbbox(tag) if hasattr(font, "getbbox") else (0, 0, len(tag) * 8, 18)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x0, y0 - th - 6, x0 + tw + 10, y0], fill=color)
        draw.text((x0 + 5, y0 - th - 4), tag, fill="white", font=font)

        items.append({
            "label": info["label"],
            "category": cat,
            "confidence": round(score, 4),
            "guideline": info["guideline"],
            "box": [x0, y0, x1, y1]
        })

    # Base64 변환
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "items": items,
        "compliance_report": {
            "total_detected": len(items),
            "epr_items": epr_count
        },
        "annotated_image_base64": img_b64
    }
