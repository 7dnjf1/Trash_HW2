import io
import base64
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# 다중 객체 인식을 위해 OwlViT 모델을 사용합니다.
detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")

COLOR_MAP = {
    "plastic": "#00FF00",  # 초록
    "glass": "#0000FF",    # 파랑
    "metal": "#FF0000",    # 빨강
    "paper": "#FFFF00",    # 노랑
    "food": "#8B4513",     # 갈색
    "general": "#808080",  # 회색
    "special": "#800080"   # 보라
}

LAW_2026 = {
    "plastic bottle": { "category": "plastic", "guideline": "rPET 10% 이상 사용 권장 + 라벨 제거 필수 (QR코드 대체)", "epr_target": True },
    "plastic toy": { "category": "plastic", "guideline": "2026년부터 EPR 대상! 플라스틱류와 동일하게 배출", "epr_target": True },
    "glass bottle": { "category": "glass", "guideline": "색상별 분리배출, 깨진 유리는 불연성 마대 사용" },
    "aluminum can": { "category": "metal", "guideline": "내용물 비우고 압착, 가스통은 구멍 뚫기" },
    "cardboard box": { "category": "paper", "guideline": "테이프 제거 후 평평하게 펴서 배출" },
    "paper": { "category": "paper", "guideline": "종이류로 배출" },
    "food waste": { "category": "food", "guideline": "물기 제거 후 전용 수거함 배출" },
    "trash bag": { "category": "general", "guideline": "종량제 봉투에 담아 배출" },
    "battery": { "category": "special", "guideline": "반드시 전용 수거함에 배출 (화재 위험)" },
    "trash": { "category": "general", "guideline": "단순 일반쓰레기는 분리배출 지침 확인" }
}

CANDIDATE_QUERIES = list(LAW_2026.keys())

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def predict_and_annotate(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # threshold를 극단적으로 낮춰 일단 모든 의심 객체를 찾도록 합니다 (0.02)
    results = detector(image, candidate_labels=CANDIDATE_QUERIES, threshold=0.02)

    items = []
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    compliance_summary = {"total_detected": 0, "epr_items": 0}

    # NMS (겹치는 영역 제거 로직)
    filtered_results = []
    # 점수가 높은 순으로 정렬
    for res in sorted(results, key=lambda x: x["score"], reverse=True):
        box = [res["box"]["xmin"], res["box"]["ymin"], res["box"]["xmax"], res["box"]["ymax"]]
        overlap = False
        for f_res in filtered_results:
            if compute_iou(box, f_res["box"]) > 0.4: # 40% 이상 영역이 겹치면 중복으로 간주
                overlap = True
                break
        
        # 박스 크기가 너무 큰 것(배경 전체 인식) 배제 로직 추가 (이미지 넓이의 90% 이상이면 무시)
        img_area = image.width * image.height
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area > img_area * 0.9:
            overlap = True

        if not overlap:
            filtered_results.append({"label": res["label"], "score": res["score"], "box": box})
            if len(filtered_results) >= 5: # 최대 5개 객체까지만 수집
                break

    for res in filtered_results:
        label = res["label"]
        score = res["score"]
        xmin, ymin, xmax, ymax = res["box"]

        law_info = LAW_2026.get(label, {"category": "general", "guideline": "일반 분리배출 기준 준수"})
        category = law_info.get("category", "general")
        color = COLOR_MAP.get(category, "#808080")
        
        if law_info.get("epr_target"):
            compliance_summary["epr_items"] += 1
        compliance_summary["total_detected"] += 1

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)
        
        # 텍스트 그리기 (배경 포함하여 가시성 확보)
        text = f"{label} ({(score*100):.1f}%)"
        bbox = draw.textbbox((xmin, max(0, ymin - 25)), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((xmin, max(0, ymin - 25)), text, fill="white", font=font)

        items.append({
            "label": label,
            "category": category,
            "confidence": round(score, 4),
            "guideline": law_info["guideline"],
            "box": [xmin, ymin, xmax, ymax]
        })

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "items": items,
        "compliance_report": compliance_summary,
        "annotated_image_base64": img_str
    }
