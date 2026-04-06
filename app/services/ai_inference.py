import base64
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import HTTPException
from paddleocr import PaddleOCR
from starlette.concurrency import run_in_threadpool


PLATE_PATTERN = re.compile(r"[^A-Z0-9\s\-]")
EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "static" / "evidence"

_ocr_engine: Optional[PaddleOCR] = None


@dataclass
class AIInferenceResult:
    request_id: str
    timestamp: str
    plate_text: str
    plate_confidence: float
    plate_obscured: bool
    vehicle_color: Optional[str]
    vehicle_type: Optional[str]
    analysis_notes: str
    model_version: str
    evidence_image_url: str
    latitude: Optional[float]
    longitude: Optional[float]
    officer_id: Optional[str]
    device_id: Optional[str]


def init_ocr_engine() -> None:
    global _ocr_engine
    if _ocr_engine is not None:
        return

    _ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
    )


def shutdown_ocr_engine() -> None:
    global _ocr_engine
    _ocr_engine = None


def _get_ocr_engine() -> PaddleOCR:
    if _ocr_engine is None:
        raise RuntimeError("OCR engine is not initialized.")
    return _ocr_engine


def decode_image(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unprocessable_image",
                "message": "Base64 decoding failed. Ensure the string has no data-URL prefix.",
            },
        ) from exc

    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unprocessable_image",
                "message": "Could not decode image. Ensure it is a valid JPEG or PNG.",
            },
        )
    return img


def save_evidence_image(img: np.ndarray) -> str:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"capture_{uuid.uuid4().hex}.jpg"
    file_path = EVIDENCE_DIR / file_name

    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise HTTPException(
            status_code=500,
            detail={"error": "evidence_encoding_failed", "message": "Could not encode evidence image."},
        )

    file_path.write_bytes(encoded.tobytes())
    return f"/static/evidence/{file_name}"


def detect_plate_region(img: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    total_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        area = cv2.contourArea(contour)
        x, y, crop_w, crop_h = cv2.boundingRect(contour)

        if crop_h == 0:
            continue

        aspect = crop_w / crop_h
        rel_area = area / total_area

        if len(approx) >= 4 and 1.5 < aspect < 6.5 and 0.003 < rel_area < 0.12:
            pad = 6
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + crop_w + pad)
            y2 = min(h, y + crop_h + pad)
            return img[y1:y2, x1:x2]

    return None


def preprocess_for_ocr(roi: np.ndarray) -> np.ndarray:
    height, width = roi.shape[:2]
    if height < 80:
        scale = 80 / height
        roi = cv2.resize(roi, (int(width * scale), 80), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def clean_plate_text(raw: str) -> str:
    upper = raw.upper()
    cleaned = PLATE_PATTERN.sub("", upper)
    return " ".join(cleaned.split()).strip()


def normalize_box(box: list[list[float]], width: int, height: int) -> tuple[int, int, int, int]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(width, int(max(xs)))
    y2 = min(height, int(max(ys)))
    return x1, y1, x2, y2


def upscale_box(
    box: tuple[int, int, int, int],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_h, src_w = source_shape[:2]
    dst_h, dst_w = target_shape[:2]
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    x1, y1, x2, y2 = box
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def extract_ocr_candidates(image: np.ndarray, box_scale_source: Optional[tuple[int, int]] = None) -> list[dict]:
    result = _get_ocr_engine().ocr(image, cls=True)
    height, width = image.shape[:2]
    candidates = []
    if result and result[0]:
        for line in result[0]:
            text_raw = line[1][0]
            conf = float(line[1][1])
            cleaned = clean_plate_text(text_raw)
            if not cleaned:
                continue

            box = normalize_box(line[0], width, height)
            if box_scale_source is not None:
                box = upscale_box(box, image.shape, box_scale_source)

            candidates.append({"text": cleaned, "confidence": conf, "box": box})
    return candidates


def expand_plate_to_vehicle_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    plate_w = max(1, x2 - x1)
    plate_h = max(1, y2 - y1)
    plate_cx = x1 + plate_w / 2
    plate_cy = y1 + plate_h / 2

    crop_width = min(w, int(max(plate_w * 5.2, w * 0.48)))
    crop_height = min(h, int(max(plate_h * 5.8, h * 0.26)))
    crop_cx = plate_cx
    crop_cy = max(crop_height / 2, plate_cy - plate_h * 1.2)

    vx1 = int(crop_cx - crop_width / 2)
    vx2 = vx1 + crop_width
    vy1 = int(crop_cy - crop_height / 2)
    vy2 = vy1 + crop_height

    if vx1 < 0:
        vx2 -= vx1
        vx1 = 0
    if vx2 > w:
        shift = vx2 - w
        vx1 = max(0, vx1 - shift)
        vx2 = w
    if vy1 < 0:
        vy2 -= vy1
        vy1 = 0
    if vy2 > h:
        shift = vy2 - h
        vy1 = max(0, vy1 - shift)
        vy2 = h

    if vx2 <= vx1 or vy2 <= vy1:
        return img
    return img[vy1:vy2, vx1:vx2]


def pick_best_plate_candidate(candidates: list[dict]) -> Optional[dict]:
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if len(candidate["text"].replace(" ", "")) >= 3:
            return candidate
    return None


def estimate_vehicle_color(img: np.ndarray) -> Optional[str]:
    if img.size == 0:
        return None

    h, w = img.shape[:2]
    y1 = max(0, int(h * 0.28))
    y2 = min(h, int(h * 0.72))
    x1 = max(0, int(w * 0.10))
    x2 = min(w, int(w * 0.90))
    focus = img[y1:y2, x1:x2]
    if focus.size == 0:
        focus = img

    hsv = cv2.cvtColor(focus, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    valid_mask = v_channel > 35
    chromatic_mask = valid_mask & (s_channel >= 55)

    chromatic_pixels = int(np.count_nonzero(chromatic_mask))
    if chromatic_pixels >= max(250, int(valid_mask.size * 0.10)):
        hues = h_channel[chromatic_mask]
        counts = {
            "red": int(np.count_nonzero((hues <= 10) | (hues >= 171))),
            "orange": int(np.count_nonzero((hues >= 11) & (hues <= 25))),
            "yellow": int(np.count_nonzero((hues >= 26) & (hues <= 34))),
            "green": int(np.count_nonzero((hues >= 35) & (hues <= 85))),
            "blue": int(np.count_nonzero((hues >= 86) & (hues <= 130))),
            "purple": int(np.count_nonzero((hues >= 131) & (hues <= 155))),
            "pink": int(np.count_nonzero((hues >= 156) & (hues <= 170))),
        }
        best_color = max(counts, key=counts.get)
        if counts[best_color] >= max(150, int(chromatic_pixels * 0.28)):
            return best_color

    achromatic_mask = valid_mask & (s_channel < 55)
    if int(np.count_nonzero(achromatic_mask)) == 0:
        return None

    avg_v = float(np.mean(v_channel[achromatic_mask]))
    if avg_v > 205:
        return "white"
    if avg_v > 155:
        return "silver"
    if avg_v > 95:
        return "gray"
    return "black"


def estimate_vehicle_type(img: np.ndarray) -> Optional[str]:
    h, w = img.shape[:2]
    if h == 0:
        return None

    aspect = w / h
    if aspect > 2.35:
        return "truck or van"
    if aspect > 1.55:
        return "sedan or SUV"
    if aspect > 1.05:
        return "hatchback or SUV"
    return "motorcycle"


def _analyse_capture_sync(
    image_base64: str,
    latitude: Optional[float],
    longitude: Optional[float],
    timestamp: Optional[str],
    officer_id: Optional[str],
    device_id: Optional[str],
) -> AIInferenceResult:
    img = decode_image(image_base64)
    evidence_image_url = save_evidence_image(img)

    plate_roi = detect_plate_region(img)
    obscured = False
    notes_parts = []
    candidates = []

    if plate_roi is not None:
        notes_parts.append("Plate-like region detected; comparing ROI OCR with full-image OCR.")
        candidates.extend(extract_ocr_candidates(plate_roi))
        preprocessed_roi = preprocess_for_ocr(plate_roi)
        candidates.extend(extract_ocr_candidates(preprocessed_roi, box_scale_source=plate_roi.shape[:2]))
    else:
        notes_parts.append("No distinct plate region found; running OCR on full image.")

    candidates.extend(extract_ocr_candidates(img))

    best_candidate = pick_best_plate_candidate(candidates)
    if best_candidate is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "no_plate_detected", "message": "Could not identify a number plate in the image."},
        )

    plate_text = best_candidate["text"]
    plate_conf = round(best_candidate["confidence"], 4)

    if plate_conf < 0.6:
        obscured = True
        notes_parts.append(f"Low OCR confidence ({plate_conf:.0%}) - plate may be obscured or blurry.")

    vehicle_crop = expand_plate_to_vehicle_crop(img, best_candidate["box"])
    vehicle_color = estimate_vehicle_color(vehicle_crop)
    vehicle_type = estimate_vehicle_type(vehicle_crop)

    notes_parts.append(f"OCR confidence: {plate_conf:.0%}.")
    notes_parts.append("Vehicle color/type estimated from a crop centered on the detected plate.")
    if vehicle_color:
        notes_parts.append(f"Dominant vehicle color: {vehicle_color}.")

    return AIInferenceResult(
        request_id=f"req_{uuid.uuid4().hex[:16]}",
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        plate_text=plate_text,
        plate_confidence=plate_conf,
        plate_obscured=obscured,
        vehicle_color=vehicle_color,
        vehicle_type=vehicle_type,
        analysis_notes=" ".join(notes_parts),
        model_version="paddleocr-v3-cpu",
        evidence_image_url=evidence_image_url,
        latitude=latitude,
        longitude=longitude,
        officer_id=officer_id,
        device_id=device_id,
    )


async def analyse_capture(
    image_base64: str,
    latitude: Optional[float],
    longitude: Optional[float],
    timestamp: Optional[str],
    officer_id: Optional[str],
    device_id: Optional[str],
) -> AIInferenceResult:
    return await run_in_threadpool(
        _analyse_capture_sync,
        image_base64,
        latitude,
        longitude,
        timestamp,
        officer_id,
        device_id,
    )
