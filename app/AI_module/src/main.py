"""
Illegal Parking Detection - AI Layer
FastAPI service: accepts base64 image + GPS, returns plate OCR + vehicle metadata.
Dependencies: fastapi, uvicorn, paddleocr, opencv-python-headless, numpy
"""

import base64
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Parking AI Layer",
    description="Licence plate OCR + vehicle metadata extraction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# PaddleOCR - loaded once at startup (CPU-only, angle classification on)
# ---------------------------------------------------------------------------

ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=False,
    show_log=False,
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnalyseRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG or PNG of the vehicle")
    latitude: float = Field(..., description="GPS latitude from the officer's device")
    longitude: float = Field(..., description="GPS longitude from the officer's device")
    timestamp: Optional[str] = Field(None, description="ISO-8601 capture time; server fills if absent")
    officer_id: Optional[str] = Field(None)
    device_id: Optional[str] = Field(None)


class PlateResult(BaseModel):
    text: Optional[str]
    confidence: float
    obscured: bool


class VehicleResult(BaseModel):
    color: Optional[str]
    type: Optional[str]


class AnalyseResponse(BaseModel):
    request_id: str
    timestamp: str
    location: dict
    plate: PlateResult
    vehicle: VehicleResult
    image_base64: str
    analysis_notes: str
    officer_id: Optional[str]
    device_id: Optional[str]
    model_version: str = "paddleocr-v3-cpu"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def decode_image(b64: str) -> np.ndarray:
    """Decode base64 string to an OpenCV BGR ndarray."""
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=422, detail={
            "error": "unprocessable_image",
            "message": "Base64 decoding failed. Ensure the string has no data-URL prefix.",
        })
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail={
            "error": "unprocessable_image",
            "message": "Could not decode image. Ensure it is a valid JPEG or PNG.",
        })
    return img


def detect_plate_region(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Attempt to isolate the licence plate region using edge detection + contour
    filtering. Returns a cropped plate ROI if found, or None to fall back to
    full-image OCR.
    """
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
    """Sharpen, upscale, and threshold a plate ROI for OCR."""
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


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

PLATE_PATTERN = re.compile(r"[^A-Z0-9\s\-]")


def clean_plate_text(raw: str) -> str:
    """Uppercase, strip noise chars, collapse whitespace."""
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


def upscale_box(box: tuple[int, int, int, int], source_shape: tuple[int, int], target_shape: tuple[int, int]) -> tuple[int, int, int, int]:
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
    """
    Run OCR and return cleaned text candidates with confidence and bounding box.
    """
    result = ocr_engine.ocr(image, cls=True)
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

            candidates.append({
                "text": cleaned,
                "confidence": conf,
                "box": box,
            })
    return candidates


def expand_plate_to_vehicle_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """
    Approximate a vehicle crop around the winning plate box.
    """
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


# ---------------------------------------------------------------------------
# Vehicle heuristics
# ---------------------------------------------------------------------------


def estimate_vehicle_color(img: np.ndarray) -> Optional[str]:
    """
    Estimate vehicle color from a crop near the detected plate.
    The crop is filtered to reduce road/background bias.
    """
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
    """
    Use the plate-derived crop aspect ratio instead of the full image aspect ratio.
    """
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


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@app.post("/api/v1/analyse", response_model=AnalyseResponse)
async def analyse(req: AnalyseRequest):
    img = decode_image(req.image_base64)

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
        raise HTTPException(status_code=400, detail={
            "error": "no_plate_detected",
            "message": "Could not identify a number plate in the image.",
        })

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

    return AnalyseResponse(
        request_id=f"req_{uuid.uuid4().hex[:16]}",
        timestamp=req.timestamp or datetime.now(timezone.utc).isoformat(),
        location={"latitude": req.latitude, "longitude": req.longitude},
        plate=PlateResult(
            text=plate_text,
            confidence=plate_conf,
            obscured=obscured,
        ),
        vehicle=VehicleResult(
            color=vehicle_color,
            type=vehicle_type,
        ),
        image_base64=req.image_base64,
        analysis_notes=" ".join(notes_parts),
        officer_id=req.officer_id,
        device_id=req.device_id,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model": "paddleocr-v3-cpu"}
