import io
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw
import numpy as np

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    piexif = None
    PIEXIF_AVAILABLE = False

from .document_scanner import mask_evidence
from .sensitive_detector import PATTERNS, CATEGORY_ROLE_MAP

# Lazy-loaded EasyOCR Reader
_EASYOCR_READER = None

def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # Load English OCR model in CPU mode
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception:
            _EASYOCR_READER = None
    return _EASYOCR_READER


def extract_exif_metadata(image: Image.Image) -> Dict[str, Any]:
    """
    Extracts EXIF metadata from photos (including GPS coordinates, camera model, date).
    """
    metadata = {
        "has_gps": False,
        "gps_info": {},
        "raw_tags": {},
        "risk_level": "Low"
    }

    if PIEXIF_AVAILABLE and piexif:
        try:
            exif_dict = piexif.load(image.info.get("exif", b""))
            gps = exif_dict.get("GPS", {})
            if gps:
                metadata["has_gps"] = True
                metadata["risk_level"] = "High"
                metadata["gps_info"] = {
                    "latitude_ref": gps.get(piexif.GPSIFD.GPSLatitudeRef, b"").decode("latin-1", errors="ignore"),
                    "longitude_ref": gps.get(piexif.GPSIFD.GPSLongitudeRef, b"").decode("latin-1", errors="ignore"),
                    "timestamp": str(gps.get(piexif.GPSIFD.GPSTimeStamp, "")),
                }

            zeroth = exif_dict.get("0th", {})
            make = zeroth.get(piexif.ImageIFD.Make, b"").decode("latin-1", errors="ignore").strip()
            model = zeroth.get(piexif.ImageIFD.Model, b"").decode("latin-1", errors="ignore").strip()
            if make or model:
                metadata["raw_tags"]["device"] = f"{make} {model}".strip()

        except Exception:
            pass
    else:
        # Fallback to Pillow's built-in _getexif()
        try:
            exif = image._getexif() if hasattr(image, "_getexif") else None
            if exif:
                from PIL.ExifTags import TAGS, GPSTAGS
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo":
                        metadata["has_gps"] = True
                        metadata["risk_level"] = "High"
                        metadata["gps_info"] = {GPSTAGS.get(k, k): v for k, v in value.items()} if isinstance(value, dict) else {}
                    elif tag in ("Make", "Model"):
                        metadata["raw_tags"]["device"] = f"{metadata['raw_tags'].get('device', '')} {value}".strip()
        except Exception:
            pass

    return metadata


def strip_image_metadata(image: Image.Image) -> Image.Image:
    """
    Completely strips EXIF, GPS, camera serials, and timestamps from the image.
    """
    clean_image = Image.new(image.mode, image.size)
    clean_image.putdata(list(image.getdata()))
    return clean_image


def scan_image_ocr(image: Image.Image) -> Tuple[List[Dict[str, Any]], str, List[Tuple[int, int, int, int]]]:
    """
    Runs Deep Local OCR on screenshot or photo to extract text, pinpoint bounding boxes,
    and detect sensitive PII, passwords, and tokens embedded in the image pixels.
    Returns: (findings_list, full_extracted_text, bounding_boxes_of_sensitive_regions)
    """
    findings = []
    extracted_text_lines = []
    sensitive_boxes = []

    reader = get_easyocr_reader()
    if reader is not None:
        try:
            # Convert PIL image to numpy array for EasyOCR
            img_np = np.array(image.convert("RGB"))
            results = reader.readtext(img_np)
            
            for bbox, text, conf in results:
                extracted_text_lines.append(text)
                
                # Check text snippet against all PII and sensitive patterns
                for pat_name, pat_regex in PATTERNS.items():
                    for match in pat_regex.finditer(text):
                        matched_val = match.group(0)
                        
                        # bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        xs = [pt[0] for pt in bbox]
                        ys = [pt[1] for pt in bbox]
                        box_rect = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                        sensitive_boxes.append(box_rect)

                        findings.append({
                            "Location": f"Bounding Box: {box_rect}",
                            "Category": pat_name,
                            "Role": CATEGORY_ROLE_MAP.get(pat_name, "Direct Identifier"),
                            "Evidence": mask_evidence(matched_val),
                            "Confidence": f"EasyOCR ({int(conf*100)}%)",
                            "Recommended Action": "Black-out / blur this image bounding box."
                        })
            
            return findings, "\n".join(extracted_text_lines), sensitive_boxes
        except Exception as e:
            extracted_text_lines.append(f"[EasyOCR Engine Error: {str(e)}]")

    # Fallback to Tesseract if present
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        for idx, line in enumerate(text.splitlines(), 1):
            for pat_name, pat_regex in PATTERNS.items():
                for match in pat_regex.finditer(line):
                    matched_val = match.group(0)
                    findings.append({
                        "Location": f"OCR Line {idx}",
                        "Category": pat_name,
                        "Role": CATEGORY_ROLE_MAP.get(pat_name, "Direct Identifier"),
                        "Evidence": mask_evidence(matched_val),
                        "Confidence": "Tesseract OCR",
                        "Recommended Action": "Black-out this region on the image."
                    })
        return findings, text, sensitive_boxes
    except Exception:
        pass

    return findings, "[OCR Notice: OCR model initialized. No text detected or failed to parse.]", sensitive_boxes


def auto_redact_sensitive_regions(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    color: str = "black"
) -> Image.Image:
    """
    Automatically blackouts all detected sensitive bounding boxes on the image.
    """
    redacted = image.copy()
    draw = ImageDraw.Draw(redacted)
    for box in boxes:
        # Add 4px padding around bounding box for clean coverage
        pad_box = (max(0, box[0] - 4), max(0, box[1] - 4), box[2] + 4, box[3] + 4)
        draw.rectangle(pad_box, fill=color)
    return redacted


def redact_image_region(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    color: str = "black"
) -> Image.Image:
    """Redacts (blacks-out) a specific rectangular bounding box on an image."""
    redacted = image.copy()
    draw = ImageDraw.Draw(redacted)
    draw.rectangle(box, fill=color)
    return redacted
