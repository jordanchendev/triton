import os

from triton.config import settings

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        if settings.whisper_device != "cuda":
            os.environ["FLAGS_use_mkldnn"] = "0"
        from paddleocr import PaddleOCR
        if settings.whisper_device == "cuda":
            _ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        else:
            _ocr = PaddleOCR(use_angle_cls=True, lang="ch", ocr_version="PP-OCRv4")
    return _ocr


def try_pymupdf_text(file_path: str) -> dict | None:
    """Try to extract text from PDF using PyMuPDF.
    Returns result dict if the PDF has embedded text, None if scanned/image-based.
    """
    if not file_path.lower().endswith(".pdf"):
        return None

    import fitz
    doc = fitz.open(file_path)
    pages_text = []
    pages_with_text = 0

    for page in doc:
        text = page.get_text().strip()
        pages_text.append(text)
        if len(text) > 50:
            pages_with_text += 1

    doc.close()

    # If most pages have extractable text, it's a native PDF
    if pages_with_text > len(pages_text) * 0.5:
        return {
            "text": "\n\n".join(pages_text),
            "metadata": {"pages": len(pages_text), "method": "pymupdf"},
        }
    return None


def extract_text(file_path: str) -> dict:
    """Extract text from PDF or image. Tries PyMuPDF first, falls back to OCR."""
    # Fast path: native PDF with embedded text
    pymupdf_result = try_pymupdf_text(file_path)
    if pymupdf_result:
        return pymupdf_result

    # Slow path: OCR for scanned PDFs and images
    ocr = _get_ocr()
    results = ocr.ocr(file_path)

    text_parts = []
    for page in results:
        if page is None:
            continue
        for line in page:
            text_parts.append(line[1][0])

    return {
        "text": "\n".join(text_parts),
        "metadata": {"pages": len(results), "method": "paddleocr"},
    }


def extract_text_from_page(image_path: str) -> str:
    """Extract text from a single page image using OCR."""
    ocr = _get_ocr()
    results = ocr.ocr(image_path)

    text_parts = []
    if results:
        for page in results:
            if page is None:
                continue
            for line in page:
                text_parts.append(line[1][0])

    return "\n".join(text_parts)


def split_pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """Split a PDF into per-page PNG images. Returns list of image paths."""
    import fitz

    doc = fitz.open(pdf_path)
    image_paths = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_dir, f"page_{page_num:04d}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths
