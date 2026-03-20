import os
import tempfile

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    return _ocr


def extract_text(file_path: str) -> dict:
    """Extract text from PDF or image using PaddleOCR."""
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
        "metadata": {
            "pages": len(results),
        },
    }


def extract_text_from_page(image_path: str) -> str:
    """Extract text from a single page image."""
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
