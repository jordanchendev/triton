_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="chinese_cht", use_gpu=True)
    return _ocr


def extract_text(file_path: str) -> dict:
    """Extract text from PDF or image using PaddleOCR."""
    ocr = _get_ocr()
    results = ocr.ocr(file_path, cls=True)

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
