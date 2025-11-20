def segment_document(text: str) -> dict:
    import re

    # Simple fallback segmentation using known section headers
    headers = [
        "Management's Discussion and Analysis",
        "Risk Factors",
        "Financial Statements"
    ]

    segments = {}
    current_header = None
    buffer = []

    for line in text.splitlines():
        line = line.strip()
        if line in headers:
            if current_header and buffer:
                segments[current_header] = "\n".join(buffer)
                buffer = []
            current_header = line
        elif current_header:
            buffer.append(line)

    if current_header and buffer:
        segments[current_header] = "\n".join(buffer)

    return segments
