from pypdf import PdfReader

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text

def create_chunks(text):

    lines = text.split("\n")

    chunks = []

    buffer = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # if line looks like structured field → keep separately
        if ":" in line or len(line) < 80:

            if buffer:
                chunks.append(buffer.strip())
                buffer = ""

            chunks.append(line)

        else:
            buffer += " " + line

    if buffer:
        chunks.append(buffer.strip())

    return chunks