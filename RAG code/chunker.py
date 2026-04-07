import re
import json

# ---------- FILE PATHS ----------
INPUT_FILE = "C:/Users/hetbh/Desktop/PBL-1/cleaning/VidhurNiti_chunks.json"
OUTPUT_FILE = "C:/Users/hetbh/Desktop/PBL-1/VidhurNiti_chunked.jsonl"


# ---------- DETECT SECTION ----------
def detect_section(text):
    if "Part First" in text:
        return "Part First", 1
    elif "Part Second" in text:
        return "Part Second", 2
    return None, None


# ---------- DETECT VERSE ----------
def detect_verse(text):
    match = re.match(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)", text.strip())
    if match:
        return match.group(1)
    return None


# ---------- CLASSIFY CONTENT ----------
def classify_content(text):
    text_lower = text.lower()

    if any(word in text_lower for word in ["said", "replied", "asked"]):
        return "dialogue"

    if any(word in text_lower for word in ["should", "must", "wise", "knowledge", "soul"]):
        return "teaching"

    return "narrative"


# ---------- EXTRACT KEYWORDS ----------
def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())
    return list(set(words[:5]))


# ---------- EXTRACT TEXT FROM CHUNK ----------
def extract_text(chunk):
    match = re.search(r"\]\s*\(\d+ chars\)\n(.+)", chunk, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------- MAIN PROCESS ----------
def process_file():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Split chunks safely
    raw_chunks = content.split("[Chunk")
    
    data_count = 0
    section = None
    section_num = None
    verse_counter = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            text = extract_text(chunk)
            if not text:
                continue

            # Detect section
            sec, sec_num = detect_section(text)
            if sec:
                section = sec
                section_num = sec_num
                verse_counter = 0  # reset for new section

            # Detect verse
            verse = detect_verse(text)
            if verse:
                verse_counter += 1

            # Skip if section not found yet
            if section is None:
                continue

            # Classify content
            content_type = classify_content(text)

            # Extract keywords
            keywords = extract_keywords(text)

            # Build entry
            entry = {
                "id": f"katha_{section_num}_{verse_counter}",
                "source": "Katha Upanishad",
                "entry_type": "verse",
                "section": section,
                "section_number": section_num,
                "verse_number": verse_counter,
                "content": {
                    "original": text,
                    "type": content_type
                },
                "metadata": {
                    "keywords": keywords,
                    "confidence": "medium"
                }
            }

            # ✅ WRITE AS JSONL (FIXED)
            out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

            data_count += 1

    print(f"✅ Done! {data_count} entries written to JSONL.")


# ---------- RUN ----------
if __name__ == "__main__":
    process_file()
