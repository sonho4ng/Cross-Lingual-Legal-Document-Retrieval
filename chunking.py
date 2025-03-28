import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def custom_split(text, max_length=1000, max_overlap_sentences=1, max_overlap_chars=200):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        joined = ' '.join(current_chunk)
        if len(joined) >= max_length:
            # Tạo chunk
            chunk_text = joined.strip()
            chunks.append(chunk_text)

            # Xử lý overlap
            overlap_sentences = current_chunk[-max_overlap_sentences:] if len(
                current_chunk) >= max_overlap_sentences else current_chunk
            overlap_text = ' '.join(overlap_sentences)


            if len(overlap_text) > max_overlap_chars:
                overlap_text = overlap_text[-max_overlap_chars:]

            current_chunk = [overlap_text]

    # Thêm phần còn lại
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())

    return chunks
