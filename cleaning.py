import re
content = ""
content = re.sub(r'Điều(\d)', r'Điều \1', content)
content = re.sub(r'CHƯƠNG(\w)', r'CHƯƠNG \1', content)
content = re.sub(r'Mục(\w)', r'Mục \1', content)
content = re.sub(r'Content from:.*?https?://[^\s]+', '', content)
content = re.sub(r'={2,}|_{2,}|-{2,}', '', content)
content = re.sub(r'\n{2,}', '\n', content)


