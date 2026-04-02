from chunking import extract_text

text = extract_text("sample_data/loan_agreement.pdf")
print(text[:500])