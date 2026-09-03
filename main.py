from src.loader import load_pdf


pdf_path = "data/paper/perclos_detection.pdf"

pages = load_pdf(pdf_path)

print(f"Total pages: {len(pages)}")

print("\n--- PAGE 1 ---\n")

print(pages[0]["text"][:2000])