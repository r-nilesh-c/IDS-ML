import re
import pathlib
from pypdf import PdfReader

pdf_path = pathlib.Path(r"d:\IOMP2\Intrusion_Detection_System_for_Healthcare_Systems_Using_Medical_and_Network_Data_A_Comparison_Study.pdf")
reader = PdfReader(str(pdf_path))
text = "\n".join((p.extract_text() or "") for p in reader.pages)

reports_dir = pathlib.Path("reports")
reports_dir.mkdir(exist_ok=True)
(reports_dir / "base_paper_extracted.txt").write_text(text, encoding="utf-8")

keywords = [
    "healthcare", "medical", "iot", "hipaa", "privacy", "ehr", "hl7", "dicom", "fhir",
    "latency", "real-time", "availability", "false positive", "explainability", "compliance", "risk",
]

print(f"pages {len(reader.pages)}")
print(f"chars {len(text)}")
print("keyword_counts")
for k in keywords:
    print(f"{k} {len(re.findall(re.escape(k), text, flags=re.I))}")

headings = []
for line in text.splitlines():
    s = line.strip()
    if 5 < len(s) < 140 and (re.match(r"^(\d+(\.\d+)*)\s+[A-Z]", s) or s.isupper()):
        headings.append(s)

print("headings_sample")
for h in headings[:80]:
    print(f"- {h}")
