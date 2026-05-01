"""Extract key frame-level AUC results and techniques from research papers."""
import sys
import re
from pathlib import Path
from io import StringIO
from pdfminer.high_level import extract_text

papers_dir = Path("research papers")

pdfs = list(papers_dir.glob("*.pdf"))
print(f"Found {len(pdfs)} PDFs\n")

# Keywords to look for in context
AUC_PATTERNS = [
    r"(\d{2}\.\d{1,2})\s*%?\s*(?:AUC|AUROC|auc)",
    r"AUC[:\s]+(\d{2}\.\d{1,2})\s*%?",
    r"UCF.{0,20}(\d{2}\.\d{1,2})\s*%?",
]

for pdf in sorted(pdfs):
    print(f"\n{'='*60}")
    print(f"PAPER: {pdf.name}")
    print('='*60)
    try:
        text = extract_text(str(pdf))
        # Clean up
        text = re.sub(r'\s+', ' ', text)
        
        # Find sections with AUC numbers
        matches = []
        for pattern in AUC_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, m.start() - 150)
                end = min(len(text), m.end() + 150)
                context = text[start:end]
                val = float(m.group(1))
                if 70.0 <= val <= 99.0:  # Valid AUC range
                    matches.append((val, context.strip()))
        
        if matches:
            # Show top unique values
            seen = set()
            for val, ctx in sorted(matches, key=lambda x: -x[0])[:5]:
                if val not in seen:
                    seen.add(val)
                    print(f"  AUC={val:.2f}% | ...{ctx[:200]}...")
        else:
            print("  No AUC values found in expected range.")
        
        # Extract abstract/intro
        abstract_match = re.search(r'(?:abstract|ABSTRACT)[:\s](.{200,800})', text, re.IGNORECASE)
        if abstract_match:
            print(f"\n  [ABSTRACT snippet]: {abstract_match.group(1)[:400]}")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\n=== DONE ===")
