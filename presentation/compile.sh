#!/bin/bash
# Compile LaTeX presentation to PDF

# Compile twice for table of contents
pdflatex PRESENTATION_JAN29_2025.tex
pdflatex PRESENTATION_JAN29_2025.tex

echo ""
echo "====================================="
echo "PDF generated: PRESENTATION_JAN29_2025.pdf"
echo "====================================="
