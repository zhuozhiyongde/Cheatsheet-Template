#!/bin/bash

# 检查是否安装了必要的工具
if ! command -v rsvg-convert &> /dev/null; then
    echo "请先安装 rsvg-convert: brew install librsvg"
    exit 1
fi

# 转换所有 SVG 文件为 PDF
find source -name "*.svg" -type f | while read svg_file; do
    pdf_file="${svg_file%.svg}.pdf"
    rsvg-convert -f pdf -o "$pdf_file" "$svg_file"
done

> source/Cheatsheet-All.md
for file in source/Cheatsheet-0*.md; do
  cat "$file" >> source/Cheatsheet-All.md
  echo -e "\n\n" >> source/Cheatsheet-All.md
done

rm -f cheatsheet.tex cheatsheet.pdf

# 生成tex文件
pandoc "source/Cheatsheet-All.md" \
    --from markdown-simple_tables-multiline_tables-pipe_tables \
    -o cheatsheet.tex \
    --variable=documentclass:extarticle \
    --variable=classoption:8pt \
    --resource-path=. \
    --resource-path=source \
    --resource-path=source/*.assets \
    --variable=graphics \
    --variable=graphics-extension=.pdf \
    --variable=graphics-path=source/*.assets \
    -H template/preamble.tex \
    -B template/before_body.tex \
    -A template/after_body.tex

perl -0777 -pi -e 's/\\begin\{figure\}\s*\\centering\s*(\\includegraphics\{.*?\})\s*\\caption\{.*?\}.*?\s*\\end\{figure\}/$1/gs' cheatsheet.tex


# 直接从tex文件编译PDF (使用xelatex以支持中文)
xelatex -interaction=nonstopmode cheatsheet.tex >> /dev/null
xelatex -interaction=nonstopmode cheatsheet.tex >> /dev/null  # 运行两次以确保引用和目录正确

# 清理临时文件
rm -f cheatsheet.aux cheatsheet.log cheatsheet.out source/Cheatsheet-All.md