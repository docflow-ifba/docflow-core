import re
import logging
from typing import List, Tuple
from collections import Counter

logger = logging.getLogger("file-utils")

IGNORED_PATTERNS = [
    re.compile(pattern) for pattern in [
        r'<!-- image -->',
        r'MINISTÉRIO DA EDUCAÇÃO',
        r'PÁGINA \d+',
        r'^\d{1,3}\s*$',
        r'^##?\s*(Gabinete|Reitoria|Pró-Reitoria|Coordenação|Universidade|Campus)\b.*',
        r'(?i)(universidade federal|instituto federal|governo federal|república federativa do brasil)',
        r'^\d{1,2}/\d{1,2}/\d{4}$',
        r'^\f$',
        r'(?i)^.*\bunknown\b.*$'
    ]
]

def _clean_markdown_lines(lines: List[str]) -> List[str]:
    logger.debug("Limpando linhas do Markdown...")
    cleaned_lines = []
    blank_line = False

    for line in lines:
        if any(pattern.search(line) for pattern in IGNORED_PATTERNS):
            continue

        if line.strip() == "":
            if not blank_line:
                cleaned_lines.append("\n")
                blank_line = True
        else:
            cleaned_lines.append(line)
            blank_line = False

    logger.debug(f"{len(cleaned_lines)} linhas restantes após limpeza")
    return cleaned_lines

def _remove_lines_repeated_more_than_n(lines: List[str], n: int = 3) -> List[str]:
    line_counts = Counter(lines)
    return [line for line in lines if line_counts[line] <= n]

def process_markdown(content: str) -> Tuple[str, List[str]]:
    logger.info("Processando conteúdo Markdown...")
    raw_lines = content.splitlines(keepends=True)
    
    cleaned_lines = _clean_markdown_lines(raw_lines)
    filtered_lines = _remove_lines_repeated_more_than_n(cleaned_lines, n=3)
    clean_md = "".join(filtered_lines)

    logger.info("Markdown processado")
    return clean_md