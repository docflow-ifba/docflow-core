from collections import Counter
import re
import logging
from typing import List, Tuple

logger = logging.getLogger("file-utils")

TABLE_PATTERN = re.compile(r'(\n\|.+?\|\n(?:\|.+?\|\n)+)')
IGNORED_PATTERNS = [
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

def clean_markdown_lines(lines: List[str]) -> List[str]:
    logger.debug("Limpando linhas do Markdown...")
    cleaned_lines = []
    blank_line = False

    for line in lines:
        if any(re.search(pattern, line) for pattern in IGNORED_PATTERNS):
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

def extract_tables(content: str) -> List[str]:
    tables = TABLE_PATTERN.findall(content)
    logger.info(f"{len(tables)} tabelas encontradas no conteúdo")
    return tables

def replace_tables_with_references(content: str, tables: List[str]) -> str:
    logger.debug("Substituindo tabelas por referências no conteúdo...")
    for i, table in enumerate(tables, 1):
        reference = f"\n[Ver Tabela {i} no arquivo de tabelas](tables.md#tabela-{i})\n"
        content = content.replace(table, reference, 1)
    return content

def process_markdown(content: str):
    logger.info("Processando conteúdo Markdown...")
    raw_lines = content.splitlines(keepends=True)
    
    cleaned_lines = clean_markdown_lines(raw_lines)
    filtered_lines = remove_lines_repeated_more_than_n(cleaned_lines, n=3)
    clean_md = "".join(filtered_lines)

    tables = extract_tables(clean_md)

    if tables:
        updated_content = replace_tables_with_references(clean_md, tables)
        logger.info("Markdown processado com referências de tabelas")
        return updated_content, tables

    logger.info("Markdown processado (sem tabelas)")
    return clean_md, []


def remove_lines_repeated_more_than_n(lines: List[str], n: int = 3) -> List[str]:
    line_counts = Counter(lines)

    filtered_lines = [line for line in lines if line_counts[line] <= n]

    return filtered_lines