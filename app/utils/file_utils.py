import re
import logging
from typing import List, Tuple

logger = logging.getLogger("file-utils")

TABLE_PATTERN = re.compile(r'(\n\|.+?\|\n(?:\|.+?\|\n)+)')
IGNORED_PATTERNS = [
    r'<!-- image -->',
    r'MINISTÉRIO DA EDUCAÇÃO',
    r'PÁGINA \d+',
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

def create_tables_markdown(tables: List[str]) -> str:
    logger.debug("Criando arquivo Markdown com tabelas extras...")
    md = "# Tabelas Extras\n\n"
    for i, table in enumerate(tables, 1):
        md += f"## Tabela {i}\n\n{table}\n\n"
    return md

def replace_tables_with_references(content: str, tables: List[str]) -> str:
    logger.debug("Substituindo tabelas por referências no conteúdo...")
    for i, table in enumerate(tables, 1):
        reference = f"\n[Ver Tabela {i} no arquivo de tabelas](tables.md#tabela-{i})\n"
        content = content.replace(table, reference, 1)
    return content

def process_markdown(content: str) -> str:
    logger.info("Processando conteúdo Markdown...")
    raw_lines = content.splitlines(keepends=True)
    cleaned_content = "".join(clean_markdown_lines(raw_lines))

    tables = extract_tables(cleaned_content)

    if tables:
        tables_md = create_tables_markdown(tables)
        updated_content = replace_tables_with_references(cleaned_content, tables)
        logger.info("Markdown processado com referências de tabelas")
        return updated_content, tables_md

    logger.info("Markdown processado (sem tabelas)")
    return cleaned_content, ""
