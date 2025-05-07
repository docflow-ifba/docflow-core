import re, os

def extract_tables_and_replace(md_file, tables_output):
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r'(\n\|.+?\|\n(?:\|.+?\|\n)+)')
    tables = pattern.findall(content)

    tables_md = "# Tabelas Extras\n\n"
    for i, t in enumerate(tables, 1):
        tables_md += f"## Tabela {i}\n\n{t}\n\n"

    with open(tables_output, "w", encoding="utf-8") as f:
        f.write(tables_md)

    for i, t in enumerate(tables, 1):
        ref = f"\n[Ver Tabela {i} no arquivo de tabelas]({os.path.basename(tables_output)}#tabela-{i})\n"
        content = content.replace(t, ref, 1)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(content)

def process_markdown(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed = []
    blank_line = False

    for line in lines:
        if "<!-- image -->" in line or "MINISTÉRIO DA EDUCAÇÃO" in line or re.search(r'PÁGINA \d+', line):
            continue

        if line.strip() == "":
            if not blank_line:
                processed.append("\n")
                blank_line = True
        else:
            processed.append(line)
            blank_line = False

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(processed)

    extract_tables_and_replace(output_file, os.path.join(os.path.dirname(output_file), "tables.md"))
