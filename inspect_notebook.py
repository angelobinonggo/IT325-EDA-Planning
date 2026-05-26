import json
import sys

with open('EDA_Planning_Activity2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('notebook_dump.txt', 'w', encoding='utf-8') as out_f:
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        out_f.write("=" * 60 + "\n")
        out_f.write(f"CELL {i+1} - {cell_type.upper()}\n")
        out_f.write("=" * 60 + "\n")
        
        source = "".join(cell.get('source', []))
        if cell_type == 'markdown':
            out_f.write(source + "\n")
        elif cell_type == 'code':
            out_f.write("CODE:\n" + source + "\n")
            outputs = cell.get('outputs', [])
            if outputs:
                out_f.write("\nOUTPUTS:\n")
                for out in outputs:
                    if out.get('output_type') == 'stream':
                        out_f.write("".join(out.get('text', []))[:1000] + "\n")
                    elif out.get('output_type') == 'execute_result':
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            out_f.write("".join(data['text/plain'])[:1000] + "\n")
                    elif out.get('output_type') == 'error':
                        out_f.write(f"ERROR: {out.get('ename')} - {out.get('evalue')}\n")
        out_f.write("\n\n")
