import json

with open('EDA_Planning_Activity2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
print()

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        outputs = cell.get('outputs', [])
        has_error = any(o.get('output_type') == 'error' for o in outputs)
        exec_count = cell.get('execution_count')
        src_preview = ''.join(cell['source'])[:100].replace('\n', ' ')
        status = 'ERROR' if has_error else 'OK'
        print(f'  Cell {i+1} | exec={exec_count} | [{status}] | {src_preview}')
        if has_error:
            for o in outputs:
                if o.get('output_type') == 'error':
                    print(f'    --> {o.get("ename")}: {o.get("evalue")}')
                    for tb in o.get('traceback', [])[-3:]:
                        clean = tb.replace('\x1b[0;31m','').replace('\x1b[0m','').replace('\x1b[1;32m','').replace('\x1b[0;32m','').replace('\x1b[1;31m','')
                        print(f'    {clean}')
