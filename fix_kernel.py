import json

with open('EDA_Planning_Activity2.ipynb', 'r') as f:
    nb = json.load(f)

# Set kernel to python314 (the one with all packages)
nb['metadata']['kernelspec'] = {
    "display_name": "Python 3.14 (IT325)",
    "language": "python",
    "name": "python314"
}

with open('EDA_Planning_Activity2.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Kernel updated to Python 3.14 (IT325) successfully!")
