import requests
import os

os.makedirs("data", exist_ok=True)

url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
output_path = "data/train-v1.1.json"

response = requests.get(url)
if response.status_code == 200:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("✅", output_path)
else:
    print("❌")
