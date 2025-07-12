import pandas as pd
import os
from tqdm import tqdm

# Caminho para os dados existentes
CSV_PATH = './melanoma-detector/archive/train.csv'  # CSV original com image_name e target
IMAGE_FOLDER = './melanoma-detector/archive/train/512x512-dataset-melanoma/512x512-dataset-melanoma'         # Pasta onde já estão as imagens redimensionadas
OUTPUT_CSV = './melanoma-detector/archive/marking.csv' #cópia gerada pelo train.csv

# Verifica se a pasta de imagens existe
if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Pasta de imagens não encontrada: {IMAGE_FOLDER}")
    exit(1)

# Carregar CSV simples: image_name,target
df_isic2020 = pd.read_csv(CSV_PATH)

# Inicializar novo dataset
dataset = {
    'image_id': [],
    'target': [],
}

for _, row in tqdm(df_isic2020.iterrows(), total=df_isic2020.shape[0]):
    image_id = row['image_name']
    target = row['target']
    image_path = os.path.join(IMAGE_FOLDER, f'{image_id}.jpg')

    if os.path.exists(image_path):
        dataset['image_id'].append(image_id)
        dataset['target'].append(target)
    else:
        print(f'⚠️ Imagem não encontrada e será ignorada: {image_path}')

# Salvar novo marking.csv
df_all = pd.DataFrame(dataset).set_index('image_id')
df_all.to_csv(OUTPUT_CSV)
print("✅ Arquivo marking.csv salvo com sucesso!")
