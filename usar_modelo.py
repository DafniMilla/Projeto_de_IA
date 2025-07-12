import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Configuração do dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Caminho base do projeto (onde está o script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho relativo do modelo e da imagem de teste
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_melanoma.pth')

IMAGE_PATH = os.path.join(
    BASE_DIR,
    'melanoma-detector',
    'archive',
    'train',
    '512x512-dataset-melanoma',
    '512x512-dataset-melanoma',
    'ISIC_0000001.jpg'  # <- Altere esse nome se quiser testar outra imagem, detecção de melanoma sem interface, resultado no terminal
)

# Transformações (mesmas do treino)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Estrutura do modelo (igual ao do treinamento)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(224*224*3, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
).to(DEVICE)

# Carregar pesos treinados
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Carregar imagem de teste
try:
    image = Image.open(IMAGE_PATH).convert('RGB')
except FileNotFoundError:
    raise RuntimeError(f"🚫 Imagem não encontrada no caminho: {IMAGE_PATH}")

image = transform(image).unsqueeze(0).to(DEVICE)  

# Fazer predição
with torch.no_grad():
    output = model(image)
    prob = output.item()

# Mostrar resultado
print(f"\n🧪 Probabilidade de melanoma: {prob:.4f}")
if prob >= 0.5:
    print("⚠️ Resultado: **MELANOMA provável**")
else:
    print("✅ Resultado: **Lesão benigna provável**")
