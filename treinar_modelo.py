import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
CSV_PATH = './melanoma-detector/archive/train.csv'
IMAGE_DIR = './melanoma-detector/archive/train/512x512-dataset-melanoma/512x512-dataset-melanoma'
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ler CSV
df = pd.read_csv(CSV_PATH)

# Detectar extensão das imagens
arquivos = os.listdir(IMAGE_DIR)
ext = '.jpg' if any(f.endswith('.jpg') for f in arquivos) else '.png'

# Criar caminho completo das imagens
df['image_path'] = df['image_name'].apply(lambda x: os.path.join(IMAGE_DIR, f'{x}{ext}'))
df['exists'] = df['image_path'].apply(os.path.exists)
df = df[df['exists']].reset_index(drop=True)

# Dividir treino e validação
df_treino, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])

# Dataset customizado
class MelanomaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = torch.tensor(row['target'], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label.unsqueeze(0)

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# DataLoaders
train_loader = DataLoader(MelanomaDataset(df_treino, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MelanomaDataset(df_val, transform), batch_size=BATCH_SIZE)

# Modelo simples
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(224*224*3, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
).to(DEVICE)

# Critério e otimizador
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Treinamento
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validação
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())

    bin_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, bin_preds)
    print(f"\n📊 Época {epoch+1}/{EPOCHS} | Loss treino: {total_loss:.4f} | Val Accuracy: {acc:.4f}")

# Salvar modelo
torch.save(model.state_dict(), 'modelo_melanoma.pth')
print("\n✅ Modelo salvo como modelo_melanoma.pth")

# Matriz de confusão
cm = confusion_matrix(all_labels, bin_preds)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

#Curva ROC
fpr, tpr, _ = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Curva ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Época 5/5 | Loss treino: 420.8299 | Val Accuracy: 0.9102