import os
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from tkinter import filedialog, Tk, Label, Button, messagebox, Frame
from torchvision import transforms

# Modelo e dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_melanoma.pth')

# Transformação
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Modelo (igual ao treinado)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(224*224*3, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Fazer inferência
def classificar_imagem(imagem):
    tensor = transform(imagem).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        prob = output.item()

    percentual = round(prob * 100, 2)
    if prob >= 0.5:
        texto = f"⚠️ MELANOMA provável\nProbabilidade: {percentual}%"
        label_resultado.config(fg="#D32F2F")
    else:
        texto = f"✅ Lesão BENIGNA provável\nProbabilidade: {percentual}%"
        label_resultado.config(fg="#388E3C")

    label_resultado.config(text=texto)

# Selecionar imagem
def selecionar_imagem():
    caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if caminho:
        imagem = Image.open(caminho).convert('RGB')
        imagem_tk = ImageTk.PhotoImage(imagem.resize((200, 200)))
        label_img.config(image=imagem_tk)
        label_img.image = imagem_tk
        classificar_imagem(imagem)

# Interface
janela = Tk()
janela.title("🧠 Classificador de Melanoma com IA")
janela.geometry("450x500")
janela.configure(bg="#e8f0fe")

# Frame centralizado com estilo
frame = Frame(janela, bg="white", bd=2, relief="groove")
frame.pack(pady=30, padx=20)

titulo = Label(frame, text="Detecção de Melanoma", font=("Helvetica", 18, "bold"), bg="white", fg="#1a237e")
titulo.pack(pady=20)

btn = Button(
    frame, text="📂 Selecionar Imagem", font=("Helvetica", 12, "bold"),
    bg="#1976D2", fg="white", activebackground="#1565C0", activeforeground="white",
    relief="flat", padx=10, pady=5, command=selecionar_imagem
)
btn.pack(pady=15)

label_img = Label(frame, bg="white")
label_img.pack(pady=10)

label_resultado = Label(frame, text="", font=("Helvetica", 13, "bold"), bg="white")
label_resultado.pack(pady=20)

janela.mainloop()
