# 🧠 Detecção de Melanoma com Inteligência Artificial

Projeto de IA para classificar imagens de lesões de pele em melanoma ou não melanoma, auxiliando no diagnóstico precoce do câncer de pele.

---
## 🛠 Tecnologias
Python 

PyTorch 

Pandas

NumPy


---

## 📂 Estrutura do projeto

melanoma-detector/

├── download_dados.py            # Script para download dos dados

├── interface.py                 # Interface principal para uso do modelo

├── marking.csv                  # Arquivo com marcações/labels das imagens

├── modelo_melanoma.pth           # Modelo treinado (PyTorch)

├── requirements.txt              # Dependências do projeto

├── train.csv                     # Dataset de treino

├── treinar_modelo.py             # Script para treinar o modelo

└── usar_modelo.py                # Script para utilizar o modelo treinado

---

## 🚀 Como usar

1. Clone o repositório:
   ```https://github.com/DafniMilla/melanoma-detector.git```
   
3. Acesse a pasta:   ```cd melanoma-detector```
   
3. Instale as dependências:
```pip install -r requirements.txt```

5. Treine o modelo:
   ````python treinar_modelo.py````

6. Execute a interface gráfica para testar imagens:
   ````python interface.py````

## ⚠️ Aviso

Projeto para fins educacionais, não substitui avaliação médica profissional.

