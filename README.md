# RL-tp
# DQN LunarLander-v3

Implementação do algoritmo Deep Q-Network (DQN) usando PyTorch e Gymnasium, estruturado em módulos para treinamento e avaliação.

---

## 🔧 1. Criar Ambiente Virtual

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate

### Instalar dependências
```bash
pip install -r requirements.txt

### Rodar o Treinamento
```bash
python main_train.py

O modelo treinado será salvo em:
models/dqn_weights.pth

### Rodar a Avaliação
```bash
python main_evaluate.py


