
# **Estrutura do Projeto**

```
your_project/
│
├── agent/
│   └── dqn_agent.py            # Implementação completa do agente DQN
│   └── nn_model.py             # Rede neural (policy/value network)
│
├── utils/
│   └── replay_buffer.py        # Implementação do Replay Buffer
│
├── training/
│   └── train.py                # Função que executa o loop de treinamento
│
├── evaluation/
│   └── evaluate.py             # Rotina de avaliação do modelo treinado
│
├── main_train.py               # Script principal para treinamento
├── main_evaluate.py            # Script principal para avaliação
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

---

# **1. Criando o Ambiente Virtual**

O ideal é isolar as dependências em um ambiente virtual.

## **Linux / Mac**

```bash
python3 -m venv venv
source venv/bin/activate
```

## **Windows**

```cmd
python -m venv venv
venv\Scripts\activate
```

---

# **2. Instalando Dependências**

Após ativar o ambiente virtual:

```bash
pip install -r requirements.txt
```

O arquivo contém:

```
numpy
pytorch
matplotlib
swig
gymnasium
gymnasium[box2d]
```

---

# **3. Rodando o Treinamento**

O script **main_train.py** contém os hiperparâmetros e chama o módulo `training/train.py`.

Execute:

```bash
python main_train.py
```

---

# **4. Rodando a Avaliação**

Para avaliar um modelo já treinado:

```bash
python main_evaluate.py
```

---

# **5. Hiperparâmetros Utilizados**

O projeto segue esta configuração (padrão do `main_train.py`):

```python
params = {
    'alpha': 0.00017195082231670288,
    'gamma': 0.9778366856839303,
    'batch_size': 128,
    'buffer_size': 50000,
    'epsilon_decay': 0.9990115359881433,
    'target_update': 500,
    'train_freq': 4,
    'episodes': 2000
}
```


```
main_train.py
```

---

# **6. Como Reproduzir do Zero**

```bash
git clone <este-repo>
cd <projeto>

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 main_train.py
python3 main_evaluate.py
```

---



