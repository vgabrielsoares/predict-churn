# 📊 Previsão de Churn de Funcionários

Modelo preditivo de rotatividade de funcionários usando Machine Learning, com dashboard interativo Streamlit para análise exploratória e simulação de risco individual em tempo real.

---

## O Problema

A rotatividade de funcionários (churn) é um dos maiores custos ocultos das empresas. Estima-se que substituir um funcionário custe entre **6 a 9 meses de salário** (SHRM). Em empresas de consultoria, onde o capital intelectual é o principal ativo, reter talentos é ainda mais crítico.

## A Solução

Pipeline completo de ciência de dados que permite **prever quais funcionários têm maior probabilidade de deixar a empresa**, usando dados de RH como satisfação, salário, horas extras, tempo de empresa e avaliações: possibilitando ações preventivas pelo RH.

```text
CSV (IBM HR) → ETL → Feature Engineering → EDA → Modelagem ML → Dashboard Streamlit
                                                                        ↓
                                                              Predição em tempo real
```

---

## Resultados Obtidos

### Top 5 Fatores que Mais Influenciam o Churn

1. **StockOptionLevel**: Nível de opções de ações
2. **Department_Sales**: Trabalhar no departamento de Vendas
3. **JobLevel**: Nível hierárquico do cargo
4. **MaritalStatus_Married**: Estado civil casado
5. **JobSatisfaction**: Satisfação no trabalho

### Insights Principais

- Funcionários com **hora extra frequente** têm taxa de churn significativamente maior
- **Salários mais baixos** estão diretamente correlacionados com maior rotatividade
- **Representantes de vendas** e **técnicos de laboratório** apresentam as maiores taxas de churn
- Funcionários **novos** (≤ 1 ano) têm risco elevado de saída
- **Satisfação no trabalho** e **equilíbrio vida-trabalho** são fatores de proteção

---

## Dataset

- **Nome**: IBM HR Analytics Employee Attrition & Performance
- **Fonte**: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Registros**: 1.470 funcionários
- **Features**: 35 variáveis originais → 48 após feature engineering e encoding
- **Target**: `Attrition` (binário): desbalanceado (~16% churn)
- **Licença**: Público, sem restrições de uso

---

## Como Rodar

### Pré-requisitos

- Python 3.10+
- pip

### 1. Clonar o repositório

```bash
git clone https://github.com/vgabrielsoares/predict-churn.git
cd predict-churn
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Baixar o dataset

Baixe o CSV do [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) e salve em:

```text
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

### 4. Executar o pipeline de processamento

```python
from src.data_processing import run_pipeline
run_pipeline()
```

Isso gera o dataset processado em `data/processed/hr_processed.csv`.

### 5. Treinar o modelo (opcional)

O notebook `notebooks/02_modeling.ipynb` treina e salva o melhor modelo em `models/best_model.joblib`. Se o modelo já existe, o dashboard o carrega automaticamente.

### 6. Iniciar o Dashboard

```bash
python -m streamlit run app/app.py
```

Acesse em **<http://localhost:8501>**.

---

## Dashboard

O dashboard Streamlit possui **4 páginas**:

| Página                      | Descrição                                                                                                  |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **🏠 Visão Geral**          | KPIs principais (total de funcionários, taxa de churn, salário médio, satisfação) e distribuição do target |
| **📈 Análise Exploratória** | Gráficos interativos Plotly com filtros por departamento, cargo e hora extra                               |
| **🤖 Modelo Preditivo**     | Métricas do XGBoost, matriz de confusão, importância das features e comparação entre modelos               |
| **🎯 Simulador de Churn**   | Formulário completo para simular o risco de churn individual com gauge chart e top fatores de risco        |

---

## Estrutura do Repositório

```text
predict-churn/
├── data/
│   ├── raw/                      # Dataset original (não versionado)
│   └── processed/                # Dataset processado (não versionado)
├── notebooks/
│   ├── 01_eda.ipynb              # Análise exploratória
│   └── 02_modeling.ipynb         # Experimentação de modelos
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # ETL e feature engineering
│   ├── model.py                  # Treinamento, avaliação e predição
│   └── utils.py                  # Funções de visualização reutilizáveis
├── app/
│   └── app.py                    # Dashboard Streamlit (4 páginas)
├── models/
│   └── best_model.joblib         # Modelo serializado (não versionado)
├── README.md                     # Este arquivo
├── requirements.txt              # Dependências Python
└── .gitignore
```

## Tecnologias

| Tecnologia               | Uso                                   |
| ------------------------ | ------------------------------------- |
| **Python 3.10+**         | Linguagem principal                   |
| **Pandas / NumPy**       | Manipulação e transformação de dados  |
| **Scikit-learn**         | Modelos ML, métricas, preprocessing   |
| **XGBoost**              | Modelo avançado de gradient boosting  |
| **imbalanced-learn**     | SMOTE para balanceamento de classes   |
| **Plotly**               | Gráficos interativos no dashboard     |
| **Seaborn / Matplotlib** | Visualizações estáticas nos notebooks |
| **Streamlit**            | Dashboard web interativo              |
| **Joblib**               | Serialização de modelos treinados     |

---

## Licença

Dataset IBM HR Analytics é de uso público.
