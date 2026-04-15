"""Módulo de processamento de dados: ETL e Feature Engineering.

Responsável pela ingestão, limpeza, transformação e exportação do dataset
IBM HR Analytics para uso nos modelos preditivos de churn.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes do Dataset
# ---------------------------------------------------------------------------

TARGET_COL = "Attrition"

# Colunas sem variância ou identificadores (serão removidas na limpeza)
COLS_TO_DROP = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Variáveis categóricas nominais → One-Hot Encoding
NOMINAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Variáveis ordinais (já numéricas no dataset original, mantidas como estão)
ORDINAL_COLS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]

# Features criadas no feature engineering
ENGINEERED_FEATURES = [
    "SalaryPerAge",
    "YearsAtCompanyRatio",
    "SatisfactionScore",
    "IsNewEmployee",
]

# Caminhos padrão
DEFAULT_RAW_PATH = Path("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
DEFAULT_PROCESSED_PATH = Path("data/processed/hr_processed.csv")


# ---------------------------------------------------------------------------
# Funções de ETL
# ---------------------------------------------------------------------------


def load_raw_data(filepath: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Carrega o CSV original do dataset IBM HR.

    Args:
        filepath: Caminho para o arquivo CSV.

    Returns:
        DataFrame com os dados brutos.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{path}'. "
            "Baixe o CSV do Kaggle e salve em data/raw/."
        )

    df = pd.read_csv(path)
    logger.info("Dataset carregado: %d registros, %d colunas", df.shape[0], df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza do dataset.

    Operações realizadas:
    - Remove colunas sem variância (EmployeeCount, Over18, StandardHours)
    - Remove coluna de ID (EmployeeNumber)
    - Converte target Attrition para binário (1 = Yes, 0 = No)
    - Verifica e trata valores nulos e duplicatas

    Args:
        df: DataFrame com dados brutos.

    Returns:
        DataFrame limpo.
    """
    df = df.copy()

    # Remover colunas sem variância e IDs
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    logger.info("Colunas removidas: %s", cols_present)

    # Converter target para binário
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        logger.info(
            "Target '%s' convertido: Distribuição: %s",
            TARGET_COL,
            df[TARGET_COL].value_counts().to_dict(),
        )

    # Verificar valores nulos
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        logger.warning("Encontrados %d valores nulos. Removendo linhas afetadas.", null_count)
        df = df.dropna()
    else:
        logger.info("Nenhum valor nulo encontrado.")

    # Remover duplicatas
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        logger.warning("Removidas %d linhas duplicadas.", n_dupes)
        df = df.drop_duplicates()

    logger.info("Dataset após limpeza: %d registros, %d colunas", df.shape[0], df.shape[1])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: cria novas variáveis derivadas.

    Features criadas:
    - SalaryPerAge: MonthlyIncome / Age
    - YearsAtCompanyRatio: YearsAtCompany / TotalWorkingYears (0 se divisão por zero)
    - SatisfactionScore: média de Environment, Job e Relationship Satisfaction
    - IsNewEmployee: 1 se YearsAtCompany <= 1, senão 0

    Args:
        df: DataFrame limpo.

    Returns:
        DataFrame com novas features.
    """
    df = df.copy()

    df["SalaryPerAge"] = df["MonthlyIncome"] / df["Age"]

    df["YearsAtCompanyRatio"] = np.where(
        df["TotalWorkingYears"] > 0,
        df["YearsAtCompany"] / df["TotalWorkingYears"],
        0.0,
    )

    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
    ]
    df["SatisfactionScore"] = df[satisfaction_cols].mean(axis=1)

    df["IsNewEmployee"] = (df["YearsAtCompany"] <= 1).astype(int)

    logger.info("Features criadas: %s", ENGINEERED_FEATURES)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding de variáveis categóricas.

    - Variáveis ordinais já estão codificadas numericamente (mantidas como estão)
    - Variáveis nominais recebem One-Hot Encoding com drop_first=True
      para evitar multicolinearidade

    Args:
        df: DataFrame com features engineered.

    Returns:
        DataFrame com variáveis categóricas codificadas.
    """
    df = df.copy()

    nominal_present = [c for c in NOMINAL_COLS if c in df.columns]

    df = pd.get_dummies(df, columns=nominal_present, drop_first=True, dtype=int)

    logger.info(
        "One-Hot Encoding aplicado em %d colunas: %s: Total de features: %d",
        len(nominal_present),
        nominal_present,
        df.shape[1],
    )
    return df


def save_processed_data(
    df: pd.DataFrame,
    filepath: str | Path = DEFAULT_PROCESSED_PATH,
) -> None:
    """Salva o dataset processado em CSV.

    Args:
        df: DataFrame processado.
        filepath: Caminho para salvar o arquivo.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Dataset processado salvo em '%s' (%d registros, %d colunas)",
        path,
        df.shape[0],
        df.shape[1],
    )


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------


def run_pipeline(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    processed_path: str | Path = DEFAULT_PROCESSED_PATH,
) -> pd.DataFrame:
    """Executa o pipeline ETL completo.

    Etapas: Ingestão → Limpeza → Feature Engineering → Encoding → Exportação.

    Args:
        raw_path: Caminho do CSV bruto.
        processed_path: Caminho para salvar o CSV processado.

    Returns:
        DataFrame processado.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=" * 60)
    logger.info("INÍCIO DO PIPELINE ETL")
    logger.info("=" * 60)

    df = load_raw_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)
    save_processed_data(df, processed_path)

    # Resumo final
    target_dist = df[TARGET_COL].value_counts()
    churn_rate = target_dist.get(1, 0) / len(df) * 100

    logger.info("=" * 60)
    logger.info("PIPELINE ETL CONCLUÍDO")
    logger.info("  Registros: %d", df.shape[0])
    logger.info("  Features:  %d", df.shape[1] - 1)
    logger.info("  Taxa de churn: %.1f%%", churn_rate)
    logger.info("  Distribuição target: %s", target_dist.to_dict())
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run_pipeline()
