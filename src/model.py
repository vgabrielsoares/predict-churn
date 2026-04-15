"""Módulo de modelagem: Treinamento, avaliação e predição de churn.

Fornece funções para preparar dados, treinar modelos (Logistic Regression,
Random Forest, XGBoost), avaliar performance e realizar predições individuais.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from src.data_processing import DEFAULT_PROCESSED_PATH, TARGET_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_MODEL_PATH = Path("models/best_model.joblib")

# Registro de modelos suportados: factory pattern
_MODEL_REGISTRY: dict[str, type] = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}

# Hiperparâmetros default por tipo de modelo
_DEFAULT_PARAMS: dict[str, dict] = {
    "logistic": {
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "n_jobs": -1,
    },
}

# Grid de busca para otimização de hiperparâmetros
_SEARCH_PARAMS: dict[str, dict] = {
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    },
}


# ---------------------------------------------------------------------------
# Preparação de dados
# ---------------------------------------------------------------------------


def prepare_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Separa features e target, faz split treino/teste estratificado.

    Args:
        df: DataFrame processado com target binário.
        target_col: Nome da coluna target.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Split realizado: Treino: %d | Teste: %d | Features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica SMOTE para balancear as classes no conjunto de treino.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        Tupla (X_resampled, y_resampled) com classes balanceadas.
    """
    dist_before = y_train.value_counts().to_dict()

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE aplicado: Antes: %s | Depois: {0: %d, 1: %d}",
        dist_before, (y_res == 0).sum(), (y_res == 1).sum(),
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------


def _get_scale_pos_weight(y_train: pd.Series) -> float:
    """Calcula scale_pos_weight para XGBoost baseado no desbalanceamento."""
    counts = y_train.value_counts()
    return counts[0] / counts[1]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    **extra_params,
) -> object:
    """Treina um modelo de classificação.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        model_type: Tipo do modelo: 'logistic', 'random_forest' ou 'xgboost'.
        **extra_params: Parâmetros adicionais para sobrescrever os defaults.

    Returns:
        Modelo treinado.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_type}' não suportado. "
            f"Opções: {list(_MODEL_REGISTRY.keys())}"
        )

    params = _DEFAULT_PARAMS[model_type].copy()

    # Ajustar scale_pos_weight para XGBoost
    if model_type == "xgboost" and "scale_pos_weight" not in extra_params:
        params["scale_pos_weight"] = _get_scale_pos_weight(y_train)

    params.update(extra_params)

    model = _MODEL_REGISTRY[model_type](**params)
    model.fit(X_train, y_train)

    logger.info("Modelo '%s' treinado com parâmetros: %s", model_type, params)
    return model


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Avalia o modelo e retorna um dicionário completo de métricas.

    Args:
        model: Modelo treinado com .predict() e .predict_proba().
        X_test: Features de teste.
        y_test: Target de teste.

    Returns:
        Dict com accuracy, precision, recall, f1, auc_roc,
        confusion_matrix, classification_report e feature_importances.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    # Feature importance (disponível em tree-based e logistic)
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = model.feature_importances_
    elif hasattr(model, "coef_"):
        metrics["feature_importances"] = np.abs(model.coef_[0])

    logger.info(
        "Avaliação: Acc: %.3f | Prec: %.3f | Rec: %.3f | F1: %.3f | AUC: %.3f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc_roc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Otimização de hiperparâmetros
# ---------------------------------------------------------------------------


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "xgboost",
    n_iter: int = 20,
    cv: int = 5,
) -> object:
    """Executa RandomizedSearchCV e retorna o melhor estimador.

    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        model_type: Tipo do modelo.
        n_iter: Número de combinações a testar.
        cv: Número de folds de cross-validation.

    Returns:
        Melhor modelo encontrado pelo search.
    """
    if model_type not in _SEARCH_PARAMS:
        raise ValueError(
            f"Otimização não configurada para '{model_type}'. "
            f"Opções: {list(_SEARCH_PARAMS.keys())}"
        )

    base_params = _DEFAULT_PARAMS[model_type].copy()
    if model_type == "xgboost":
        base_params["scale_pos_weight"] = _get_scale_pos_weight(y_train)

    base_model = _MODEL_REGISTRY[model_type](**base_params)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=_SEARCH_PARAMS[model_type],
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    logger.info(
        "Otimização '%s': Melhor AUC-ROC CV: %.4f | Params: %s",
        model_type,
        search.best_score_,
        search.best_params_,
    )
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Serialização
# ---------------------------------------------------------------------------


def save_model(model, filepath: str | Path = DEFAULT_MODEL_PATH) -> None:
    """Serializa o modelo com joblib.

    Args:
        model: Modelo treinado.
        filepath: Caminho para salvar.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modelo salvo em '%s'", path)


def load_model(filepath: str | Path = DEFAULT_MODEL_PATH) -> object:
    """Carrega modelo serializado.

    Args:
        filepath: Caminho do arquivo .joblib.

    Returns:
        Modelo carregado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em '{path}'.")
    model = joblib.load(path)
    logger.info("Modelo carregado de '%s'", path)
    return model


# ---------------------------------------------------------------------------
# Predição individual
# ---------------------------------------------------------------------------


def predict_churn(
    model,
    employee_data: dict,
    feature_names: list[str],
) -> dict:
    """Recebe dados de um funcionário e retorna a predição de churn.

    Args:
        model: Modelo treinado.
        employee_data: Dict com os valores das features do funcionário.
        feature_names: Lista com os nomes das features na ordem esperada pelo modelo.

    Returns:
        Dict com:
        - probability: float (0-1)
        - risk_label: str ('Alto Risco' / 'Médio Risco' / 'Baixo Risco')
        - risk_color: str (cor hex correspondente)
        - top_factors: list dos fatores que mais pesaram na predição.
    """
    # Montar DataFrame com uma linha, preenchendo features ausentes com 0
    input_df = pd.DataFrame([{f: employee_data.get(f, 0) for f in feature_names}])

    probability = model.predict_proba(input_df)[0, 1]

    # Classificação de risco
    if probability < 0.3:
        risk_label, risk_color = "Baixo Risco", "#2ecc71"
    elif probability < 0.6:
        risk_label, risk_color = "Médio Risco", "#f39c12"
    else:
        risk_label, risk_color = "Alto Risco", "#e74c3c"

    # Top fatores de risco baseado em feature importance
    top_factors = _get_top_factors(model, feature_names, input_df)

    return {
        "probability": float(probability),
        "risk_label": risk_label,
        "risk_color": risk_color,
        "top_factors": top_factors,
    }


def _get_top_factors(
    model,
    feature_names: list[str],
    input_df: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    """Extrai os fatores que mais contribuem para a predição de churn.

    Usa feature importance global do modelo ponderada pelos valores do input.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return []

    # Ponderar importância pelo valor absoluto do input
    weighted = importances * np.abs(input_df.values[0])
    top_indices = np.argsort(weighted)[::-1][:top_n]
    return [feature_names[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Utilitário: comparação de modelos
# ---------------------------------------------------------------------------


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Cria tabela comparativa das métricas de vários modelos.

    Args:
        results: Dict com nome do modelo como chave e dict de métricas como valor
                 (output de evaluate_model).

    Returns:
        DataFrame formatado com as métricas de cada modelo.
    """
    summary_cols = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    rows = {
        name: {col: metrics[col] for col in summary_cols}
        for name, metrics in results.items()
    }
    df = pd.DataFrame(rows).T.round(4)
    df.index.name = "Modelo"
    return df.sort_values("auc_roc", ascending=False)
