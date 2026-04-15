"""Funções auxiliares e de visualização reutilizáveis.

Fornece gráficos Plotly interativos usados tanto nos notebooks de EDA
quanto no dashboard Streamlit.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_processing import TARGET_COL

# ---------------------------------------------------------------------------
# Paleta de cores padrão
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#636EFA",
    "secondary": "#EF553B",
    "churn_no": "#636EFA",
    "churn_yes": "#EF553B",
    "risk_low": "#2ecc71",
    "risk_medium": "#f39c12",
    "risk_high": "#e74c3c",
}

ATTRITION_LABELS = {0: "Não saiu", 1: "Saiu"}
ATTRITION_COLOR_MAP = {"Não saiu": COLORS["churn_no"], "Saiu": COLORS["churn_yes"]}


def _default_layout(fig: go.Figure, title: str) -> go.Figure:
    """Aplica layout padrão consistente aos gráficos."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white",
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Gráficos de EDA
# ---------------------------------------------------------------------------


def plot_churn_rate_by(df: pd.DataFrame, column: str) -> go.Figure:
    """Gráfico de barras da taxa de churn por uma variável categórica.

    Args:
        df: DataFrame com a coluna target (Attrition) em formato binário.
        column: Nome da coluna categórica para agrupar.

    Returns:
        Figura Plotly com taxa de churn (%) por categoria.
    """
    rates = (
        df.groupby(column)[TARGET_COL]
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
        .reset_index()
    )
    rates.columns = [column, "Taxa de Churn (%)"]

    fig = px.bar(
        rates,
        x=column,
        y="Taxa de Churn (%)",
        text="Taxa de Churn (%)",
        color="Taxa de Churn (%)",
        color_continuous_scale=["#636EFA", "#EF553B"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    return _default_layout(fig, f"Taxa de Churn por {column}")


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    hue: str = TARGET_COL,
    kind: str = "box",
) -> go.Figure:
    """Distribuição de uma variável numérica, separada por hue (Attrition).

    Args:
        df: DataFrame com os dados.
        column: Variável numérica a visualizar.
        hue: Coluna categórica para separar (default: Attrition).
        kind: Tipo de gráfico: 'box' ou 'histogram'.

    Returns:
        Figura Plotly.
    """
    plot_df = df.copy()
    plot_df[hue] = plot_df[hue].map(ATTRITION_LABELS)

    if kind == "box":
        fig = px.box(
            plot_df,
            x=hue,
            y=column,
            color=hue,
            color_discrete_map=ATTRITION_COLOR_MAP,
        )
    else:
        fig = px.histogram(
            plot_df,
            x=column,
            color=hue,
            barmode="overlay",
            opacity=0.7,
            color_discrete_map=ATTRITION_COLOR_MAP,
        )

    return _default_layout(fig, f"Distribuição de {column} por Attrition")


def plot_target_distribution(df: pd.DataFrame) -> go.Figure:
    """Gráfico de rosca (donut) da distribuição do target Attrition.

    Args:
        df: DataFrame com a coluna target binária.

    Returns:
        Figura Plotly donut chart.
    """
    counts = df[TARGET_COL].value_counts().reset_index()
    counts.columns = [TARGET_COL, "Contagem"]
    counts[TARGET_COL] = counts[TARGET_COL].map(ATTRITION_LABELS)

    fig = go.Figure(
        go.Pie(
            labels=counts[TARGET_COL],
            values=counts["Contagem"],
            hole=0.45,
            marker=dict(colors=[COLORS["churn_no"], COLORS["churn_yes"]]),
            textinfo="label+percent+value",
        )
    )
    return _default_layout(fig, "Distribuição de Attrition")


def plot_correlation_heatmap(
    df: pd.DataFrame,
    max_features: int = 20,
) -> go.Figure:
    """Heatmap de correlação interativo com Plotly.

    Seleciona as features mais correlacionadas com o target para manter
    a legibilidade do gráfico.

    Args:
        df: DataFrame com variáveis numéricas.
        max_features: Número máximo de features a exibir.

    Returns:
        Figura Plotly heatmap.
    """
    numeric_df = df.select_dtypes(include="number")

    # Selecionar as top features mais correlacionadas com o target
    if TARGET_COL in numeric_df.columns and len(numeric_df.columns) > max_features:
        corr_with_target = numeric_df.corr()[TARGET_COL].abs().sort_values(ascending=False)
        top_cols = corr_with_target.head(max_features).index.tolist()
        numeric_df = numeric_df[top_cols]

    corr = numeric_df.corr().round(2)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=9),
        )
    )
    fig.update_layout(
        width=800,
        height=700,
        xaxis=dict(tickangle=45),
    )
    return _default_layout(fig, "Heatmap de Correlação")


def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    top_n: int = 15,
) -> go.Figure:
    """Gráfico de barras horizontal da importância das features.

    Args:
        feature_names: Nomes das features.
        importances: Valores de importância correspondentes.
        top_n: Quantidade de features a exibir.

    Returns:
        Figura Plotly.
    """
    imp_df = (
        pd.DataFrame({"Feature": feature_names, "Importância": importances})
        .sort_values("Importância", ascending=True)
        .tail(top_n)
    )

    fig = px.bar(
        imp_df,
        x="Importância",
        y="Feature",
        orientation="h",
        color="Importância",
        color_continuous_scale="Blues",
    )
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=max(400, top_n * 28))
    return _default_layout(fig, f"Top {top_n} Features Mais Importantes")


def get_top_correlations(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    top_n: int = 10,
) -> pd.DataFrame:
    """Retorna as top N features mais correlacionadas com o target.

    Args:
        df: DataFrame numérico.
        target: Coluna target.
        top_n: Quantidade de features.

    Returns:
        DataFrame com colunas ['Feature', 'Correlação'] ordenado por valor absoluto.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    result = corr.head(top_n).reset_index()
    result.columns = ["Feature", "Correlação"]
    return result
