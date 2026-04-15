"""Dashboard Streamlit: Previsão de Churn de Funcionários.

Aplicação web interativa para explorar os dados de RH, visualizar
resultados do modelo preditivo e simular o risco de churn individual.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Garantir que o diretório raiz do projeto esteja no path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_processing import (
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_PATH,
    NOMINAL_COLS,
    TARGET_COL,
    clean_data,
    engineer_features,
    load_raw_data,
)
from src.model import (
    DEFAULT_MODEL_PATH,
    apply_smote,
    compare_models,
    evaluate_model,
    load_model,
    predict_churn,
    prepare_data,
    train_model,
)
from src.utils import (
    COLORS,
    plot_churn_rate_by,
    plot_correlation_heatmap,
    plot_distribution,
    plot_feature_importance,
    plot_target_distribution,
)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Previsão de Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cache de dados e modelo
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_processed_data() -> pd.DataFrame:
    """Carrega o dataset processado."""
    return pd.read_csv(DEFAULT_PROCESSED_PATH)


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    """Carrega o dataset bruto e aplica limpeza + feature engineering."""
    df = load_raw_data(DEFAULT_RAW_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    return df


@st.cache_resource(show_spinner=False)
def load_trained_model():
    """Carrega o modelo serializado."""
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_model_metrics() -> dict:
    """Treina e avalia os 3 modelos para obter métricas atualizadas.

    Retorna dict com resultados de cada modelo e a tabela comparativa.
    """
    df = load_processed_data()
    X_train, X_test, y_train, y_test = prepare_data(df, TARGET_COL)
    X_res, y_res = apply_smote(X_train, y_train)

    results = {}
    for model_type in ["logistic", "random_forest", "xgboost"]:
        model = train_model(X_res, y_res, model_type=model_type)
        metrics = evaluate_model(model, X_test, y_test)
        results[model_type] = metrics

    comparison_df = compare_models(results)
    return {
        "results": results,
        "comparison": comparison_df,
        "feature_names": X_test.columns.tolist(),
    }


# ---------------------------------------------------------------------------
# Sidebar: Navegação
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Previsão de Churn")
st.sidebar.markdown("**People Analytics Dashboard**")
st.sidebar.markdown("---")

PAGES = {
    "🏠 Visão Geral": "home",
    "📈 Análise Exploratória": "eda",
    "🤖 Modelo Preditivo": "model",
    "🎯 Simulador de Churn": "simulator",
}

selected_page = st.sidebar.radio("Navegação", list(PAGES.keys()))
page = PAGES[selected_page]


# =========================================================================
# Página 1: Visão Geral (Home)
# =========================================================================


def page_home() -> None:
    """Renderiza a página inicial com KPIs e distribuição do target."""
    st.title("Previsão de Churn de Funcionários")
    st.markdown(
        "Análise preditiva da rotatividade de funcionários usando o dataset "
        "IBM HR Analytics."
    )
    st.markdown("---")

    with st.spinner("Carregando dados..."):
        df_raw = load_raw()

    # KPIs
    total = len(df_raw)
    churn_rate = df_raw[TARGET_COL].mean() * 100
    avg_salary = df_raw["MonthlyIncome"].mean()
    avg_satisfaction = df_raw["JobSatisfaction"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Funcionários", f"{total:,}")
    col2.metric("Taxa de Churn", f"{churn_rate:.1f}%")
    col3.metric("Salário Médio", f"R$ {avg_salary:,.0f}")
    col4.metric("Satisfação Média", f"{avg_satisfaction:.2f} / 4")

    st.markdown("---")

    # Distribuição do target
    st.subheader("Distribuição de Attrition")
    fig = plot_target_distribution(df_raw)
    st.plotly_chart(fig, use_container_width=True)


# =========================================================================
# Página 2: Análise Exploratória
# =========================================================================


def page_eda() -> None:
    """Renderiza a página de análise exploratória com filtros e gráficos."""
    st.title("Análise Exploratória dos Dados")

    with st.spinner("Carregando dados..."):
        df_raw = load_raw()

    # ---- Filtros no sidebar ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros")

    departments = sorted(df_raw["Department"].unique())
    sel_dept = st.sidebar.multiselect(
        "Departamento", departments, default=departments
    )

    job_roles = sorted(df_raw["JobRole"].unique())
    sel_roles = st.sidebar.multiselect("Cargo", job_roles, default=job_roles)

    sel_overtime = st.sidebar.multiselect(
        "Hora Extra", ["Yes", "No"], default=["Yes", "No"]
    )

    # Aplicar filtros
    df_filt = df_raw[
        (df_raw["Department"].isin(sel_dept))
        & (df_raw["JobRole"].isin(sel_roles))
        & (df_raw["OverTime"].isin(sel_overtime))
    ]

    st.caption(f"Mostrando {len(df_filt):,} de {len(df_raw):,} funcionários")

    if df_filt.empty:
        st.warning("Nenhum registro encontrado com os filtros selecionados.")
        return

    # ---- Gráficos ----

    # Taxa de churn por departamento
    st.subheader("Taxa de Churn por Departamento")
    fig1 = plot_churn_rate_by(df_filt, "Department")
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander("💡 Insight"):
        st.markdown(
            "O departamento de **Vendas** tende a apresentar a maior taxa de churn, "
            "possivelmente por pressão de metas e natureza competitiva do setor."
        )

    # Taxa de churn por cargo
    st.subheader("Taxa de Churn por Cargo")
    fig2 = plot_churn_rate_by(df_filt, "JobRole")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander("💡 Insight"):
        st.markdown(
            "Representantes de vendas e técnicos de laboratório costumam ter "
            "taxas de churn mais elevadas, indicando maior rotatividade em "
            "cargos de entrada ou alta pressão."
        )

    # Distribuição de salário por churn
    st.subheader("Distribuição de Salário por Churn")
    fig3 = plot_distribution(df_filt, "MonthlyIncome", kind="box")
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander("💡 Insight"):
        st.markdown(
            "Funcionários que saíram tendem a ter salários menores. "
            "Remuneração competitiva é um fator de retenção importante."
        )

    # Heatmap de correlação
    st.subheader("Heatmap de Correlação")
    numeric_df = df_filt.select_dtypes(include="number")
    fig4 = plot_correlation_heatmap(numeric_df, max_features=15)
    st.plotly_chart(fig4, use_container_width=True)
    with st.expander("💡 Insight"):
        st.markdown(
            "Variáveis como **OverTime**, **MonthlyIncome**, **Age** e "
            "**TotalWorkingYears** possuem correlações relevantes com o churn."
        )


# =========================================================================
# Página 3: Modelo Preditivo
# =========================================================================


def page_model() -> None:
    """Renderiza métricas, confusion matrix, importância e comparação."""
    st.title("Modelo Preditivo de Churn")

    with st.spinner("Avaliando modelos... (pode levar alguns segundos)"):
        data = get_model_metrics()

    results = data["results"]
    comparison = data["comparison"]
    feature_names = data["feature_names"]

    # Métricas do melhor modelo (XGBoost)
    best = results["xgboost"]

    st.subheader("Métricas do Melhor Modelo (XGBoost)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{best['accuracy']:.3f}")
    col2.metric("Precision", f"{best['precision']:.3f}")
    col3.metric("Recall", f"{best['recall']:.3f}")
    col4.metric("F1-Score", f"{best['f1']:.3f}")
    col5.metric("AUC-ROC", f"{best['auc_roc']:.3f}")

    st.markdown("---")

    # Layout em duas colunas: Confusion Matrix e Feature Importance
    left, right = st.columns(2)

    with left:
        st.subheader("Matriz de Confusão")
        cm = best["confusion_matrix"]
        labels = ["Não saiu (0)", "Saiu (1)"]
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=18),
                showscale=False,
            )
        )
        fig_cm.update_layout(
            xaxis_title="Predito",
            yaxis_title="Real",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        st.subheader("Importância das Features")
        if "feature_importances" in best:
            fig_imp = plot_feature_importance(
                feature_names, best["feature_importances"], top_n=15
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # Tabela comparativa
    st.subheader("Comparação entre Modelos")
    styled = comparison.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda")
    st.dataframe(styled, use_container_width=True)


# =========================================================================
# Página 4: Simulador de Churn
# =========================================================================

# Valores para selectboxes (devem corresponder ao encoding do dataset)
_DEPARTMENTS = ["Human Resources", "Research & Development", "Sales"]
_JOB_ROLES = [
    "Healthcare Representative",
    "Human Resources",
    "Laboratory Technician",
    "Manager",
    "Manufacturing Director",
    "Research Director",
    "Research Scientist",
    "Sales Executive",
    "Sales Representative",
]
_MARITAL_STATUS = ["Divorced", "Married", "Single"]
_BUSINESS_TRAVEL = ["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
_EDUCATION_FIELDS = [
    "Human Resources",
    "Life Sciences",
    "Marketing",
    "Medical",
    "Other",
    "Technical Degree",
]
_GENDERS = ["Female", "Male"]


def page_simulator() -> None:
    """Formulário de predição individual de risco de churn."""
    st.title("Simulador de Risco de Churn")
    st.markdown(
        "Insira os dados de um funcionário para calcular a probabilidade "
        "de saída da empresa."
    )

    with st.spinner("Carregando modelo..."):
        model = load_trained_model()
        df_proc = load_processed_data()
        feature_names = [c for c in df_proc.columns if c != TARGET_COL]

    st.markdown("---")

    # ---- Formulário de inputs ----
    with st.form("churn_form"):
        st.subheader("Dados do Funcionário")

        # Linha 1: dados pessoais
        c1, c2, c3, c4 = st.columns(4)
        age = c1.slider("Idade", 18, 60, 35)
        gender = c2.selectbox("Gênero", _GENDERS)
        marital_status = c3.selectbox("Estado Civil", _MARITAL_STATUS, index=1)
        distance = c4.slider("Distância de Casa (km)", 1, 30, 10)

        # Linha 2: dados profissionais
        c5, c6, c7, c8 = st.columns(4)
        department = c5.selectbox("Departamento", _DEPARTMENTS, index=1)
        job_role = c6.selectbox("Cargo", _JOB_ROLES, index=6)
        job_level = c7.slider("Nível do Cargo", 1, 5, 2)
        monthly_income = c8.slider("Salário Mensal", 1000, 20000, 5000, step=500)

        # Linha 3: tempo de empresa
        c9, c10, c11, c12 = st.columns(4)
        years_company = c9.slider("Anos na Empresa", 0, 40, 5)
        years_role = c10.slider("Anos no Cargo Atual", 0, 18, 3)
        years_manager = c11.slider("Anos com Gestor Atual", 0, 17, 3)
        years_promo = c12.slider("Anos desde Promoção", 0, 15, 1)

        # Linha 4: satisfação e desempenho
        c13, c14, c15, c16 = st.columns(4)
        job_satisfaction = c13.slider("Satisfação no Trabalho", 1, 4, 3)
        env_satisfaction = c14.slider("Satisfação com Ambiente", 1, 4, 3)
        work_life = c15.slider("Work-Life Balance", 1, 4, 3)
        job_involvement = c16.slider("Envolvimento no Trabalho", 1, 4, 3)

        # Linha 5: outros fatores
        c17, c18, c19, c20 = st.columns(4)
        overtime = c17.selectbox("Hora Extra", ["No", "Yes"])
        business_travel = c18.selectbox("Viagens de Negócio", _BUSINESS_TRAVEL, index=2)
        education = c19.slider("Nível de Educação", 1, 5, 3)
        education_field = c20.selectbox("Área de Formação", _EDUCATION_FIELDS, index=1)

        # Linha 6: extras
        c21, c22, c23, c24 = st.columns(4)
        num_companies = c21.slider("Empresas Anteriores", 0, 9, 2)
        training = c22.slider("Treinamentos no Ano", 0, 6, 3)
        stock_option = c23.slider("Opção de Ações", 0, 3, 1)
        perf_rating = c24.slider("Avaliação de Desempenho", 1, 4, 3)

        # Demais features com defaults razoáveis
        c25, c26, c27, _ = st.columns(4)
        daily_rate = c25.number_input("Taxa Diária", 100, 1500, 800)
        hourly_rate = c26.number_input("Taxa por Hora", 30, 100, 65)
        monthly_rate = c27.number_input("Taxa Mensal", 2000, 27000, 14000)

        c28, c29, _, _ = st.columns(4)
        percent_hike = c28.slider("% Aumento Salarial", 11, 25, 14)
        relationship_sat = c29.slider("Satisfação com Relacionamento", 1, 4, 3)

        submitted = st.form_submit_button(
            "🔮 Calcular Risco de Churn", use_container_width=True
        )

    # ---- Resultado da predição ----
    if submitted:
        # Montar dicionário com features numéricas
        total_working_years = years_company + num_companies * 3
        employee_data = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance,
            "Education": education,
            "EnvironmentSatisfaction": env_satisfaction,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies,
            "PercentSalaryHike": percent_hike,
            "PerformanceRating": perf_rating,
            "RelationshipSatisfaction": relationship_sat,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training,
            "WorkLifeBalance": work_life,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager,
            # Features engenheiradas
            "SalaryPerAge": monthly_income / max(age, 1),
            "YearsAtCompanyRatio": (
                years_company / max(total_working_years, 1)
            ),
            "SatisfactionScore": (
                job_satisfaction + env_satisfaction + relationship_sat
            ) / 3,
            "IsNewEmployee": int(years_company <= 1),
            # One-hot: BusinessTravel (drop_first → Non-Travel é baseline)
            "BusinessTravel_Travel_Frequently": int(
                business_travel == "Travel_Frequently"
            ),
            "BusinessTravel_Travel_Rarely": int(
                business_travel == "Travel_Rarely"
            ),
            # One-hot: Department (drop_first → Human Resources é baseline)
            "Department_Research & Development": int(
                department == "Research & Development"
            ),
            "Department_Sales": int(department == "Sales"),
            # One-hot: EducationField (drop_first → Human Resources é baseline)
            "EducationField_Life Sciences": int(
                education_field == "Life Sciences"
            ),
            "EducationField_Marketing": int(education_field == "Marketing"),
            "EducationField_Medical": int(education_field == "Medical"),
            "EducationField_Other": int(education_field == "Other"),
            "EducationField_Technical Degree": int(
                education_field == "Technical Degree"
            ),
            # One-hot: Gender (drop_first → Female é baseline)
            "Gender_Male": int(gender == "Male"),
            # One-hot: JobRole (drop_first → Healthcare Representative é baseline)
            "JobRole_Human Resources": int(job_role == "Human Resources"),
            "JobRole_Laboratory Technician": int(
                job_role == "Laboratory Technician"
            ),
            "JobRole_Manager": int(job_role == "Manager"),
            "JobRole_Manufacturing Director": int(
                job_role == "Manufacturing Director"
            ),
            "JobRole_Research Director": int(job_role == "Research Director"),
            "JobRole_Research Scientist": int(job_role == "Research Scientist"),
            "JobRole_Sales Executive": int(job_role == "Sales Executive"),
            "JobRole_Sales Representative": int(
                job_role == "Sales Representative"
            ),
            # One-hot: MaritalStatus (drop_first → Divorced é baseline)
            "MaritalStatus_Married": int(marital_status == "Married"),
            "MaritalStatus_Single": int(marital_status == "Single"),
            # One-hot: OverTime (drop_first → No é baseline)
            "OverTime_Yes": int(overtime == "Yes"),
        }

        result = predict_churn(model, employee_data, feature_names)

        st.markdown("---")
        st.subheader("Resultado da Predição")

        # Gauge chart
        prob = result["probability"] * 100
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%"},
                title={"text": "Probabilidade de Churn"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": result["risk_color"]},
                    "steps": [
                        {"range": [0, 30], "color": "#d5f5e3"},
                        {"range": [30, 60], "color": "#fdebd0"},
                        {"range": [60, 100], "color": "#fadbd8"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": prob,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=350, template="plotly_white")

        r1, r2 = st.columns([2, 1])
        with r1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with r2:
            st.markdown("### Classificação")
            st.markdown(
                f"<h1 style='color: {result['risk_color']}; text-align: center;'>"
                f"{result['risk_label']}</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Probabilidade:** {prob:.1f}%")

            st.markdown("### Top Fatores de Risco")
            for i, factor in enumerate(result["top_factors"][:5], 1):
                val = employee_data.get(factor, "—")
                st.markdown(f"**{i}.** {factor} = `{val}`")


# =========================================================================
# Roteamento de páginas
# =========================================================================

_PAGE_MAP = {
    "home": page_home,
    "eda": page_eda,
    "model": page_model,
    "simulator": page_simulator,
}

_PAGE_MAP[page]()

# Rodapé
st.sidebar.markdown("---")
st.sidebar.caption("IBM HR Analytics • 2026")
