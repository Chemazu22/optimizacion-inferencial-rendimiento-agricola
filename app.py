"""Dashboard de rendimiento agrícola e inferencia de tratamientos."""

from __future__ import annotations

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import streamlit as st


st.set_page_config(
    page_title="Rendimiento agrícola | Análisis inferencial",
    page_icon="🌱",
    layout="wide",
)

NUMERIC_COLUMNS = [
    "volumen_riego",
    "fertilizante_kg",
    "temperatura_c",
    "rendimiento_ton",
]
TREATMENTS = ["Tratamiento_A", "Tratamiento_B"]
ALPHA = 0.05


@st.cache_data(show_spinner="Generando las 100 parcelas en DuckDB…")
def load_simulated_data() -> pd.DataFrame:
    """Genera una simulación reproducible en DuckDB y la extrae con SQL."""
    con = duckdb.connect(database=":memory:")
    try:
        # Debe ejecutarse antes de cualquier llamada a random().
        con.execute("SELECT setseed(0.42);")
        con.execute(
            """
            CREATE TABLE parcelas AS
            WITH aleatorios AS MATERIALIZED (
                SELECT
                    parcela_id,
                    random() AS u_riego,
                    random() AS u_fertilizante,
                    random() AS u_temperatura,
                    random() AS u_tratamiento,
                    random() AS u_ruido
                FROM range(1, 101) AS ids(parcela_id)
            ),
            variables AS (
                SELECT
                    parcela_id,
                    ROUND(100.0 + 200.0 * u_riego, 2) AS volumen_riego,
                    ROUND(50.0 + 100.0 * u_fertilizante, 2) AS fertilizante_kg,
                    ROUND(18.0 + 14.0 * u_temperatura, 2) AS temperatura_c,
                    CASE WHEN u_tratamiento < 0.50
                         THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END
                        AS tipo_tratamiento,
                    (1.8 * CASE WHEN u_tratamiento < 0.50 THEN 0.0 ELSE 1.0 END)
                        AS efecto_tratamiento,
                    -0.8 + 1.6 * u_ruido AS ruido
                FROM aleatorios
            )
            SELECT
                parcela_id,
                volumen_riego,
                fertilizante_kg,
                temperatura_c,
                tipo_tratamiento,
                ROUND(
                    5.0
                    + 0.035 * volumen_riego
                    + 0.020 * fertilizante_kg
                    + efecto_tratamiento
                    + ruido,
                    2
                ) AS rendimiento_ton
            FROM variables
            ORDER BY parcela_id;
            """
        )
        return con.execute("SELECT * FROM parcelas ORDER BY parcela_id").df()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fit_ols(df: pd.DataFrame):
    x = sm.add_constant(df[["volumen_riego", "fertilizante_kg"]], has_constant="add")
    return sm.OLS(df["rendimiento_ton"], x).fit()


def treatment_inference(df: pd.DataFrame) -> dict:
    group_a = df.loc[df["tipo_tratamiento"] == "Tratamiento_A", "rendimiento_ton"]
    group_b = df.loc[df["tipo_tratamiento"] == "Tratamiento_B", "rendimiento_ton"]
    if len(group_a) < 2 or len(group_b) < 2:
        return {"available": False, "n_a": len(group_a), "n_b": len(group_b)}

    # Varianzas agrupadas, coherentes con la prueba t clásica de Student.
    test = stats.ttest_ind(group_b, group_a, equal_var=True, alternative="greater")
    difference = float(group_b.mean() - group_a.mean())
    degrees_freedom = len(group_a) + len(group_b) - 2
    pooled_variance = (
        (len(group_a) - 1) * group_a.var(ddof=1)
        + (len(group_b) - 1) * group_b.var(ddof=1)
    ) / degrees_freedom
    standard_error = (pooled_variance * (1 / len(group_a) + 1 / len(group_b))) ** 0.5
    critical_value = stats.t.ppf(0.975, degrees_freedom)
    return {
        "available": True,
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "difference": difference,
        "p_value": float(test.pvalue),
        "statistic": float(test.statistic),
        "ci_low": difference - critical_value * standard_error,
        "ci_high": difference + critical_value * standard_error,
    }


def show_inference_result(result: dict, scope: str) -> None:
    if not result["available"]:
        st.warning(
            f"No hay al menos dos observaciones de cada grupo {scope}; "
            "no se puede calcular la prueba t ni su intervalo."
        )
        return
    st.write(
        f"Diferencia de medias (B − A): **{result['difference']:.2f} ton/ha**; "
        f"IC 95% [{result['ci_low']:.2f}, {result['ci_high']:.2f}]; "
        f"t = {result['statistic']:.3f}; p unilateral = {result['p_value']:.4g}."
    )
    if result["p_value"] < ALPHA:
        st.success(
            f"Se rechaza H₀ a α = {ALPHA:.2f}. La evidencia apoya un mayor "
            "rendimiento con B; se recomienda evaluar el escalamiento con un "
            "piloto económico y operativo antes de convertirlo en inversión total."
        )
    else:
        st.info(
            f"No se rechaza H₀ a α = {ALPHA:.2f}. La evidencia no basta para "
            "recomendar la inversión generalizada en B."
        )


def make_chart() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)


st.title("Optimización inferencial del rendimiento agrícola")
st.caption(
    "Simulación reproducible de parcelas de invernadero · DuckDB, OLS y prueba t"
)

data = load_simulated_data()
model = fit_ols(data)
global_inference = treatment_inference(data)

st.sidebar.header("Controles de análisis")
selected_treatments = st.sidebar.multiselect(
    "Tipo de tratamiento", options=TREATMENTS, default=TREATMENTS
)
water_min, water_max = float(data.volumen_riego.min()), float(data.volumen_riego.max())
fert_min, fert_max = float(data.fertilizante_kg.min()), float(data.fertilizante_kg.max())
water_range = st.sidebar.slider(
    "Volumen de riego (m³/ha)", water_min, water_max,
    (water_min, water_max), step=0.01,
)
fert_range = st.sidebar.slider(
    "Fertilizante (kg/ha)", fert_min, fert_max,
    (fert_min, fert_max), step=0.01,
)
filtered = data[
    data["tipo_tratamiento"].isin(selected_treatments)
    & data["volumen_riego"].between(*water_range)
    & data["fertilizante_kg"].between(*fert_range)
].copy()

st.sidebar.divider()
st.sidebar.caption(
    "Las gráficas y la tabla respetan los filtros. Las tarjetas KPI, el modelo "
    "OLS y la prueba inferencial principal usan las 100 parcelas globales."
)
with st.sidebar.expander("Inspeccionar y descargar datos"):
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "Descargar CSV filtrado",
        data=filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="parcelas_invernadero.csv",
        mime="text/csv",
    )

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Rendimiento promedio global", f"{data.rendimiento_ton.mean():.2f} ton/ha")
kpi2.metric("R² ajustado OLS", f"{model.rsquared_adj:.1%}")
if global_inference["available"]:
    kpi3.metric(
        "P-valor t unilateral (global)",
        f"{global_inference['p_value']:.4g}",
        delta="Significativo" if global_inference["p_value"] < ALPHA else "No significativo",
        delta_color="normal" if global_inference["p_value"] < ALPHA else "off",
    )
else:
    kpi3.metric("P-valor t unilateral (global)", "N/D")

with st.expander("Modelo OLS y prueba de hipótesis global", expanded=False):
    st.markdown(
        "**Modelo:** rendimiento ~ riego + fertilizante. "
        "**H₀:** μB − μA ≤ 0 · **H₁:** μB − μA > 0 · α = 0.05. "
        "La prueba compara los dos tratamientos globalmente, sin los filtros del sidebar."
    )
    rmse = float((model.resid.pow(2).mean()) ** 0.5)
    st.write(f"**RMSE OLS global:** {rmse:.3f} ton/ha")
    st.text(model.summary().as_text())
    st.dataframe(
        pd.DataFrame({"coeficiente": model.params, "p_valor": model.pvalues}),
        width="stretch",
    )
    show_inference_result(global_inference, "globales")

st.subheader(f"Exploración de parcelas filtradas ({len(filtered)} de {len(data)})")
row1_left, row1_right = st.columns(2)
row2_left, row2_right = st.columns(2)

with row1_left:
    st.markdown("#### 1. Correlación de Pearson")
    if len(filtered) >= 2:
        corr = filtered[NUMERIC_COLUMNS].corr(method="pearson")
        fig, ax = make_chart()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                    vmin=-1, vmax=1, square=True, ax=ax)
        ax.set_title("Correlación entre variables numéricas")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        associations = corr["rendimiento_ton"].drop("rendimiento_ton").abs()
        strongest = associations.idxmax()
        st.caption(
            f"En la selección actual, **{strongest}** tiene la asociación lineal "
            f"más fuerte en magnitud con el rendimiento (|r| = {associations[strongest]:.2f}). "
            "La correlación describe asociación, no causalidad."
        )
    else:
        st.info("Se requieren al menos dos filas para calcular la correlación.")

with row1_right:
    st.markdown("#### 2. Riego y rendimiento: ajuste lineal")
    if len(filtered) >= 2 and filtered.volumen_riego.nunique() > 1:
        fig, ax = make_chart()
        sns.regplot(
            data=filtered, x="volumen_riego", y="rendimiento_ton", ax=ax,
            ci=95, scatter_kws={"alpha": 0.75, "s": 42},
            line_kws={"color": "#c0392b", "linewidth": 2},
        )
        ax.set(xlabel="Volumen de riego (m³/ha)", ylabel="Rendimiento (ton/ha)")
        ax.set_title("Dispersión y recta OLS con banda de confianza del 95%")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    else:
        st.info("Se requieren dos o más valores de riego distintos para el ajuste.")

with row2_left:
    st.markdown("#### 3. Rendimiento por tratamiento")
    if not filtered.empty:
        fig, ax = make_chart()
        sns.boxplot(data=filtered, x="tipo_tratamiento", y="rendimiento_ton",
                    order=TREATMENTS, color="#8ecf9b", showfliers=True, ax=ax)
        means = filtered.groupby("tipo_tratamiento")["rendimiento_ton"].mean()
        for position, treatment in enumerate(TREATMENTS):
            if treatment in means.index:
                ax.scatter(position, means[treatment], marker="D", color="#b03a2e",
                           zorder=4, label="Media" if position == 0 else None)
        ax.set(xlabel="Tratamiento", ylabel="Rendimiento (ton/ha)")
        ax.set_title("Mediana, cuartiles, atípicos y media (◆)")
        ax.legend(loc="best")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        filtered_inference = treatment_inference(filtered)
        show_inference_result(filtered_inference, "filtrados")
    else:
        st.info("No hay parcelas para los filtros seleccionados.")

with row2_right:
    st.markdown("#### 4. Distribución de residuos OLS")
    # Los residuos corresponden al modelo global de dos predictores.
    residuals = model.resid
    fig, ax = make_chart()
    sns.histplot(residuals, kde=True, stat="density", bins=12, color="#4c78a8", ax=ax)
    ax.axvline(0, color="#c0392b", linestyle="--", linewidth=1.7, label="Residuo = 0")
    ax.set(xlabel="Residuo (ton/ha)", ylabel="Densidad")
    ax.set_title("Residuos del modelo global")
    ax.legend()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "El histograma/KDE permite inspección visual, pero no sustituye una prueba "
        "formal de normalidad."
    )

st.divider()
st.caption(
    "Datos sintéticos para fines académicos. La decisión de inversión requiere "
    "validación en campo y análisis de costos, calidad, consumo hídrico y riesgos."
)
