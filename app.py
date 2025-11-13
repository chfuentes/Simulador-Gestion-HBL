# app.py (versión producción)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Configuración optimizada para producción
st.set_page_config(
    page_title="Simulador Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
MONTHS = ['ENERO','FEBRERO','MARZO','ABRIL','MAYO','JUNIO',
          'JULIO','AGOSTO','SEPTIEMBRE','OCTUBRE','NOVIEMBRE','DICIEMBRE']
M2N = {m:i+1 for i,m in enumerate(MONTHS)}
N2M = {i+1:m for i,m in enumerate(MONTHS)}
N_SIM = 10000  # Reducido para mejor performance web
SEED = 42

@st.cache_data(show_spinner=False)
def read_and_trim(uploaded_file):
    """Lee y procesa el archivo CSV subido - Versión robusta"""
    try:
        # Leer el contenido del archivo para diagnóstico
        content = uploaded_file.getvalue().decode('latin-1')
        lines = content.split('\n')
        
        st.write("🔍 **Diagnóstico:** Primera línea del archivo:")
        st.code(lines[0] if lines else "Archivo vacío")
        
        # Intentar diferentes combinaciones
        encodings = ['latin-1', 'utf-8', 'cp1252']
        separators = [';', ',', '\t']
        
        for encoding in encodings:
            for separator in separators:
                try:
                    uploaded_file.seek(0)  # Resetear el archivo
                    df = pd.read_csv(
                        uploaded_file, 
                        sep=separator,
                        encoding=encoding,
                        decimal=',',
                        engine='python'
                    )
                    st.success(f"✅ Archivo leído con: encoding={encoding}, separator='{separator}'")
                    break
                except:
                    continue
            else:
                continue
            break
        else:
            st.error("No se pudo leer el archivo con ninguna combinación de encoding/separador")
            return None
            
    except Exception as e:
        st.error(f"Error crítico al leer el archivo: {e}")
        return None

    # Mostrar columnas detectadas para debugging
    st.write("📋 **Columnas detectadas en el archivo:**")
    st.write(df.columns.tolist())
    
    # Normalizar nombres de columnas (manejar "AčO" y otros problemas)
    column_mapping = {}
    for col in df.columns:
        col_clean = col.strip().upper()
        if 'AčO' in col_clean or 'AÑO' in col_clean or 'ANIO' in col_clean or col_clean == 'AčO':
            column_mapping[col] = 'ANIO'
        elif 'MES' in col_clean:
            column_mapping[col] = 'MES'
        elif 'NUMERADOR' in col_clean:
            column_mapping[col] = 'NUMERADOR'
        elif 'DENOMINADOR' in col_clean:
            column_mapping[col] = 'DENOMINADOR'
        elif '%' in col_clean:
            continue  # Saltar columna de porcentaje
    
    df = df.rename(columns=column_mapping)
    
    # Verificar que tenemos las columnas necesarias
    required_columns = ['ANIO', 'MES', 'NUMERADOR', 'DENOMINADOR']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Faltan columnas requeridas: {missing_columns}")
        st.write("Columnas disponibles:", df.columns.tolist())
        return None
    
    # Continuar con el procesamiento normal...
    df["MES"] = df["MES"].astype(str).str.strip().str.upper()
    df["m"] = df["MES"].map(M2N)

    for c in ["NUMERADOR","DENOMINADOR"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all").sort_values(["ANIO","m"], na_position="last").reset_index(drop=True)

    essential = ["ANIO","m","NUMERADOR","DENOMINADOR"]
    invalid_mask = df[essential].isna().any(axis=1)
    if invalid_mask.any():
        cut_idx = invalid_mask.idxmax()
        df = df.loc[:cut_idx-1].reset_index(drop=True)

    df = df.dropna(subset=["ANIO","m","NUMERADOR","DENOMINADOR"]).reset_index(drop=True)
    df["pct"] = df["NUMERADOR"] / df["DENOMINADOR"]
    
    st.success(f"✅ Datos procesados correctamente: {len(df)} registros")
    return df

def average_method(df):
    """Calcula el promedio de los últimos 3 meses"""
    last = df.tail(3)
    den = last["DENOMINADOR"].sum(min_count=1)
    num = last["NUMERADOR"].sum(min_count=1)
    if not np.isfinite(den) or den <= 0:
        den = df["DENOMINADOR"].sum(min_count=1)
        num = df["NUMERADOR"].sum(min_count=1)
    return np.divide(num, den)

def linear_trend_forecast(y, steps):
    """Proyecta una serie temporal usando regresión lineal"""
    y = pd.to_numeric(y, errors="coerce").dropna().reset_index(drop=True)
    if len(y) < 3:
        raise ValueError("Datos insuficientes para regresión.")
    x = np.arange(len(y)).reshape(-1,1)
    model = LinearRegression().fit(x, y.values)
    xf = np.arange(len(y), len(y)+steps).reshape(-1,1)
    return model.predict(xf)

def seasonal_indices(df):
    """Calcula índices estacionales multiplicativos"""
    muN, muD = df["NUMERADOR"].mean(), df["DENOMINADOR"].mean()
    idxN = df.groupby("m")["NUMERADOR"].mean() / muN
    idxD = df.groupby("m")["DENOMINADOR"].mean() / muD
    return idxN, idxD

def mc_ratio_poisson(muN, muD, n=N_SIM, seed=SEED):
    """Monte Carlo con distribución Poisson"""
    rng = np.random.default_rng(seed)
    N = rng.poisson(muN, size=n)
    D = rng.poisson(muD, size=n)
    D = np.clip(D, 1, None)
    return float(np.median(N/D))

def mc_ratio_negbinom(muN, varN, muD, varD, n=N_SIM, seed=SEED):
    """Monte Carlo con distribución Binomial Negativa"""
    rng = np.random.default_rng(seed)
    
    def nb_params(mu, var):
        if var <= mu:
            var = mu * 1.1
        p = mu / var
        r = mu * mu / (var - mu)
        return max(r, 0.1), min(max(p, 1e-6), 1-1e-6)
    
    rN, pN = nb_params(muN, varN)
    rD, pD = nb_params(muD, varD)
    
    N = rng.negative_binomial(rN, pN, size=n)
    D = rng.negative_binomial(rD, pD, size=n)
    D = np.clip(D, 1, None)
    return float(np.median(N/D))

def mc_ratio_gamma(muN, sdN, muD, sdD, n=N_SIM, seed=SEED):
    """Monte Carlo con distribución Gamma"""
    rng = np.random.default_rng(seed)
    
    def gamma_params(mu, sd):
        if sd <= 0:
            sd = mu * 0.1
        k = (mu / sd) ** 2
        theta = sd ** 2 / mu
        return max(k, 0.1), max(theta, 1e-6)
    
    kN, thetaN = gamma_params(muN, sdN)
    kD, thetaD = gamma_params(muD, sdD)
    
    N = rng.gamma(kN, thetaN, size=n)
    D = rng.gamma(kD, thetaD, size=n)
    D = np.clip(D, 1e-6, None)
    return float(np.median(N/D))

def mc_ratio_adaptive(muN, sdN, muD, sdD, n=N_SIM, seed=SEED):
    """Monte Carlo adaptativo"""
    rng = np.random.default_rng(seed)
    
    # Determinar distribución para NUMERADOR
    if muN < 20:
        N = rng.poisson(muN, size=n)
    elif muN >= 30 and sdN**2 <= muN * 1.5:
        N = rng.normal(muN, sdN, size=n)
        N = np.maximum(N, 0)
    else:
        if sdN <= 0:
            sdN = np.sqrt(muN)
        k = (muN / sdN) ** 2
        theta = sdN ** 2 / muN
        N = rng.gamma(max(k, 0.1), max(theta, 1e-6), size=n)
    
    # Determinar distribución para DENOMINADOR
    if muD < 20:
        D = rng.poisson(muD, size=n)
    elif muD >= 30 and sdD**2 <= muD * 1.5:
        D = rng.normal(muD, sdD, size=n)
        D = np.maximum(D, 1)
    else:
        if sdD <= 0:
            sdD = np.sqrt(muD)
        k = (muD / sdD) ** 2
        theta = sdD ** 2 / muD
        D = rng.gamma(max(k, 0.1), max(theta, 1e-6), size=n)
    
    D = np.clip(D, 1e-6, None)
    return float(np.median(N/D))

def next_three_months_from_last(df):
    """Determina los próximos 3 meses a proyectar"""
    last_m = int(df["m"].iloc[-1])
    months = [((last_m + i - 1) % 12) + 1 for i in (1,2,3)]
    return months

def generate_html_report(df, out, muN, sdN, muD, sdD, varN, varD, overdispersed_N, overdispersed_D):
    """Genera el reporte HTML"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Proyecciones</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 100%; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-sizing: border-box; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            
            .table-wrapper {{ overflow-x: auto; margin: 20px 0; }}
            table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 13px; }}
            th, td {{ padding: 10px 8px; text-align: right; border: 1px solid #ddd; white-space: nowrap; }}
            th {{ background: #3498db; color: white; position: sticky; top: 0; z-index: 10; }}
            td:first-child, th:first-child {{ 
                text-align: left; 
                font-weight: bold; 
                position: sticky; 
                left: 0; 
                background: white; 
                z-index: 5;
            }}
            th:first-child {{ background: #3498db; z-index: 15; }}
            tr:nth-child(even) td:not(:first-child) {{ background: #f2f2f2; }}
            tr:hover td {{ background: #e8f4f8; }}
            
            .recomendacion {{ background: #d5f4e6; padding: 15px; border-left: 4px solid #27ae60; margin: 20px 0; border-radius: 5px; }}
            .info {{ background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 20px 0; border-radius: 5px; }}
            
            .metodos-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 15px; 
                margin: 20px 0; 
            }}
            .metodo-card {{ 
                background: #f8f9fa; 
                padding: 15px; 
                border-left: 4px solid #95a5a6; 
                border-radius: 5px; 
            }}
            .metodo-card.recomendado {{ border-left-color: #27ae60; background: #d5f4e6; }}
            .metodo-card h4 {{ margin-top: 0; color: #2c3e50; }}
            
            @media (max-width: 768px) {{
                .container {{ padding: 15px; }}
                table {{ font-size: 11px; }}
                th, td {{ padding: 6px 4px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Reporte de Proyecciones Epidemiológicas</h1>
            
            <div class="info">
                <h3>ℹ️ Información del Dataset</h3>
                <p><strong>Registros:</strong> {len(df)}</p>
                <p><strong>Período:</strong> {df['ANIO'].min()}/{df['MES'].iloc[0]} - {df['ANIO'].max()}/{df['MES'].iloc[-1]}</p>
                <p><strong>Numerador (Media ± SD):</strong> {muN:.2f} ± {sdN:.2f}</p>
                <p><strong>Denominador (Media ± SD):</strong> {muD:.2f} ± {sdD:.2f}</p>
                <p><strong>Sobredispersión N:</strong> {'Sí' if overdispersed_N else 'No'} (Var/Media = {varN/muN:.2f})</p>
                <p><strong>Sobredispersión D:</strong> {'Sí' if overdispersed_D else 'No'} (Var/Media = {varD/muD:.2f})</p>
            </div>

            <div class="recomendacion">
                <h3>✅ Recomendación</h3>
                <p>Para datos epidemiológicos, el <strong>Método Monte Carlo Adaptativo Estacional</strong> 
                es el más apropiado ya que:</p>
                <ul>
                    <li>Selecciona automáticamente la distribución probabilística correcta</li>
                    <li>Considera la estacionalidad de los datos</li>
                    <li>Maneja apropiadamente conteos pequeños y grandes</li>
                    <li>Respeta la naturaleza no-negativa de los datos</li>
                </ul>
            </div>

            <h2>Proyecciones para los Próximos 3 Meses</h2>
            <div class="table-wrapper">
                {out.to_html(index=False, classes='styled-table', float_format=lambda x: f'{x*100:.2f}%')}
            </div>
            
            <h2>📈 Descripción de Métodos</h2>
            <div class="metodos-grid">
                <div class="metodo-card">
                    <h4>📊 Promedio</h4>
                    <p>Baseline simple basado en los últimos 3 meses disponibles.</p>
                </div>
                
                <div class="metodo-card">
                    <h4>📈 Pronóstico Lineal</h4>
                    <p>Extrapolación de la tendencia histórica mediante regresión lineal.</p>
                </div>
                
                <div class="metodo-card">
                    <h4>🎲 MC Poisson</h4>
                    <p>Simulación Monte Carlo con distribución Poisson. Apropiado para conteos de eventos raros (λ < 20).</p>
                </div>
                
                <div class="metodo-card">
                    <h4>🎲 MC Binomial Negativa</h4>
                    <p>Para datos con sobredispersión (varianza > media). Común en epidemiología.</p>
                </div>
                
                <div class="metodo-card">
                    <h4>🎲 MC Gamma</h4>
                    <p>Para datos continuos estrictamente positivos con sesgo hacia la derecha.</p>
                </div>
                
                <div class="metodo-card recomendado">
                    <h4>⭐ MC Adaptativo (Recomendado)</h4>
                    <p>Selecciona automáticamente la distribución más apropiada según las características de los datos.</p>
                </div>
                
                <div class="metodo-card recomendado">
                    <h4>🌟 Versiones Estacionales</h4>
                    <p>Incluyen ajuste por patrones mensuales históricos. Más precisas para datos con estacionalidad.</p>
                </div>
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background: #ecf0f1; border-radius: 5px;">
                <h3>💡 Cómo Interpretar los Resultados</h3>
                <ol>
                    <li><strong>Compare los métodos:</strong> Si hay gran diferencia entre métodos, los datos tienen alta incertidumbre.</li>
                    <li><strong>Prefiera métodos estacionales:</strong> Si sus datos tienen patrones mensuales conocidos.</li>
                    <li><strong>Use MC Adaptativo Estacional:</strong> Como mejor estimación general para epidemiología.</li>
                    <li><strong>Considere intervalos de confianza:</strong> Los métodos Monte Carlo capturan mejor la variabilidad.</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    # Header con información
    st.title("📊 Simulador Epidemiológico - Proyecciones Online")
    st.markdown("""
    **Aplicación web para proyecciones epidemiológicas**  
    Carga un archivo CSV con datos históricos para generar proyecciones de los próximos 3 meses usando múltiples métodos estadísticos.
    Uselo sólo con fines referenciales.
    """)
    
    # Ejemplo de formato
    with st.expander("📋 ¿Qué formato debe tener el CSV?"):
        st.markdown("""
        Tu archivo CSV debe tener estas columnas:
        
        | AÑO | MES | NUMERADOR | DENOMINADOR |
        |-----|-----|-----------|-------------|
        | 2023 | ENERO | 150 | 1000 |
        | 2023 | FEBRERO | 120 | 950 |
        | 2023 | MARZO | 180 | 1100 |
        
        **Requisitos:**
        - Encabezados en español
        - Meses en mayúsculas (ENERO, FEBRERO, etc.)
        - Valores numéricos para numerador y denominador
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Cargar Datos")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo CSV", 
            type=['csv'],
            help="El archivo debe tener columnas: AÑO, MES, NUMERADOR, DENOMINADOR"
        )
        
        st.markdown("---")
        st.header("ℹ️ Información")
        st.markdown("""
        Esta aplicación usa:
        - **Métodos deterministas**: Promedio, Tendencia Lineal
        - **Métodos probabilísticos**: Monte Carlo con diferentes distribuciones
        - **Ajuste estacional**: Patrones mensuales históricos
        """)
        
        st.markdown("---")
        st.markdown("**Desarrollado para análisis epidemiológico**")
        st.markdown("*Versión 1.0 - Christian Fuentes*")

    if uploaded_file is not None:
        # Procesar datos
        with st.spinner("Procesando datos..."):
            df = read_and_trim(uploaded_file)
            
            if df is None or len(df) == 0:
                st.error("No se pudieron procesar los datos. Verifica el formato del archivo.")
                return

            # Mostrar información básica
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Registros procesados", len(df))
            with col2:
                st.metric("Período inicial", f"{df['ANIO'].min()}/{df['MES'].iloc[0]}")
            with col3:
                st.metric("Período final", f"{df['ANIO'].max()}/{df['MES'].iloc[-1]}")

            # Realizar proyecciones
            future_months = next_three_months_from_last(df)

            # Métodos deterministas
            avg_pct = average_method(df)
            predN_lin = linear_trend_forecast(df["NUMERADOR"], steps=3)
            predD_lin = linear_trend_forecast(df["DENOMINADOR"], steps=3)
            lin_pct = predN_lin / predD_lin

            # Parámetros globales
            muN, sdN = df["NUMERADOR"].mean(), df["NUMERADOR"].std(ddof=1)
            muD, sdD = df["DENOMINADOR"].mean(), df["DENOMINADOR"].std(ddof=1)
            varN, varD = df["NUMERADOR"].var(ddof=1), df["DENOMINADOR"].var(ddof=1)

            # Índices estacionales
            idxN, idxD = seasonal_indices(df)

            # Detectar sobredispersión
            overdispersed_N = varN > muN * 1.2
            overdispersed_D = varD > muD * 1.2

            rows = []
            for i, m in enumerate(future_months):
                # Parámetros ajustados estacionalmente
                muNm = muN * idxN.get(m, 1.0)
                muDm = muD * idxD.get(m, 1.0)
                sdNm = sdN * np.sqrt(idxN.get(m, 1.0))
                sdDm = sdD * np.sqrt(idxD.get(m, 1.0))
                varNm = varN * idxN.get(m, 1.0)
                varDm = varD * idxD.get(m, 1.0)

                # Diferentes métodos Monte Carlo
                mc_poisson = mc_ratio_poisson(muN, muD, seed=SEED+i)
                mc_poisson_seas = mc_ratio_poisson(muNm, muDm, seed=SEED+100+i)
                
                mc_negbinom = mc_ratio_negbinom(muN, varN, muD, varD, seed=SEED+200+i)
                mc_negbinom_seas = mc_ratio_negbinom(muNm, varNm, muDm, varDm, seed=SEED+300+i)
                
                mc_gamma = mc_ratio_gamma(muN, sdN, muD, sdD, seed=SEED+400+i)
                mc_gamma_seas = mc_ratio_gamma(muNm, sdNm, muDm, sdDm, seed=SEED+500+i)
                
                mc_adaptive = mc_ratio_adaptive(muN, sdN, muD, sdD, seed=SEED+600+i)
                mc_adaptive_seas = mc_ratio_adaptive(muNm, sdNm, muDm, sdDm, seed=SEED+700+i)

                rows.append({
                    "Mes": N2M[m],
                    "Promedio": avg_pct,
                    "Pronostico_Lineal": lin_pct[i],
                    "MC_Poisson": mc_poisson,
                    "MC_Poisson_Estacional": mc_poisson_seas,
                    "MC_NegBinom": mc_negbinom,
                    "MC_NegBinom_Estacional": mc_negbinom_seas,
                    "MC_Gamma": mc_gamma,
                    "MC_Gamma_Estacional": mc_gamma_seas,
                    "MC_Adaptativo": mc_adaptive,
                    "MC_Adaptativo_Estacional": mc_adaptive_seas
                })

            out = pd.DataFrame(rows)

        # Mostrar resultados
        st.success("✅ Proyecciones calculadas exitosamente!")
        
        # Pestañas para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Resultados", "📊 Gráfico", "📋 Datos", "🌐 Reporte HTML"])
        
        with tab1:
            st.subheader("Proyecciones para los Próximos 3 Meses")
            
            # Formatear tabla para mostrar
            display_df = out.copy()
            for col in display_df.columns:
                if col != "Mes":
                    display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
            
            st.dataframe(display_df, use_container_width=True)
            
            # Método recomendado
            st.info("""
            **🎯 Recomendación:** Para datos epidemiológicos, use **MC Adaptativo Estacional** como método principal, 
            ya que selecciona automáticamente la distribución más apropiada y considera la estacionalidad.
            """)

        with tab2:
            st.subheader("Comparación de Métodos")
            
            methods_to_plot = [
                "Promedio",
                "Pronostico_Lineal", 
                "MC_Adaptativo_Estacional",
                "MC_Poisson_Estacional",
                "MC_Gamma_Estacional"
            ]
            
            labels_plot = [
                "Promedio",
                "Tendencia Lineal",
                "MC Adaptativo (Recomendado)",
                "MC Poisson",
                "MC Gamma"
            ]

            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(out))
            width = 0.15
            
            for j, (mth, label) in enumerate(zip(methods_to_plot, labels_plot)):
                offset = (j - 2) * width
                bars = ax.bar(x + offset, out[mth].values * 100.0, width, label=label)
                
                for xi, val in zip(x + offset, out[mth].values * 100.0):
                    ax.text(xi, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(out["Mes"])
            ax.set_ylabel("Porcentaje (%)")
            ax.set_title("Proyecciones por Método (Próximos 3 Meses)")
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            
            st.pyplot(fig)

        with tab3:
            st.subheader("Datos Procesados")
            st.dataframe(df, use_container_width=True)
            
            # Estadísticas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numerador - Media", f"{muN:.2f}")
                st.metric("Numerador - Desviación", f"{sdN:.2f}")
                st.metric("Sobredispersión N", "Sí" if overdispersed_N else "No")
            with col2:
                st.metric("Denominador - Media", f"{muD:.2f}")
                st.metric("Denominador - Desviación", f"{sdD:.2f}")
                st.metric("Sobredispersión D", "Sí" if overdispersed_D else "No")

        with tab4:
            st.subheader("Reporte HTML Completo")
            
            # Generar HTML
            html_content = generate_html_report(
                df, out, muN, sdN, muD, sdD, varN, varD, 
                overdispersed_N, overdispersed_D
            )
            
            # Mostrar vista previa del HTML
            st.components.v1.html(html_content, height=800, scrolling=True)
            
            # Descargar HTML
            st.download_button(
                label="📥 Descargar Reporte HTML",
                data=html_content,
                file_name="reporte_proyecciones.html",
                mime="text/html"
            )
            
            # Descargar CSV con resultados
            csv = out.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Descargar Resultados CSV",
                data=csv,
                file_name="proyecciones.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()