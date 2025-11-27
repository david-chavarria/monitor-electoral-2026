import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import spacy

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(
    page_title="VOTO 360° | Monitor Ciudadano",
    layout="wide",
    page_icon="🇨🇷",
    initial_sidebar_state="expanded"
)

# --- 2. SISTEMA DE CARGA ROBUSTO (A PRUEBA DE FALLOS) ---

@st.cache_resource
def cargar_modelo_nlp():
    """
    Intenta cargar el modelo de IA. Si falla por restricciones de la nube,
    devuelve None para activar el 'Modo Ligero' sin romper la app.
    """
    try:
        # Intento 1: Carga directa
        if spacy.util.is_package("es_core_news_sm"):
            return spacy.load("es_core_news_sm")
        
        # Intento 2: Descarga en tiempo de ejecución (puede fallar en nube gratuita)
        from spacy.cli import download
        download("es_core_news_sm")
        return spacy.load("es_core_news_sm")
        
    except Exception:
        # Si todo falla, no rompemos la app. Retornamos None y usamos diccionarios.
        return None

nlp = cargar_modelo_nlp()

@st.cache_data
def cargar_datos():
    # Lógica de reintento para lectura de archivos (UTF-8 vs Latin-1)
    df = None
    archivo_target = None
    
    # Buscar archivo (prioridad CSV por velocidad)
    if os.path.exists('datos.csv'):
        archivo_target = 'datos.csv'
        try:
            df = pd.read_csv(archivo_target, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(archivo_target, encoding='latin-1')
            
    elif os.path.exists('Base_Enriquecida_IA.xlsx'):
        archivo_target = 'Base_Enriquecida_IA.xlsx'
        df = pd.read_excel(archivo_target)
    
    if df is None: return None

    # Normalización de Nombres
    NOMBRES = {
        "PSD": "Progreso Social Democrático", "PLN": "Partido Liberación Nacional", "PUSC": "Partido Unidad Social Cristiana",
        "PAC": "Agenda Ciudadana", "FA": "Frente Amplio", "PLP": "Partido Liberal Progresista",
        "PNR": "Nueva República", "PNG": "Partido Nueva Generación", "PIN": "Partido Integración Nacional",
        "PA": "Avanza", "PDLCT": "De la Clase Trabajadora", "ACRM": "Aquí Costa Rica Manda",
        "PPSO": "Pueblo Soberano", "UP": "Unidos Podemos", "CR1": "Alianza Costa Rica Primero",
        "PJSC": "Justicia Social Costarricense", "PUCD": "Unión Costarricense Democrática", "CDS": "Centro Democrático y Social",
        "PEN": "Esperanza Nacional", "PEL": "Esperanza y Libertad", "CAC": "Agenda Ciudadana"
    }
    df['partido_sigla'] = df['partido']
    df['partido'] = df['partido'].map(NOMBRES).fillna(df['partido'])
    
    # Filtro de calidad y limpieza de columnas
    if 'longitud' in df.columns:
        df = df[df['longitud'] > 60]
    
    # Recalcular métricas si no existen (Respaldo)
    if 'SUBJETIVIDAD' not in df.columns:
        df['SUBJETIVIDAD'] = df['texto'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

    # Índices Heurísticos (Siempre disponibles)
    def calc_idx(txt, kw): return 1 if any(k in str(txt).lower() for k in kw) else 0
    
    if 'IDX_ESTATISMO' not in df.columns:
        df['IDX_ESTATISMO'] = df['texto'].apply(lambda x: calc_idx(x, ['estado', 'público', 'institución', 'rectoría', 'regulación']))
        df['IDX_MERCADO'] = df['texto'].apply(lambda x: calc_idx(x, ['privado', 'empresa', 'mercado', 'emprendimiento', 'apertura']))
        df['IDX_GLOBAL'] = df['texto'].apply(lambda x: calc_idx(x, ['internacional', 'mundo', 'ocde', 'fmi', 'exportación']))
        df['IDX_SOCIAL'] = df['texto'].apply(lambda x: calc_idx(x, ['pobreza', 'mujer', 'vulnerable', 'niñez', 'derecho', 'humano']))

    return df

df = cargar_datos()

# --- 3. DISEÑO UI/UX ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    
    :root {
        --primary-blue: #1e3a8a;
        --accent-cyan: #0ea5e9;
        --interactive: #4f46e5;
        --bg-gradient: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }

    .stApp { background: var(--bg-gradient); font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: var(--primary-blue) !important; font-weight: 900; letter-spacing: -0.5px; }
    
    /* HEADER */
    .header-container {
        display: flex; align-items: center; padding: 1.5rem 0;
        border-bottom: 3px solid var(--accent-cyan); margin-bottom: 2rem;
        background: white; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .logo-badge { font-size: 2.5rem; margin-right: 15px; }
    .logo-text { font-size: 2.2rem; font-weight: 900; color: var(--primary-blue); line-height: 1; }
    .logo-highlight { color: var(--accent-cyan); }
    .logo-sub { font-size: 1rem; color: #64748b; font-weight: 600; letter-spacing: 0.05em; }

    /* FOOTER */
    .footer-container {
        margin-top: 4rem; padding: 1.5rem; background: var(--primary-blue);
        color: white; border-radius: 12px 12px 0 0; text-align: center; font-size: 0.9rem;
        opacity: 0.95;
    }

    /* Cards */
    .party-card {
        background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px;
        border-left: 5px solid var(--interactive); box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s; height: 100%;
    }
    .party-card:hover { transform: translateY(-3px); }
    .party-header { font-size: 1.2rem; font-weight: 800; color: var(--primary-blue); margin-bottom: 5px; }
    
    /* Cajas de Interpretación */
    .interpretation-box {
        background-color: #ffffff; border-left: 5px solid var(--accent-cyan); padding: 20px;
        border-radius: 8px; font-size: 0.95rem; color: #334155; margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); line-height: 1.6;
    }
    .interpretation-title {
        font-weight: 800; color: var(--primary-blue); display: block; margin-bottom: 10px;
        text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em;
        border-bottom: 2px solid #f1f5f9; padding-bottom: 5px;
    }
    
    .author-box { background-color: #ffffff; padding: 30px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
    .author-header { font-size: 1.4rem; font-weight: 800; color: var(--primary-blue); margin-bottom: 5px; }
    .stPlotlyChart { background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); padding: 10px; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
    <div class="header-container">
        <div class="logo-badge">🇨🇷</div>
        <div>
            <div class="logo-text">VOTO<span class="logo-highlight">360°</span></div>
            <div class="logo-sub">Monitor de Inteligencia Electoral Ciudadana</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 4. DATOS DE CONTENIDO ---
INFO_PARTIDOS = {
    "Agenda Ciudadana": {"Candidato": "Claudia Dobles", "Equipo": "Andrea Centeno, Luis F. Arauz", "Tendencia": "Centro-progresismo", "Estrategia": "Rebranding del PAC."},
    "Partido Liberación Nacional": {"Candidato": "Álvaro Ramos", "Equipo": "Karen Segura, Xinia Chaves", "Tendencia": "Socialdemocracia", "Estrategia": "Ruptura técnica con figuras tradicionales."},
    "Partido Unidad Social Cristiana": {"Candidato": "Juan Carlos Hidalgo", "Equipo": "Yolanda Fernández, Steven Barrantes", "Tendencia": "Socialcristianismo Liberal", "Estrategia": "Visión globalista + músculo municipal."},
    "Frente Amplio": {"Candidato": "Ariel Robles", "Equipo": "Margarita Salas, Guillermo Arroyo", "Tendencia": "Izquierda / Progresismo", "Estrategia": "Voto joven, DDHH y seguridad preventiva."},
    "Partido Liberal Progresista": {"Candidato": "Eliécer Feinzaig", "Equipo": "Tannia Molina, Gabriel Zamora", "Tendencia": "Liberalismo Económico", "Estrategia": "Reducción del Estado y cohesión liberal."},
    "Nueva República": {"Candidato": "Fabricio Alvarado", "Equipo": "David Segura, Rosalía Brown", "Tendencia": "Conservadurismo Religioso", "Estrategia": "Fórmula endogámica y voto duro."},
    "Pueblo Soberano": {"Candidato": "Laura Fernández", "Equipo": "Francisco Gamboa, Douglas Soto", "Tendencia": "Oficialismo Rodriguista", "Estrategia": "Continuidad del gobierno Chaves."},
    "Partido Integración Nacional": {"Candidato": "Luis Amador", "Equipo": "Jorge Borbón, Katya Berdugo", "Tendencia": "Populismo Tecnocrático", "Estrategia": "Capitalizar popularidad personal post-ruptura."},
    "Unidos Podemos": {"Candidato": "Natalia Díaz", "Equipo": "Jorge Ocampo, Luis Diego Vargas", "Tendencia": "Liberalismo / Ex-Oficialismo", "Estrategia": "Busca capturar un voto mixto."},
    "Progreso Social Democrático": {"Candidato": "Luz Mary Alpízar", "Equipo": "Frank Mc Kenzie, Maritza Bustamante", "Tendencia": "Oficialismo Estructural", "Estrategia": "Marca 2022 sin respaldo presidencial."},
    "Partido Nueva Generación": {"Candidato": "Fernando Zamora", "Equipo": "Lisbeth Quesada, Yeudy Araya", "Tendencia": "Derecha Conservadora", "Estrategia": "Importación de figura externa (Ex-PLN)."},
    "Avanza": {"Candidato": "Jose Miguel Aguilar", "Equipo": "Evita Arguedas, Marcela Ortiz", "Tendencia": "Derecha Populista", "Estrategia": "Narrativa 'Modelo Salvadoreño'."},
    "Esperanza Nacional": {"Candidato": "Claudio Alpízar", "Equipo": "Andrés Castillo, Nora González", "Tendencia": "Personalismo / Disidencia PLN", "Estrategia": "Crítica a cúpula tradicional."},
    "Justicia Social Costarricense": {"Candidato": "Walter Hernández", "Equipo": "Shirley González, Eduardo Rojas", "Tendencia": "Regionalismo", "Estrategia": "Fuerza en Limón y liderazgos locales."},
    "Centro Democrático y Social": {"Candidato": "Ana Virginia Calzada", "Equipo": "Oldemar Rodríguez, Heilen Díaz", "Tendencia": "Institucionalismo", "Estrategia": "Voto conservador institucional."},
    "Aquí Costa Rica Manda": {"Candidato": "Ronny Castillo", "Equipo": "Hazel Arias, William Anderson", "Tendencia": "Oficialismo Rodriguista", "Estrategia": "Célula espejo del chavismo."},
    "De la Clase Trabajadora": {"Candidato": "David Hernández", "Equipo": "Yeimy Castro, Obeth Morales", "Tendencia": "Izquierda Radical", "Estrategia": "Voto obrero y sindical exclusivo."},
    "Alianza Costa Rica Primero": {"Candidato": "Douglas Caamaño", "Equipo": "Lissa Freckleton, Carlos Moya", "Tendencia": "Localismo Independiente", "Estrategia": "Voto rural y caribeño."},
    "Unión Costarricense Democrática": {"Candidato": "Boris Molina", "Equipo": "José Morales, Maricela Morales", "Tendencia": "Personalista", "Estrategia": "Reciclaje de candidaturas internas."},
    "Esperanza y Libertad": {"Candidato": "Marco Rodríguez", "Equipo": "Carlos Palacios, Karla Romero", "Tendencia": "Tecnocracia Burocrática", "Estrategia": "Mandos medios ex-oficialistas."}
}

AUTOR_HTML = """
<div class='author-box'>
    <div class='author-header'>David Arturo Chavarría Camacho, M.Sc.</div>
    <div class='author-role'>Elaborado por</div>
    <p><b>Formación Académica Superior:</b></p>
    <ul>
        <li><b>Doctorado en Gestión Pública y Ciencias Empresariales</b> (PhD. Candidate) - <i>Instituto Centroamericano de Administración Pública (ICAP)</i></li>
        <li><b>Doctorado en Historia</b> (En curso) - <i>Universidad de Costa Rica (UCR)</i></li>
        <li><b>Maestría Académica en Historia</b> (Graduación de Honor, 2017) - <i>Universidad de Costa Rica</i></li>
        <li><b>Bachillerato en Historia</b> (2013) - <i>Universidad de Costa Rica</i></li>
        <li><b>Diplomado en Electrónica</b> (2008) - <i>Instituto Tecnológico de Costa Rica</i></li>
    </ul>
    <p><b>Trayectoria Profesional y Académica:</b></p>
    <ul>
        <li><b>Docente e Investigador (2014-2025):</b> Escuela de Historia y Escuela de Estudios Generales, Universidad de Costa Rica.</li>
        <li><b>Investigación Especializada:</b> Investigador en el Centro de Investigaciones Históricas de América Central (CIHAC). Especialista en Historia Digital, Ciencia, Tecnología y Sociedad (CTS).</li>
    </ul>
    <hr style="margin: 20px 0; border: 0; border-top: 1px solid #e2e8f0;">
    <p style='font-size:0.95rem; color:#64748b;'>📧 <b>Contacto Institucional:</b> david.chavarriacamacho@ucr.ac.cr</p>
</div>
"""

METODOLOGIA_TEXTO = """
### 1. Fundamentos Teóricos
El análisis se basa en la metodología del **Comparative Manifestos Project (CMP)**, el estándar académico global para el análisis de contenido político. Utilizamos técnicas de **Procesamiento de Lenguaje Natural (NLP)** para transformar texto no estructurado (PDFs) en datos cuantificables.

### 2. Variables y Algoritmos
* **Volumen de Propuestas:** Cantidad total de unidades de sentido (párrafos o bloques semánticos) extraídos tras la limpieza de ruido.
* **Análisis de Sentimiento (Polarity):** Se utiliza el algoritmo *TextBlob*. Asigna un valor de -1 (Muy Negativo) a +1 (Muy Positivo).
* **Complejidad Léxica:** Calculada mediante índice *Fernandez-Huerta*. Mide la dificultad de lectura.
* **Similitud del Coseno:** Distancia matemática entre los vectores de texto (TF-IDF) de cada partido.
"""

STOPWORDS_BASURA = {'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'las', 'un', 'una', 'por', 'con', 'no', 'su', 'sus', 'para', 'al', 'lo', 'como', 'más', 'pero', 'o', 'este', 'esta', 'son', 'ese', 'esa', 'si', 'sin', 'sobre', 'entre', 'ya', 'todo', 'toda', 'todos', 'todas', 'esta', 'estos', 'estas', 'nos', 'ni', 'muy', 'donde', 'que', 'qué', 'uno', 'dos', 'tres', 'parte', 'tiene', 'tienen', 'ser', 'es', 'fue', 'sido', 'está', 'están', 'sea', 'sean', 'ante', 'bajo', 'cabe', 'contra', 'desde', 'durante', 'hacia', 'hasta', 'mediante', 'para', 'según', 'so', 'tras', 'versus', 'vía', 'costa', 'rica', 'nacional', 'país', 'gobierno', 'plan', 'programa', 'propuesta', 'desarrollo', 'social', 'política', 'sistema', 'servicio', 'servicios', 'sector', 'sectores', 'hacer', 'cada', 'año', 'años', '2026', '2030', 'acciones', 'objetivo', 'estrategia', 'marco', 'nivel', 'forma', 'manera', 'caso', 'tema', 'temas', 'través', 'además', 'así', 'ello', 'bien', 'gran', 'mismo', 'fin', 'tal', 'vez', 'cual', 'cuales', 'debe', 'ser', 'parte', 'tipo', 'siguiente', 'san', 'josé', 'jose', 'república', 'central', 'general', 'materia', 'ámbito', 'punto', 'página', 'artículo', 'se', 'e', 'le', 'les', 'me', 'mi', 'mis', 'ha', 'han', 'hay', 'hubo', 'sino', 'porque', 'pues', 'aunque', 'mientras', 'cuando', 'donde', 'quien', 'quienes', 'ello', 'cuyo', 'cuya', 'donde', 'aquel', 'mediante', 'embargo', 'través', 'implementar', 'fortalecer'}

# --- 5. FUNCIONES ANALÍTICAS ROBUSTAS ---

def generar_insight_texto(df_sub, variable, nombre_variable, tipo="max"):
    if df_sub.empty: return "Sin datos suficientes para análisis."
    if tipo == "max":
        dato = df_sub.groupby('partido')[variable].mean().sort_values(ascending=False)
        if dato.empty: return ""
        top_p = dato.idxmax()
        val = dato.max()
        return f"🤖 **Análisis Automático:** El partido **{top_p}** lidera el índice de {nombre_variable} ({val:.2f})."
    elif tipo == "dist":
        est = df_sub['IDX_ESTATISMO'].mean()
        merc = df_sub['IDX_MERCADO'].mean()
        conclusion = "más Estatista" if est > merc else "más Pro-Mercado"
        return f"🤖 **Balance Ideológico:** En el agregado, la discusión es **{conclusion}**."

def generar_nube(texto):
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS_BASURA, colormap='tab10', min_font_size=10, max_words=60, regexp=r"\w+").generate(str(texto).lower())
    return wc

def interpretacion(texto):
    st.markdown(f"""<div class='interpretation-box'><span class='interpretation-title'>📘 Guía de Interpretación</span>{texto}</div>""", unsafe_allow_html=True)

# --- 6. INTERFAZ PRINCIPAL ---

if df is not None:
    
    # SIDEBAR
    with st.sidebar:
        st.header("Panel de Control")
        lista_partidos = sorted(df['partido'].unique())
        partidos = st.multiselect("Seleccione Partidos a Comparar:", lista_partidos, default=lista_partidos[:3] if len(lista_partidos)>2 else lista_partidos)
        
        st.divider()
        menu = st.radio("Navegación:", ["1. Visión Estratégica", "2. Psicometría del Discurso", "3. Brújula Ideológica", "4. Geopolítica", "5. Semántica Profunda", "6. Buscador Avanzado", "7. Perfiles Partidarios", "8. Metodología y Créditos"])
        
        if not partidos and menu not in ["7. Perfiles Partidarios", "8. Metodología y Créditos"]:
            st.warning("⚠️ Selecciona al menos un partido.")
            st.stop()

    # Filtrar datos
    df_m = df[df['partido'].isin(partidos)]

    # --- CONTENIDO ---
    if menu == "1. Visión Estratégica":
        st.markdown("## 🔭 Estrategia y Prioridades Temáticas")
        k1, k2, k3 = st.columns(3)
        with k1: st.metric("Volumen de Propuestas", f"{len(df_m):,}", help="Número total de bloques semánticos.")
        with k2: st.metric("Tema Dominante", df_m[df_m['TEMA_IA']!='OTROS']['TEMA_IA'].mode()[0], help="Categoría más repetida.")
        with k3: st.metric("Partidos Analizados", len(partidos))

        t1, t2, t3 = st.tabs(["Mapa de Calor", "Radar Comparativo", "Distribución Porcentual"])
        with t1:
            df_tree = df_m[df_m['TEMA_IA']!='OTROS']
            fig = px.treemap(df_tree, path=['partido', 'TEMA_IA'], color='TEMA_IA', height=600)
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("Muestra el <b>peso visual</b> de cada tema. Recuadros grandes = Mayor prioridad.")
        with t2:
            conteo = df_tree.groupby(['partido', 'TEMA_IA']).size().reset_index(name='n')
            conteo['pct'] = conteo.groupby('partido')['n'].transform(lambda x: 100 * x / x.sum())
            fig = px.line_polar(conteo, r='pct', theta='TEMA_IA', color='partido', line_close=True, height=600)
            fig.update_traces(fill='toself', opacity=0.2)
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("Forma puntiaguda = Especialización. Forma redonda = Generalismo.")
        with t3:
            fig = px.histogram(df_tree, x="partido", color="TEMA_IA", barnorm="percent", height=500)
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("Comparativa normalizada al 100%.")

    elif menu == "2. Psicometría del Discurso":
        st.markdown("## 🧠 Psicometría y Estilo")
        st.success(generar_insight_texto(df_m, 'COMPLEJIDAD', 'Complejidad Técnica'))
        c1, c2 = st.columns([2, 1])
        with c1:
            agg = df_m.groupby('partido').agg({'COMPLEJIDAD':'mean', 'SENTIMIENTO':'mean', 'longitud':'count'}).reset_index()
            fig = px.scatter(agg, x='COMPLEJIDAD', y='SENTIMIENTO', size='longitud', color='partido', text='partido', height=600)
            fig.update_traces(textposition='top center')
            fig.add_vline(x=agg['COMPLEJIDAD'].mean(), line_dash="dash", line_color="gray")
            fig.add_hline(y=agg['SENTIMIENTO'].mean(), line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("<b>Eje X:</b> Complejidad (Simple vs Técnico). <b>Eje Y:</b> Sentimiento (Crítico vs Positivo).")
        with c2:
            fig = px.box(df_m, x='partido', y='SUBJETIVIDAD', color='partido', points=False)
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("Grado de opinión vs hechos.")

    elif menu == "3. Brújula Ideológica":
        st.markdown("## 🧭 Posicionamiento Ideológico")
        st.success(generar_insight_texto(df_m, None, None, tipo="dist"))
        t1, t2 = st.tabs(["Modelo Económico", "Modelo Político"])
        idx_data = df_m.groupby('partido')[['IDX_ESTATISMO', 'IDX_MERCADO', 'IDX_SOCIAL', 'IDX_GLOBAL']].mean().reset_index()
        with t1:
            fig = px.scatter(idx_data, x='IDX_ESTATISMO', y='IDX_MERCADO', text='partido', size_max=60, color='partido', height=550)
            fig.update_traces(textposition='top center', marker=dict(size=20))
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("<b>Eje X:</b> Estado. <b>Eje Y:</b> Mercado.")
        with t2:
            fig = px.scatter(idx_data, x='IDX_SOCIAL', y='IDX_GLOBAL', text='partido', size_max=60, color='partido', height=550)
            fig.update_traces(textposition='top center', marker=dict(size=20))
            st.plotly_chart(fig, use_container_width=True)
            interpretacion("<b>Eje X:</b> Social. <b>Eje Y:</b> Global.")

    elif menu == "4. Geopolítica":
        st.markdown("## 🌍 Instituciones y Territorio")
        INSTITUCIONES = ["CCSS", "MEP", "MOPT", "ICE", "INS", "AyA", "IMAS", "INA", "UCR", "Hacienda", "Banco Central", "Contraloría", "Poder Judicial", "OIJ", "Fuerza Pública", "RECOPE", "SINAC", "Sala IV"]
        LUGARES = ['Guanacaste', 'Limón', 'Puntarenas', 'Cartago', 'Heredia', 'Alajuela', 'San José', 'Zona Norte', 'Zona Sur', 'GAM', 'Estados Unidos', 'China', 'Europa', 'OCDE']
        
        def count_entities(df_sub, lista):
            res = []
            for p in df_sub['partido'].unique():
                txt = " ".join(df_sub[df_sub['partido']==p]['texto'].astype(str)).lower()
                for i in lista:
                    c = txt.count(i.lower())
                    if c > 0: res.append({'Partido': p, 'Entidad': i, 'Menciones': c})
            return pd.DataFrame(res)

        c1, c2 = st.columns(2)
        with c1:
            df_i = count_entities(df_m, INSTITUCIONES)
            if not df_i.empty:
                fig = px.bar(df_i, y='Entidad', x='Menciones', color='Partido', orientation='h', barmode='group', height=600, title="Instituciones Públicas")
                st.plotly_chart(fig, use_container_width=True)
                interpretacion("Foco Burocrático: Instituciones más mencionadas.")
            else: st.warning("Sin datos.")
        with c2:
            df_l = count_entities(df_m, LUGARES)
            if not df_l.empty:
                fig = px.bar(df_l, x='Entidad', y='Menciones', color='Partido', title="Foco Territorial")
                st.plotly_chart(fig, use_container_width=True)
                interpretacion("Foco Geográfico: Regiones prioritarias.")
            else: st.warning("Sin datos.")

    elif menu == "5. Semántica Profunda":
        st.markdown("## 🗣️ Lenguaje y Conceptos")
        t1, t2 = st.tabs(["Nubes de Palabras", "Similitud Matemática"])
        with t1:
            col_sel, col_wc = st.columns([1, 3])
            with col_sel: p_sel = st.radio("Ver nube de:", partidos)
            with col_wc:
                txt_p = " ".join(df_m[df_m['partido']==p_sel]['texto'].astype(str))
                wc = generar_nube(txt_p)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
            interpretacion("Palabras más repetidas (Obsesión discursiva).")
        with t2:
            grouped = df_m.groupby('partido')['texto'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
            if len(grouped) > 1:
                tfidf = TfidfVectorizer(stop_words=list(STOPWORDS_BASURA))
                matriz = tfidf.fit_transform(grouped['texto'])
                sim = cosine_similarity(matriz)
                fig = px.imshow(sim, x=grouped['partido'], y=grouped['partido'], text_auto='.2f', color_continuous_scale='Blues', height=600)
                st.plotly_chart(fig, use_container_width=True)
                interpretacion("1.0 = Idénticos. 0.0 = Opuestos.")

    elif menu == "6. Buscador Avanzado":
        st.markdown("## 🔎 Explorador Semántico")
        q = st.text_input("Buscar:", placeholder="Ej: corrupción, deuda...")
        if q:
            res = df_m[df_m['texto'].str.contains(q, case=False, na=False)]
            if not res.empty:
                st.success(f"Encontrados: {len(res)}")
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig_pie = px.pie(res, names='partido', title=f"Share of Voice: '{q}'", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2: st.dataframe(res[['partido', 'TEMA_IA', 'texto']], use_container_width=True, height=400)
            else: st.warning("No encontrado.")

    elif menu == "7. Perfiles Partidarios":
        st.markdown("## 🗳️ Fichas Técnicas 2026")
        partidos_completos = sorted(list(INFO_PARTIDOS.keys()))
        cols = st.columns(2)
        for i, partido in enumerate(partidos_completos):
            with cols[i % 2]:
                info = INFO_PARTIDOS.get(partido)
                st.markdown(f"""
                <div class='party-card'>
                    <div class='party-header'>{partido}</div>
                    <div class='party-sub'>{info['Tendencia']}</div>
                    <div class='party-body'><b>👤 Candidato:</b> {info['Candidato']}<br><b>👥 Equipo:</b> {info['Equipo']}</div>
                    <div class='strategic-data'><b>⚡ Dato Estratégico:</b> {info['Estrategia']}</div>
                </div>""", unsafe_allow_html=True)

    elif menu == "8. Metodología y Créditos":
        st.markdown("## 🧬 Ficha Técnica")
        t1, t2 = st.tabs(["🔬 Metodología", "👨‍💻 Autor"])
        with t1: st.markdown(METODOLOGIA_TEXTO)
        with t2: st.markdown(AUTOR_HTML, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer-container">
        <p>© 2025 VOTO 360°. Análisis basado en datos públicos.</p>
    </div>
    """, unsafe_allow_html=True)
