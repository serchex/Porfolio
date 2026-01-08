export default function ProyectoChido() {
  return (
<div className="bg-[#f7f7f7] text-[#222] min-h-screen">
      <header className="bg-[#1a73e8] text-white p-10 text-center">
        <h1 className="text-4xl m-0">Customer Segmentation & Sales Analysis Dashboard</h1>
        <p>Análisis RFM y Clustering KMeans aplicado a datos de ventas</p>
      </header>

      <section className="max-w-[1200px] mx-auto my-10 p-6 bg-white rounded-xl shadow-lg">
        <h2 className="text-2xl mb-2 text-[#333]">1. Customer Segmentation Overview</h2>
        <p className="text-[17px] leading-7 text-[#444]">
          Este dashboard presenta la segmentación de clientes basada en métricas RFM
          (Recency, Frequency, Monetary) y un modelo de clustering. Los gráficos permiten
          visualizar el volumen de clientes por segmento y por cluster, así como métricas
          clave como número total de clientes y ventas totales.
        </p>

        <div className="w-full text-center my-8">
          <img
            src="/customers/imagen1.png"
            alt="Customer Segmentation Overview"
            className="w-[95%] rounded-lg"
          />
        </div>
        <div className="w-full text-center my-8">
          <img
            src="/customers/imagen2.png"
            alt="Customer Segmentation Overview"
            className="w-[95%] rounded-lg"
          />
        </div>
        <p className="text-[17px] leading-7 text-[#444]">
            Como puedes ver, los clientes VIP se reparten en diferentes clusters ya que pueden haber clientes VIP PREMIUM, VIPs que compran en grandes cantidades pero pocas veces, VIPs que compran poco pero compran productos de alto valor etc..
        </p>
        <div className="w-full text-center my-8">
          <img
            src="/customers/imagen3.png"
            alt="Customer Segmentation Overview"
            className="w-[95%] rounded-lg"
          />
        </div>
        <p className="text-[17px] leading-7 text-[#444]">
            Aqui, los clientes que estan en riesgo NO estan dentro de los clusters VIP o Leales/Frecuentes, esto es buena señal ya que mas de la mitad de los clientes estan dentro de estos clusters y los clientes que estan en riesgo de abandonar la tienda, son clientes que compran esporadicamente o una sola vez.
        </p>
      </section>

      <section className="max-w-[1200px] mx-auto my-10 p-6 bg-white rounded-xl shadow-lg">
        <h2 className="text-2xl mb-2 text-[#333]">2. Sales Analysis</h2>
        <p className="text-[17px] leading-7 text-[#444]">
          El análisis de ventas muestra el comportamiento mensual por cluster,
          identificando estacionalidad y patrones de compra. También se presenta una gráfica
          detallada del gasto por categoría de producto.
        </p>

        <div className="w-full text-center my-8">
          <img
            src="/customers/imagen4.png"
            alt="Sales Analysis"
            className="w-[95%] rounded-lg"
          />
        </div>
      </section>

      <section className="max-w-[1200px] mx-auto my-10 p-6 bg-white rounded-xl shadow-lg">
        <h2 className="text-2xl mb-2 text-[#333]">3. Cluster’s Name Analysis</h2>
        <p className="text-[17px] leading-7 text-[#444]">
          Para interpretar correctamente los clusters generados por el modelo, se analizaron
          promedios de RFM y patrones de compra por temporada y categoría. A partir de este análisis
          se definieron nombres claros y representativos:
        </p>

        <ul className="list-disc pl-6 text-[#444]">
          <li><b>Cluster 2 – VIP Premium:</b> clientes de alto valor, alta frecuencia y alto gasto.</li>
          <li><b>Cluster 3 – Leales/Frecuentes:</b> clientes que compran seguido y mantienen gasto estable.</li>
          <li><b>Cluster 0 – Regulares:</b> mayoría estable, con gasto medio y frecuencia moderada.</li>
          <li><b>Cluster 1 – Esporádicos:</b> clientes inactivos o de bajo compromiso.</li>
        </ul>

        <div className="w-full text-center my-8">
          <img
            src="/customers/imagen5.png"
            alt="Cluster Name Analysis"
            className="w-[95%] rounded-lg"
          />
        </div>
      </section>

      <section className="max-w-[1200px] mx-auto my-10 p-6 bg-white rounded-xl shadow-lg">
        <h2 className="text-2xl mb-2 text-[#333]">5. Creacion del Modelo LDA (Python)</h2>
        <p className="text-[17px] leading-7 text-[#444]">
          En esta seccion arme el modelo LDA (Latent Dirichlet Allocation) y Count Vectorizer para formar palabras relacionadas con el data set y agrupar los productos por categorias
        </p>
        <div className="flex justify-center items-center relative">
            <div className="h-[200px] w-[500px] bg-white border-2 border-[#333] rounded-[15px] overflow-hidden">
                <div className="h-[30px] bg-[#333] flex items-center justify-between px-2 shadow-md">
                    <p className="text-white font-bold m-0 p-0">LDA.py</p>

                    <div className="flex items-center">
                        <div className="w-[10px] h-[10px] rounded-full m-[5px] bg-green-500 hover:bg-green-600 cursor-pointer"></div>
                        <div className="w-[10px] h-[10px] rounded-full m-[5px] bg-yellow-400 hover:bg-yellow-500 cursor-pointer"></div>
                        <div className="w-[10px] h-[10px] rounded-full m-[5px] bg-red-600 hover:bg-red-700 cursor-pointer"></div>
                    </div>
                </div>

                {/* Console */}
                <div className="w-full h-[calc(100%-30px)] bg-black text-white overflow-auto">
                    <pre className="m-0 p-2 text-[15px]">
                        <code className="text-white">
                    {`import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib, numpy as np

# Carga de datos y preprocesamiento
df = pd.read_excel('Online retail.xlsx')
df = df.drop_duplicates()
df.dropna(subset=['CustomerID', 'Description'], inplace=True)
df['Description'] = df['Description'].str.lower()
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

print('Transformando el texto a conteos de palabras')
count_vectorizer = CountVectorizer(
    stop_words='english',
    max_df=0.95,
    min_df=2
)

count_matrix = count_vectorizer.fit_transform(df['Description'])
print(f'Forma de la matriz de conteo: {count_matrix.shape}')

n_topics = 100
print(f'Aplicando LDA para encontrar {n_topics} temas')
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    learning_method='batch' # para datasets grandes
)
lda_result = lda.fit_transform(count_matrix)
print(lda_result)

# guardar modelo LDA entrenado
joblib.dump(lda, 'lda_model.joblib')
print('Modelo LDA guardado en lda_model.joblib')

# guardar el countvectorizer para poder usarlo despues
joblib.dump(count_vectorizer, 'count_vectorizer.joblib')
print('CountVectorizer guardado en count_vectirizer.joblib')

df['main_topic'] = np.argmax(lda_result, axis=1)
print(df['main_topic'])
# guardar el DataFrame con la nueva columna 'main_topic'
df.to_pickle('online_retail_with_topics.pkl')
print({'DataFrame con temas guardados en online_retail_with_topics.pkl'})


def mostrar_temas(modelo_lda, vectorizador, n_palabras_top):

    #  Obtener el vocabulario (las palabras) del CountVectorizer
    palabras = vectorizador.get_feature_names_out()
    
    #  Iterar sobre cada tema en los componentes del modelo LDA
    for indice_tema, distribucion_tema in enumerate(modelo_lda.components_):
        # Imprimir el encabezado del tema
        print(f"--- Tema #{indice_tema} ---")
        
        #  Obtener los índices de las palabras más importantes para este tema
        # .argsort() devuelve los índices que ordenarían el array
        # [:-n_palabras_top - 1:-1] para obtener los últimos n índices (los más altos) en orden descendente
        indices_palabras_top = distribucion_tema.argsort()[:-n_palabras_top - 1:-1]
        
        #  Mapear los índices a las palabras reales y unirlas en un string
        palabras_top = [palabras[i] for i in indices_palabras_top]
        print(" ".join(palabras_top))

# 
# Llama a la función con el modelo,el vectorizador y el número de palabras para ver
n_palabras_a_mostrar = 15
print(f"\nMostrando las {n_palabras_a_mostrar} palabras más importantes por tema:\n")
mostrar_temas(lda, count_vectorizer, n_palabras_a_mostrar)
`}
                        </code>
                    </pre>
                </div>
            </div>
        </div>
      </section>
      <section className="max-w-[1200px] mx-auto my-10 p-6 bg-white rounded-xl shadow-lg">
        <h2 className="text-2xl mb-2 text-[#333]">6. Limpieza, Normalizacion, Mapeo de palabras y Clusterizacion (Python)</h2>
        <p className="text-[17px] leading-7 text-[#444]">
          Este codigo muestra como se realizo la limpieza, la normalizacion con StandardScaler, el mapeo de las palabras retornadas por el modelo LDA, la separacion entre articulos cancelados, temporadas, descuentos, graficas con plotly, las operaciones para sacar el RFM y por supuesto, el clusterizado con Kmeans++ y elbow method para optimizar el numero de clusters retornados.
        </p>
        <div className="flex justify-center items-center relative">
            <div className="h-[200px] w-[500px] bg-white border-2 border-[#333] rounded-[15px] overflow-hidden">
                <div className="h-[30px] bg-[#333] flex items-center justify-between px-2 shadow-md">
                    <p className="text-white font-bold m-0 p-0">LDA.py</p>

                    <div className="flex items-center">
                        <div className="w-[10px] h-[10px] rounded-full m-[5px] bg-green-500 hover:bg-green-600 cursor-pointer"></div>
                        <div className="w-[10px] h-[10px] rounded-full m-[5px] bg-yellow-400 hover:bg-yellow-500 cursor-pointer"></div>
                        <div className="w-[10px] h-[10px] rounded-full m-[5px] bg-red-600 hover:bg-red-700 cursor-pointer"></div>
                    </div>
                </div>

                {/* Console */}
                <div className="w-full h-[calc(100%-30px)] bg-black text-white overflow-auto">
                    <pre className="m-0 p-2 text-[15px]">
                        <code className="text-white">
                    {`from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os, joblib
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import lda_topic_to_category

pio.renderers.default = 'browser'

# cargar data
lda_file = 'data/lda_model.joblib'
df_file = 'data/online_retail_with_topics.pkl'
vectorizer_file = 'data/count_vectorizer.joblib'

if os.path.exists(lda_file) and os.path.exists(df_file) and os.path.exists(vectorizer_file):
    print('Cargando modelo LDA y DataFrame desde archivos')
    lda = joblib.load(lda_file)
    count_vectorizer = joblib.load(vectorizer_file)
    df = pd.read_pickle(df_file)
else:
    print('\nArchivos no encontrados.\n')

print(f'\nFilas totales antes de la separacion {len(df)}')
df_returns = df[df['InvoiceNo'].astype(str).str.startswith('C')].copy()
df_returns['Quantity'] = df['Quantity'].abs()
df_returns['TotalPrice'] = df['TotalPrice'].abs()

df_sales = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
df_sales = df_sales[(df_sales['UnitPrice'] > 0) & (df_sales['Quantity'] > 0)] # quito valores nulos

print(f"Filas de ventas para analizar: {len(df_sales)}")
print(f"Filas de devoluciones para analizar: {len(df_returns)}")
# Mapeo de temas LDA a categorías significativas

lda_topic_to_category = lda_topic_to_category.lda_topic_to_category

#print('MAIN TOPIC: \n', df['main_topic'])
# mapeo con categorias
df_sales['Product_category'] = df_sales['main_topic'].map(lda_topic_to_category)
df_returns['Product_category'] = df_returns['main_topic'].map(lda_topic_to_category)


print('\n--- Palabras clave para cada tema ---')
feature_names = count_vectorizer.get_feature_names_out()

print('\nPalabras clave del tema 3 (Decoración navideña vintage)\n')
print(' '.join([feature_names[i] for i in lda.components_[0].argsort()[:-30:-1]]))

# Grafico cancelled Items
def graf_cancelled_items():
    cancelled_items_count = df_returns['Product_category'].value_counts().nlargest(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=cancelled_items_count.values, y=cancelled_items_count.index)
    plt.title('Top 10 de Artículos con Mayor Tasa de Cancelación')
    plt.xlabel('Número de Cancelaciones')
    plt.ylabel('Descripción del Artículo')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

def monthly_cancellation():
    # 'InvoiceDate' a formato de fecha y hora
    df_returns['InvoiceDate'] = pd.to_datetime(df_returns['InvoiceDate'])

    # Agrupa por mes 
    df_returns['Month'] = df_returns['InvoiceDate'].dt.to_period('M')
    monthly_cancellations = df_returns.groupby('Month').size()

    # gráfico
    plt.figure(figsize=(14, 7))
    monthly_cancellations.plot(kind='line', marker='o', color='red')
    plt.title('Tendencia de Cancelaciones por Mes')
    plt.xlabel('Mes')
    plt.ylabel('Número de Cancelaciones')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def display_topics(model, feature_names, no_top_words):

    for topic_idx, topic in enumerate(model.components_):
        print(f"Tema {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print("---")

no_top_words = 15 # Mostrar las 15 palabras más importantes por tema
#display_topics(lda, feature_names, no_top_words)

# Analizar los resultados
print('\n--- Resultados del análisis de temas ---')
# Contar cuantas descripciones hay en cada tema principal
print(df_sales['main_topic'].value_counts().sort_index())

print("\n--- Conteo de productos por categoría ---")
print(df_sales['Product_category'].value_counts())
print(df_sales[['Description','main_topic','Product_category']].head())

df_sales['InvoiceDate'] = pd.to_datetime(df_sales['InvoiceDate'])

# Extraer mes y año
df_sales['Month'] = df_sales['InvoiceDate'].dt.month
df_sales['Year'] = df_sales['InvoiceDate'].dt.year
df_sales['Hour'] = df_sales['InvoiceDate'].dt.hour

# Crear una columna de temporada

def getSeason(month):
    if month in [12,1,2]:
        return 'Winter'
    elif month in [3,4,5]:
        return 'Spring'
    elif month in [6,7,8]:
        return 'Summer'
    else:
        return 'Autumn'
    
df_sales['Season'] = df_sales['Month'].apply(getSeason)

# Analizar los descuentos
df_sales['Discount'] = (df_sales['UnitPrice'] * df_sales['Quantity']) - df_sales['TotalPrice']
df_sales['Has_Discount'] = df_sales['Discount'] > 0

sales_by_season = df_sales.groupby('Season')['TotalPrice'].sum().sort_values(ascending=False)
print('\nVentas por temporada: ', sales_by_season)

# Analizar la frecuencia de descuentos por mes
discounts_by_month = df_sales.groupby('Month')['Has_Discount'].mean()
print('\nFrecuencia de descuentos por mes:\n', discounts_by_month)

# gasto total por cliente y categoria de producto
customer_frecuency = df_sales.groupby(['CustomerID', 'Product_category'])['InvoiceNo'].count().unstack(fill_value=0)
customer_frecuency_cleaned = customer_frecuency.clip(lower=0)
customer_frecuency_log = np.log1p(customer_frecuency_cleaned)
scaler = StandardScaler()

customer_frecuency_scaled = scaler.fit_transform(customer_frecuency_log)

def plot_elbow_method(data, max_clusters=20):
    print(f'Calculando el metodo del codo para 1 a {max_clusters} clusters...')
    inertia_values = []

    for i in range(1, max_clusters + 1):
        kmeans = KMeans(
            n_clusters=i,
            init="k-means++",   # aquí está la magia
            n_init=10,          # corre 10 inicializaciones y elige la mejor
            max_iter=300,
            tol=1e-4,
            algorithm="elkan",  # acelera con distancia euclidiana
            random_state=42
        )
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(10,6))
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='--')
    plt.xlabel('Numero de clusters (k)')
    plt.ylabel('Inercia')
    plt.grid(True)
    plt.show()

#plot_elbow_method(customer_frecuency_scaled)
kmeans = KMeans(
    n_clusters=4,
    init="k-means++",   
    n_init=10,          # corre 10 inicializaciones y elige la mejor
    max_iter=300,
    tol=1e-4,
    algorithm="elkan",  # distancia euclidiana
    random_state=42
)
kmeans.fit(customer_frecuency_scaled)

# Asignar cada cliente con un cluster
customer_frecuency['Custom_Cluster'] = kmeans.labels_
df_sales['Customer_Cluster'] = df_sales['CustomerID'].map(customer_frecuency['Custom_Cluster'])

print('\nNumero de clientes por cluster:\n', customer_frecuency['Custom_Cluster'].value_counts())

# 1. Reducción de la dimensionalidad para la visualización de clientes
print('Reduciendo la dimensionalidad de los datos de clientes con PCA...')
pca_3d_customers = PCA(n_components=3, random_state=42)
principal_components_3d = pca_3d_customers.fit_transform(customer_frecuency_scaled)

# 2. Crear un DataFrame para la visualización
df_sales_plot_customers_3d = pd.DataFrame(data=principal_components_3d,
                                    columns=['pc1', 'pc2', 'pc3'])

# 3. Añadir la columna de clúster al nuevo DataFrame
# Usamos customer_frecuency['Custom_Cluster'] para que los índices coincidan
df_sales_plot_customers_3d['Customer_Cluster'] = customer_frecuency['Custom_Cluster'].values

def check_log1p():
    #  Ejemplo de cómo se vería con los datos originales y con log1p

    feature_name = 'Decoración navideña vintage' # Reemplaza con una categoría de tu DataFrame

    if feature_name in customer_frecuency.columns:
        original_data = customer_frecuency[feature_name]

        # transformación log1p a los datos originales
        log_transformed_data = np.log1p(original_data)


        fig_original = px.histogram(original_data, nbins=50, title=f'Distribución original de {feature_name}')
        fig_original.show()

        fig_log = px.histogram(log_transformed_data, nbins=50, title=f'Distribución con log1p de {feature_name}')
        fig_log.show()

def graficas():
    # 3D con Plotly
    print('Graficando los clústeres de clientes en 3D...')
    fig1 = px.scatter_3d(df_sales_plot_customers_3d,
                        x='pc1',
                        y='pc2',
                        z='pc3',
                        color='Customer_Cluster',
                        title='Clustering de Clientes con K-Means (Visualización 3D con PCA)',
                        labels={'pc1': 'Componente Principal 1', 'pc2': 'Componente Principal 2', 'pc3': 'Componente Principal 3'})

    fig1.show()
    pio.write_html(fig1, file="plots/cluster_clientes_3d.html", auto_open=True)
    print('\nAnalisis')
    cluster_summary = df_sales.groupby('Customer_Cluster')['Product_category'].value_counts(normalize=True).unstack(fill_value=0)
    print('\nDistribucion de categorias por cluster:\n', cluster_summary)

    cluster_summary_plot = cluster_summary.reset_index()

    melted_df_sales = cluster_summary_plot.melt(
        id_vars=['Customer_Cluster'],
        var_name='Product_category',
        value_name='Proportion'
    )

    # Crear el gráfico de barras apiladas
    fig2 = px.bar(
        melted_df_sales,
        x='Customer_Cluster',
        y='Proportion',
        color='Product_category',
        title='Composición de Categorías de Productos por Clúster de Clientes',
        labels={'Customer_Cluster': 'Clúster de Clientes', 'Proportion': 'Proporción de Compras'},
        hover_data={'Proportion': ':.2%'} # Formato de porcentaje en el hover
    )

    fig2.show()
    pio.write_html(fig2, file="plots/categorias_productos_per_Cluster.html", auto_open=True)
    season_category_sales = (
        df_sales.groupby(['Season', 'Product_category'])['TotalPrice'].sum().unstack(fill_value=0)
    )

    scaler = StandardScaler()
    season_scaled = scaler.fit_transform(season_category_sales)

    pca = PCA(n_components=3, random_state=42)
    season_pca = pca.fit_transform(season_scaled)

    season_pca_df_sales = pd.DataFrame(season_pca, columns=['pc1', 'pc2', 'pc3'])
    season_pca_df_sales['Season'] = season_category_sales.index

    # Gráfica 3D
    fig_3d = px.scatter_3d(
        season_pca_df_sales,
        x='pc1', y='pc2', z='pc3',
        color='Season',
        title='Patrones de ventas por temporada (PCA 3D)',
        labels={'pc1': 'Componente 1', 'pc2': 'Componente 2', 'pc3': 'Componente 3'}
    )
    fig_3d.show()
    pio.write_html(fig_3d, file="plots/ventas_por_temporada.html", auto_open=True)
    # 1. Agrupar y sumar las ventas por temporada y categoría
    season_category_sales = df_sales.groupby(['Season', 'Product_category'])['TotalPrice'].sum()

    # 2Calcular la proporción dividiendo la venta de cada categoría entre el total de su temporada
    season_totals = season_category_sales.groupby('Season').transform('sum')
    season_category_prop_series = season_category_sales / season_totals

    # Convertir la Serie resultante a un DataFrame limpio para la gráfica
    season_category_prop = season_category_prop_series.reset_index(name='Proportion')

    fig_bar = px.bar(
        season_category_prop,
        x='Season',
        y='Proportion',
        color='Product_category',
        title='Proporción de ventas por categoría en cada temporada',
        labels={'Proportion': 'Proporción de ventas'},
        hover_data={'Proportion': ':.2%'}
    )
    fig_bar.show()
    pio.write_html(fig_bar, file="plots/ventas_categoria_temporada.html", auto_open=True)

# 1) Normalizar devoluciones
df_returns_signed = df_returns.copy()
df_returns_signed['Quantity']   = -df_returns_signed['Quantity'].abs()
df_returns_signed['TotalPrice'] = -df_returns_signed['TotalPrice'].abs()

# Asegurar tipos de fecha
df_sales['InvoiceDate'] = pd.to_datetime(df_sales['InvoiceDate'], errors='coerce')
df_returns_signed['InvoiceDate'] = pd.to_datetime(df_returns_signed['InvoiceDate'], errors='coerce')

# 2) Transacciones unificadas
df_tx = pd.concat([df_sales[['InvoiceNo','InvoiceDate','CustomerID','Quantity','TotalPrice']],
                   df_returns_signed[['InvoiceNo','InvoiceDate','CustomerID','Quantity','TotalPrice']]],
                  ignore_index=True)

# Limpieza básica
df_tx = df_tx.dropna(subset=['CustomerID', 'InvoiceDate'])
df_tx = df_tx[df_tx['TotalPrice'].notna()]

# 3) Fecha de referencia (para Recency)
ref_date = df_tx['InvoiceDate'].max() + pd.Timedelta(days=1)

# 4) Frequency: facturas únicas solo de ventas
#InvoiceNo que NO empieza con 'C'
mask_sales_invoices = ~df_tx['InvoiceNo'].astype(str).str.startswith('C')
freq = (df_tx.loc[mask_sales_invoices]
          .groupby('CustomerID')['InvoiceNo']
          .nunique()
          .rename('Frequency'))

# 5) Monetary: suma neta ventas - devoluciones
monetary = (df_tx.groupby('CustomerID')['TotalPrice']
              .sum()
              .rename('Monetary'))

# 6) Recency: días desde última compra (tomando cualquier transacción con fecha)
last_purchase = (df_tx.groupby('CustomerID')['InvoiceDate']
                   .max()
                   .rename('LastPurchase'))
recency = (ref_date - last_purchase).dt.days.rename('Recency')

# 7) DataFrame RFM
rfm = pd.concat([recency, freq, monetary], axis=1).fillna({'Frequency':0, 'Monetary':0})

# rfm = rfm[rfm['Monetary'] > 0]

# 8) Scores 1-5 por quintiles
def quintile_scores(series, reverse=False):
    s = series.copy()
    # En Monetary y Frequency, mayor es mejor. En Recency, menor es mejor (reverse=True).
    if reverse:
        s = -s
    try:
        return pd.qcut(s.rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    except ValueError:
        # Fallback cuando hay demasiados empates
        return pd.cut(s.rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)

def log1p_safe(s: pd.Series) -> pd.Series:
    s = s.astype(float).copy()
    mn = s.min()
    if mn <= -1:
       
        s = s - mn + 1e-6
    # por redondeos, evito cruzar -1
    s = s.clip(lower=-0.999999)
    return np.log1p(s)

rfm['Recency_log']    = np.log1p(rfm['Recency'])
rfm['Frequency_log']  = np.log1p(rfm['Frequency'])
rfm['Monetary_log']   = log1p_safe(rfm['Monetary'])

rfm['R_Score'] = quintile_scores(rfm['Recency_log'], reverse=True)     # menor recency = mejor score
rfm['F_Score'] = quintile_scores(rfm['Frequency_log'], reverse=False)  # más compras = mejor
rfm['M_Score'] = quintile_scores(rfm['Monetary_log'], reverse=False)   # más gasto = mejor
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# 9) Segmentos simples
def segment_row(row):
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'VIP'
    if r >= 4 and f >= 3:
        return 'Leal'
    if r <= 2 and f <= 2 and m <= 2:
        return 'Dormido/Perdido'
    if r <= 2 and (f >= 3 or m >= 3):
        return 'En riesgo'
    if f == 1 and m <= 2:
        return 'Una-vez/Explorador'
    return 'Medio'

rfm['Segment'] = rfm.apply(segment_row, axis=1)

# customer_frecuency['Custom_Cluster'] existe en tu script y su índice son CustomerID
rfm = rfm.join(customer_frecuency[['Custom_Cluster']], how='left')


# a) Top clientes por Monetary dentro de cada Segmento
top_monetary = (rfm.reset_index()
                  .sort_values(['Segment','Monetary'], ascending=[True, False])
                  .groupby('Segment')
                  .head(10))

# b) Resumen de KPIs por segmento
segment_summary = (rfm.groupby('Segment')
                     .agg(customers=('Recency','size'),
                          avg_recency=('Recency','mean'),
                          avg_freq=('Frequency','mean'),
                          avg_monetary=('Monetary','mean'))
                     .sort_values('avg_monetary', ascending=False))

print('\nRFM listo. Muestra:\n', rfm.head())
print('\nTop 10 por segmento (Monetary):\n', top_monetary.head(20))
print('\nResumen por segmento:\n', segment_summary)

def _prep_rfm_for_plot(rfm: pd.DataFrame,
                       scale: str = "none"  # "none" | "minmax" | "log1p"
                      ) -> pd.DataFrame:
    df = rfm.copy()

    # Validaciones mínimas
    needed = {'Recency','Frequency','Monetary'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas para el 3D: {missing}")

    # Opcional: log o escalado para que R no aplaste F y M
    if scale == "log1p":
        for col in ['Recency','Frequency','Monetary']:
            # Evitar negativos si tienes Monetary neto con devoluciones
            min_val = df[col].min()
            if min_val < 0:
                df[col] = df[col] - min_val
            df[col] = np.log1p(df[col])
    elif scale == "minmax":
        for col in ['Recency','Frequency','Monetary']:
            c = df[col].astype(float)
            mn, mx = c.min(), c.max()
            df[col] = (c - mn) / (mx - mn) if mx > mn else 0.0

    # Etiquetas amigables en hover
    if 'CustomerID' not in df.columns:
        df['CustomerID'] = df.index

    # Asegurar tipos
    if 'Segment' not in df.columns:
        df['Segment'] = 'SinSegmento'
    if 'Custom_Cluster' not in df.columns:
        df['Custom_Cluster'] = -1

    return df


def plot_3d_segment_vs_cluster(rfm: pd.DataFrame,
                               scale: str = "none",
                               save_html: str | None = None):
    """
    Color por Segment y símbolo por Custom_Cluster para ver ambos a la vez.
    """
    df = _prep_rfm_for_plot(rfm, scale=scale)
    # Limitar símbolos si hay demasiados clusters
    unique_clusters = df['Custom_Cluster'].astype(str).unique()
    fig = px.scatter_3d(
        df,
        x='Recency', y='Frequency', z='Monetary',
        color='Segment',
        symbol=df['Custom_Cluster'].astype(str) if len(unique_clusters) <= 10 else None,
        hover_data=['CustomerID','Recency','Frequency','Monetary','Custom_Cluster'],
        opacity=0.85
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        title=f"RFM 3D Segment vs Cluster (scale={scale})",
        scene=dict(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            zaxis_title='Monetary'
        ),
        legend_title_text='Segment'
    )
    if save_html:
        fig.write_html(save_html, include_plotlyjs='cdn')
    return fig

fig3 = plot_3d_segment_vs_cluster(rfm, scale="log1p")
#graficas()

# EXPORT TO POWERBI

df_sales[['CustomerID','Product_category','UnitPrice','Quantity','TotalPrice','Season','Has_Discount','InvoiceDate']].to_csv('df_sales_final.csv',index=False)

rfm_reset = rfm.reset_index()
rfm_reset[['CustomerID','Recency','Frequency','Monetary','R_Score','F_Score','M_Score','Segment','Custom_Cluster']].to_csv("rfm_final.csv", index=False)

customer_frecuency[['Custom_Cluster']].to_csv("clusters_final.csv", index=True)`}
                        </code>
                    </pre>
                </div>
            </div>
        </div>
      </section>
      <footer className="text-center p-6 text-sm text-[#777]">
        Dashboard creado por Sergio Gonzalez • Power BI • Python
      </footer>
    </div>
  );
}