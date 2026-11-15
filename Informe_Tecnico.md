# Informe técnico

## Optimización inferencial del rendimiento agrícola y eficiencia de riego en cultivos de invernadero

**Materia:** Análisis y Exploración de Datos  
**Clave:** INC-2505  
**Contexto:** Empresa agrotecnológica con invernaderos inteligentes de jitomate y aguacate  
**Entregables de software:** `app.py` y `requirements.txt`  
**Fecha:** noviembre de 2026

## 1. Introducción y planteamiento del problema

La operación de un invernadero tecnificado requiere asignar agua y fertilizantes de manera eficiente, y distinguir el efecto de una nueva práctica agronómica de la variación normal entre parcelas. Un aumento de producción por sí solo no basta para justificar una inversión: debe estimarse con un método reproducible y contrastarse estadísticamente.

Este proyecto construye un flujo analítico para 100 parcelas sintéticas. Sus objetivos son (1) describir la asociación entre riego, fertilizante, temperatura y rendimiento; (2) ajustar un modelo lineal múltiple con riego y fertilizante como predictores; y (3) evaluar si el rendimiento medio observado bajo el bio-estimulante B supera al del tratamiento estándar A. El conjunto simulado demuestra el procedimiento; no sustituye un ensayo agronómico en campo.

## 2. Metodología y arquitectura de datos: DuckDB

La aplicación abre una instancia de DuckDB en memoria y ejecuta `SELECT setseed(0.42);` antes de invocar el generador aleatorio. Una consulta SQL materializa cinco extracciones aleatorias independientes para cada identificador consecutivo de parcela (1–100). Se transforman a los rangos requeridos y se redondean los insumos a dos decimales:

| Campo | Tipo/escala | Generación |
|---|---|---|
| `parcela_id` | entero | 1 a 100 |
| `volumen_riego` | m³/ha | Uniforme [100, 300], 2 decimales |
| `fertilizante_kg` | kg/ha | Uniforme [50, 150], 2 decimales |
| `temperatura_c` | °C | Uniforme [18, 32], 2 decimales |
| `tipo_tratamiento` | categórica | A si U < 0.50; B en otro caso |
| `rendimiento_ton` | ton/ha | Ecuación y ruido uniforme, 2 decimales |

El rendimiento se simula mediante:

\[
Y = 5.0 + 0.035R + 0.020F + E_T + \varepsilon,
\]

donde \(R\) es el volumen de riego, \(F\) es la cantidad de fertilizante, \(E_T=1.8\) ton/ha para B y 0 para A, y \(\varepsilon\sim U(-0.8,0.8)\) ton/ha. La tabla `parcelas` se recupera con una consulta SQL y se convierte directamente a pandas para el análisis.

La semilla vuelve reproducible la generación al ejecutar la misma consulta bajo la misma versión de DuckDB y el mismo entorno. Para comparaciones reproducibles entre instalaciones conviene mantener controladas las versiones de las dependencias, ya que los generadores pueden cambiar entre versiones.

## 3. Correlación y modelo predictivo OLS

### 3.1 Correlación de Pearson

La aplicación calcula la matriz de Pearson para riego, fertilizante, temperatura y rendimiento. El panel presenta los coeficientes de la selección filtrada. El coeficiente \(r\) indica dirección e intensidad de una relación lineal, no un efecto causal.

Por construcción, el rendimiento aumenta con el riego a razón de 0.035 ton/ha por cada m³/ha y con el fertilizante a razón de 0.020 ton/ha por cada kg/ha. Sin embargo, la magnitud de la correlación depende también de los rangos simulados y de la variación de cada variable. En la ejecución verificada, el riego tuvo la asociación absoluta más alta con el rendimiento (\(|r|=0.89\)); la temperatura se genera de forma independiente y no participa en la ecuación, por lo que su correlación muestral es incidental y cercana a cero. El coeficiente de Pearson describe asociación lineal y no un efecto causal.

La aplicación identifica de forma automática cuál de las tres variables muestra la mayor correlación absoluta con el rendimiento en los datos actualmente filtrados. Esta comparación es descriptiva y puede cambiar al mover los filtros.

### 3.2 Regresión lineal múltiple

Se ajusta por mínimos cuadrados ordinarios el modelo:

\[
\hat{Y}=\hat{\beta}_0+\hat{\beta}_1R+\hat{\beta}_2F.
\]

El estimador de referencia en la ecuación generadora es \(\beta_0=5.0\), \(\beta_1=0.035\) y \(\beta_2=0.020\); los coeficientes estimados con 100 observaciones pueden diferir por el ruido y el redondeo. Para la ejecución con DuckDB 1.5.6, el modelo estimó \(\hat{\beta}_0=6.0454\) (p = \(3.27\times10^{-22}\)), \(\hat{\beta}_1=0.0359\) (p = \(1.93\times10^{-36}\)) y \(\hat{\beta}_2=0.0162\) (p = 0.0000284). Los dos predictores resultaron estadísticamente significativos al nivel 0.05.

El \(R^2\) ajustado mide la proporción de variabilidad explicada por el modelo, penalizando la inclusión de predictores. En la ejecución verificada, el rendimiento promedio global fue 14.98 ton/ha y el \(R^2\) ajustado fue 0.823 (82.3%), lo que indica que el ajuste explicó una proporción alta de la variación muestral del rendimiento. El RMSE fue 1.007 ton/ha. Los p-valores de los predictores fueron menores que 0.05, por lo que existe evidencia de asociación lineal con el rendimiento al controlar el otro predictor. Estos valores corresponden a la semilla 0.42 y DuckDB 1.5.6; otras versiones del generador aleatorio pueden alterar la muestra sintética.

El histograma con KDE permite inspeccionar si los residuos se distribuyen de manera aproximadamente simétrica alrededor de cero. En el resumen OLS de esta ejecución, la prueba Jarque–Bera dio \(p=0.0234\), señal de desviación de normalidad al nivel 0.05, aunque la asimetría estimada fue baja (0.038). Por ello, la inferencia OLS debe interpretarse con cautela y se recomienda revisar diagnósticos adicionales, posibles observaciones influyentes y, para decisiones críticas, usar errores estándar robustos o validación adicional. También deben considerarse linealidad, independencia y homocedasticidad.

## 4. Inferencia: evaluación del bio-estimulante

Se compara el rendimiento de dos grupos independientes. La formulación unilateral es:

\[
H_0:\mu_B-\mu_A\leq 0,\qquad H_1:\mu_B-\mu_A>0.
\]

La aplicación ejecuta la prueba t de Student independiente con varianzas iguales (`scipy.stats.ttest_ind`, `equal_var=True`, `alternative="greater"`) y nivel \(\alpha=0.05\). El intervalo de confianza bilateral del 95% para \(\mu_B-\mu_A\) se calcula con la varianza agrupada de la misma prueba. La comparación principal usa las 100 parcelas y permanece global aunque se apliquen filtros a las gráficas; el panel también calcula la comparación de los datos filtrados cuando ambos grupos conservan observaciones suficientes.

**Regla de decisión:** si el p-valor unilateral es menor que 0.05, se rechaza \(H_0\) y existe evidencia estadística de que el promedio de B supera al de A. En caso contrario, no se rechaza \(H_0\); la simulación no aporta evidencia suficiente para afirmar la mejora. La aplicación expone la diferencia de medias, el IC 95%, el estadístico t y el p-valor para documentar la decisión.

La ecuación generadora incorpora deliberadamente un incremento de 1.8 ton/ha en B. En la ejecución verificada, la diferencia de medias observada fue 1.97 ton/ha, con IC 95% [1.09, 2.86], estadístico \(t=4.428\) y p-valor unilateral \(1.238\times10^{-5}\). Como \(p<0.05\), se rechaza \(H_0\): hay evidencia estadística de mayor rendimiento promedio para B en la muestra simulada. La recomendación es avanzar a un piloto controlado con evaluación económica, no realizar de inmediato una inversión total. La significancia estadística no equivale a rentabilidad; antes de escalar B se deben comparar costo del bio-estimulante, aplicación, valor comercial de la producción adicional, calidad, efectos sobre el consumo de agua y riesgos agronómicos.

## 5. Aplicación web y visualización

El archivo `app.py` concentra la generación SQL, la conversión a pandas, los análisis estadísticos y la interfaz Streamlit. La estructura lógica separa la carga simulada, el ajuste OLS y la prueba de hipótesis en funciones. La interfaz usa un diseño de ancho completo, una barra lateral con filtros interactivos y descarga CSV, tres tarjetas KPI y una cuadrícula de cuatro visualizaciones.

1. **Mapa de calor de Pearson:** representa la matriz de correlación numérica con los coeficientes anotados. Un mensaje identifica el factor con mayor asociación absoluta con el rendimiento en la selección.
2. **Dispersión y ajuste OLS:** muestra riego frente a rendimiento, con observaciones, recta de mínimos cuadrados y banda de confianza del 95%. La banda comunica incertidumbre del ajuste medio.
3. **Diagrama de cajas por tratamiento:** muestra mediana, cuartiles, rango y observaciones atípicas; los rombos indican las medias. La diferencia formal de medias se consulta junto al gráfico y en el panel inferencial.
4. **Histograma y KDE de residuos:** representa los residuos del modelo múltiple global y una línea vertical en cero. Sirve como inspección inicial de centrado y forma de la distribución.

Los filtros de tratamiento, riego y fertilizante se aplican a la tabla y a las gráficas descriptivas. Para evitar que filtrar un grupo invalide la comparación experimental, el KPI y la prueba principal se calculan con los datos globales. El panel deja explícito este alcance.

## 6. Ejecución y despliegue

### Ejecución local

1. Instalar Python 3.11 o una versión compatible con las dependencias indicadas.
2. Abrir una terminal en la carpeta del proyecto, donde se encuentran `app.py` y `requirements.txt`.
3. Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # macOS/Linux
   # .venv\\Scripts\\activate     # Windows PowerShell
   ```

4. Instalar las librerías:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Iniciar el servidor local y abrir la dirección que muestra Streamlit (normalmente `http://localhost:8501`):

   ```bash
   streamlit run app.py
   ```

### Publicación en Streamlit Community Cloud

1. Crear un repositorio en GitHub y agregar `app.py` y `requirements.txt` en la raíz (o anotar la ruta si están en un subdirectorio).
2. Confirmar los cambios y subirlos a la rama que se utilizará para el despliegue:

   ```bash
   git add app.py requirements.txt Informe_Tecnico.md
   git commit -m "Agregar dashboard de análisis agrícola"
   git push origin main
   ```

3. Iniciar sesión en Streamlit Community Cloud en [share.streamlit.io](https://share.streamlit.io/) y conectar y autorizar el acceso a GitHub.
4. Elegir **Create app**, seleccionar el repositorio, la rama y `app.py` como archivo de entrada. Definir opcionalmente el subdominio de la aplicación.
5. Pulsar **Deploy**. Community Cloud instalará las dependencias declaradas en `requirements.txt`; revisar los logs si la compilación informa un error.
6. Al finalizar, abrir y compartir la dirección pública bajo `*.streamlit.app`. Los cambios enviados a la rama configurada se reflejan en el despliegue.

La definición de dependencias en `requirements.txt` permite que el entorno de despliegue instale explícitamente las librerías que utiliza la aplicación. Para auditoría reproducible, se recomienda fijar versiones exactas después de validar el entorno local y mantener sincronizadas esas versiones con el despliegue.

## 7. Conclusiones técnicas

El proyecto implementa un recorrido reproducible de generación de datos, modelado predictivo e inferencia estadística con una interfaz interactiva. El modelo separa la asociación lineal del riego y fertilizante, mientras la prueba t aborda de forma directa la pregunta operativa del tratamiento. El filtrado ayuda a explorar subconjuntos sin alterar inadvertidamente la comparación global principal.

La decisión de adoptar B debe basarse primero en la regla estadística de la prueba y, después, en una evaluación económica y agronómica. Dado que los registros son sintéticos y la mejora de B forma parte de la ecuación de generación, los resultados validan el flujo de análisis, no la eficacia real del producto. Para sustentar una decisión de inversión se requiere un ensayo aleatorizado con réplicas en invernadero, control de variedad y ciclo de cultivo, mediciones de calidad y costos, y un análisis de rentabilidad.

## Fuentes técnicas

- [Documentación oficial: ejecutar una aplicación Streamlit](https://docs.streamlit.io/get-started/installation/command-line)
- [Documentación oficial: dependencias en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Documentación oficial: desplegar en Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
