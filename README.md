El proyecto es minimalista: contiene un único script de Python (pipeline_permisos_laserena_2024.py) y un archivo de datos Excel (permiso-de-circulacion-2024.xlsx). El script implementa una pipeline de limpieza, auditoría normativa y generación de reportes/gráficos para la información de permisos de circulación de 2024 en La Serena, Chile.

Estructura del script
Configuración y constantes

Define rutas de entrada y salida, año de análisis y opciones como MAKE_PLOTS o MODO_AUDITORIA.

Establece rangos válidos para campos numéricos y mapeos para normalizar textos (combustible, transmisión, etc.).

Funciones auxiliares

Limpieza de strings (normalize_str), formateo de números, creación de etiquetas para gráficos, etc.

Funciones de visualización (boxplot_legible, hist_legible) que generan gráficos con enfoque en legibilidad.

Pipeline

Carga del Excel con pandas.

Renombrar/normalizar columnas y valores.

Deduplicados: elimina registros duplicados por placa/año usando la fecha de pago.

Auditoría normativa: genera archivos CSV con filas que incumplen reglas (ventanas de pago, monto mínimo legal, formato de patente, combustible no estándar, etc.).

Cap/recorte: limita valores a rangos razonables sin eliminar filas.

Resumen de consola con memoria estimada, tipos de datos, nulos y fechas.

Gráficos y reportes: histograma/boxplot para variables numéricas y gráficos de barras para variables categóricas; se exportan CSV, Parquet y Excel con datos limpios.

Salidas

Todo se deposita en la carpeta salidas/, creada automáticamente.

Archivos de auditoría sólo se generan si se detectan inconsistencias.

Qué es importante saber
Dependencias clave: pandas, numpy, matplotlib, openpyxl. Asegúrate de tenerlas instaladas antes de ejecutar el script.

Formato de datos: el script asume nombres y formatos específicos de columnas; cualquier cambio en la estructura del Excel requerirá actualizar los mapeos o reglas.

Parámetros de auditoría: los umbrales (fechas, montos, rangos) están hardcodeados; revisar si son válidos para futuros años o datasets.

Modo auditoría: con MODO_AUDITORIA=True se reportan inconsistencias pero no se eliminan filas. Cambiarlo a False aplica filtros duros.

Recomendaciones para aprender a continuación
Pandas avanzado

Manejo de DataFrame, filtrado, groupby, tratamiento de valores faltantes.

Lectura/escritura en múltiples formatos (CSV, Excel, Parquet).

Validación de datos y reglas de negocio

Expresiones regulares para validar formatos (patentes, códigos SII).

Diseño de pipelines reproducibles y parametrizables.

Visualización con Matplotlib

Personalización de gráficos, formato de ejes y anotaciones.

Exportación de figuras con buena resolución.

Automatización y reutilización

Convertir secciones del script en funciones o módulos reutilizables.

Parametrizar el año/dataset, para facilitar su uso con datos futuros.
