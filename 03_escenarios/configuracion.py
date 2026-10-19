"""Parámetros configurables del modelo SIRP.

Este es el ÚNICO lugar donde viven los parámetros que uno podría querer
cambiar sin tocar la lógica. El resto del código importa desde acá o lee
los archivos de `datos/`. Los parámetros del problema (costos, capacidades,
vida útil, horizonte, etc.) NO van acá: se leen de `datos/parametros.csv`
y demás archivos mediante `cargador_datos.py`.

Cada constante tiene encima un comentario con: qué es, de dónde sale el
valor y qué pasa si se cambia.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas y lectura de archivos
# ---------------------------------------------------------------------------

# Carpeta con los datos del enunciado. Se calcula relativa a este archivo para
# que el código funcione sin importar desde dónde se ejecute.
# Si se cambia: todos los cargadores leerán desde la nueva carpeta.
RUTA_DATOS = Path(__file__).resolve().parents[1] / "01_datos"

# Nombre de cada archivo dentro de RUTA_DATOS. `distancias.csv` es una
# transcripción de la Tabla A.4 del Informe 1 (no venía en los datos del
# enunciado). Si se renombra un archivo, se actualiza sólo acá.
ARCHIVOS = {
    "demanda_historica": "demanda_historica.csv",
    "demanda_red_diaria": "demanda_red_diaria.csv",
    "oferta_historica": "oferta_historica.csv",
    "hospitales": "hospitales.csv",
    "capacidades_hospitales": "capacidades_hospitales.csv",
    "flota": "flota.csv",
    "distancias": "distancias.csv",
    "parametros": "parametros.csv",
}

# Codificación de los CSV. Los archivos traen BOM de UTF-8, por eso "utf-8-sig".
CODIFICACION_ARCHIVOS = "utf-8-sig"

# Formato de las fechas en los CSV (mes/día/año, p. ej. "1/31/2022").
FORMATO_FECHA = "%m/%d/%Y"

# ---------------------------------------------------------------------------
# Oferta: Normal por mes, independiente día a día
# ---------------------------------------------------------------------------
# No hay parámetro de autocorrelación: quitada la estacionalidad mensual, la
# oferta no tiene dependencia entre días (ρ = 0.013, IC 95% −0.046 a 0.072;
# ver analisis_de_la_oferta_y_demanda/Oferta_autocorrelacion.ipynb). El 0.48
# del Informe 1 medía la estacionalidad, que ya captura la Normal mensual.

# Cota inferior para la oferta muestreada. La Normal puede generar valores
# negativos (muy improbable con medias ~640–880 y desviaciones ~80–100), y
# una oferta negativa no tiene sentido físico.
# Si se cambia: los escenarios truncan la oferta en este valor.
OFERTA_MINIMA = 0

# ---------------------------------------------------------------------------
# Demanda: distribución marginal por hospital
# ---------------------------------------------------------------------------

# Familia de distribución de la demanda de cada hospital.
# Fuente: Tabla A.3 del Informe 1 (mejor ajuste por AIC). Los parámetros
# de cada distribución se estiman desde la serie histórica en
# distribuciones.py; acá sólo va la familia.
# Supuesto: estos ajustes son correctos. Su validación formal (KS, AD, χ²)
# está pendiente (PENDIENTES.md).
# Si se cambia: se debe usar uno de "binomial_negativa", "binomial" o "poisson".
TIPO_DISTRIBUCION_DEMANDA = {
    "Instituto Traumatológico": "binomial_negativa",
    "Hospital Calvo Mackenna": "binomial_negativa",
    "Hospital San José": "binomial_negativa",
    "Hospital Roberto del Río": "binomial_negativa",
    "Hospital del Salvador": "binomial_negativa",
    "Hospital San Juan de Dios": "binomial_negativa",
    "Hospital Barros Luco": "binomial_negativa",
    "Ex Posta Central": "binomial_negativa",
    "Hospital Félix Bulnes": "binomial_negativa",
    "Hospital San Borja Arriarán": "binomial_negativa",
    "Hospital Luis Tisné": "binomial",
    "Hospital Padre Hurtado": "binomial_negativa",
    "Hospital Parroquial San Bernardo": "poisson",
    "Hospital Lucio Córdova": "binomial",
    "Hospital San José de Maipo": "binomial",
    "Hospital Exequiel González Cortés": "binomial_negativa",
    "Hospital Sótero del Río": "binomial_negativa",
    "Hospital El Pino": "binomial_negativa",
    "Instituto Geriatría": "binomial",
    "Instituto Nacional del Cáncer": "binomial",
    "Instituto Neurocirugía": "binomial",
    "Instituto Psiquiátrico": "binomial_negativa",
}

# Rango de búsqueda de log(r) en la máxima verosimilitud de la binomial
# negativa. Es el mismo del notebook Demanda_Capstone.ipynb, con el que se
# obtuvo la Tabla A.3; se mantiene para reproducir esos ajustes.
# Si se cambia: si el óptimo queda en el borde, ampliar el rango.
COTAS_LOG_R_BINOMIAL_NEGATIVA = (-8.0, 12.0)

# Rango de búsqueda del parámetro n (número de ensayos) de la binomial,
# igual que en el notebook de demanda:
#   n_min = max(máximo observado, ceil(media))
#   n_max = max(n_min + HOLGURA_N_BINOMIAL, FACTOR_N_BINOMIAL * n_min)
# Si se cambia: un rango más amplio es más lento pero explora más valores.
HOLGURA_N_BINOMIAL = 100
FACTOR_N_BINOMIAL = 4

# ---------------------------------------------------------------------------
# Escenarios y horizonte rodante
# ---------------------------------------------------------------------------

# Número de escenarios de oferta y demanda que se generan en cada resolución
# del horizonte rodante (al inicio de la semana y cada día).
# Si se cambia: más escenarios representan mejor la incertidumbre, pero los
# modelos de optimización se vuelven más grandes y lentos.
N_ESCENARIOS_DEFAULT = 30

# Día de la semana en que parte el ciclo semanal (0 = lunes, convención de
# Python). Toda la semana usa el mes de este día para la oferta (decisión
# del grupo, ver METODOLOGIA.md).
DIA_INICIO_SEMANA = 0

# Semilla del generador aleatorio. None = no reproducible. Fijar un entero
# para obtener siempre los mismos escenarios (útil para depurar y comparar).
# En la simulación, de esta semilla se derivan dos generadores independientes:
# uno para las semanas "reales" y otro para los escenarios del modelo, de modo
# que nunca coincidan.
SEMILLA = None

# ---------------------------------------------------------------------------
# Variantes exploradas del caso base (codigo/caso_base.py, METODOLOGIA.md Anexo 15.A)
# ---------------------------------------------------------------------------

# Cuantil de la demanda diaria con que se fija el nivel objetivo de cada
# hospital: η_i = min(cuantil de su demanda diaria, C_i) (§15.A.1).
# Caso base principal: 0.95 (P95), con sensibilidad a 0.90 y 0.99 (decisión
# de José Ignacio del 2026-10-06, por aprobar con el grupo; §15.A.5).
# "critico" = fractil crítico con los costos de
# datos/parametros.csv, π / (π + γ) = 2300 / 2980 ≈ 0.77.
# Si se cambia: un cuantil mayor lleva más unidades (menos faltantes, más holding).
CUANTIL_CASO_BASE = 0.95

# De dónde sale el percentil del nivel objetivo (§15.A.5).
#   "historico": el percentil directo de la demanda diaria histórica de cada
#       hospital (como PERCENTIL.INCLUIR en Excel), redondeado hacia arriba. No
#       usa las distribuciones ajustadas: el caso base queda independiente del
#       modelamiento de la metodología. Es la del caso base principal.
#   "distribucion": el percentil de la distribución ajustada de cada hospital
#       (Tabla A.3), como en la propuesta original de la §15.A.1.
# Si se cambia: los niveles cambian a lo más en 1 o 2 unidades por hospital.
FUENTE_NIVEL_CASO_BASE = "historico"

# Cuánta oferta admite el banco cada día (§15.A.1).
#   "ajustada": sólo lo que se despacha hoy más μ_b (el mínimo terminal del
#       banco, 10% de su capacidad), sin pasar su capacidad. Es la del caso
#       base principal (§15.A.5).
#   "todo": toda la que quepa en el banco (propuesta original de la §15.A.1;
#       descartada porque llena el banco sin atender más demanda, §15.A.2).
# Si se cambia: "ajustada" deja menos unidades pasando la noche en el banco.
ADMISION_CASO_BASE = "ajustada"

# Variantes que corre `python -m codigo.caso_base` para la sensibilidad
# (§15.A.1): cada par es (cuantil, admisión). Los resultados quedan en
# resultados/caso_base_semanas_2024_5xmes.csv, una fila por variante y semana.
VARIANTES_CASO_BASE = [
    (cuantil, admision)
    for admision in ("todo", "ajustada")
    for cuantil in (0.90, 0.95, 0.99, "critico")
]

# Las variantes que se reportan como caso base (§15.A.5): el principal (P95 con
# admisión ajustada) y su sensibilidad al percentil. Las demás de
# VARIANTES_CASO_BASE y la variante con escenarios (§15.A.3) se siguen corriendo
# como registro de lo explorado, pero no son el caso base.
VARIANTE_PRINCIPAL_CASO_BASE = (0.95, "ajustada")
VARIANTES_OFICIALES_CASO_BASE = [(0.90, "ajustada"), (0.95, "ajustada"), (0.99, "ajustada")]

# ---------------------------------------------------------------------------
# Caso base OFICIAL: agnóstico (codigo/caso_base_agnostico.py, METODOLOGIA.md §15)
# ---------------------------------------------------------------------------
# Las variantes de arriba (CUANTIL_CASO_BASE, ADMISION_CASO_BASE, etc.) quedan
# como registro de lo explorado; el caso base que se reporta es el agnóstico.

# Fracción β de su capacidad hasta la que se rellena cada hospital cada día.
# None = capacidad diaria de la flota / capacidad total de los hospitales
# (715 / 2544 ≈ 0,281): sale solo de capacidades, sin mirar demanda ni oferta.
# Si se cambia: un β mayor lleva más unidades (menos faltantes, más holding).
BETA_CASO_BASE = None

# Valores de β para la sensibilidad (None = el principal).
BETAS_SENSIBILIDAD_CASO_BASE = [0.25, None, 0.35]

# Escenarios construidos sobre las 52 semanas reales de 2024 (§15.4): la
# demanda y la oferta reales se multiplican por estos factores. "ruido" agrega
# un factor al azar en [1 − ruido, 1 + ruido], distinto cada día y hospital.
# Si se cambia: los escenarios quedan más (o menos) extremos.
ESCENARIOS_CONSTRUIDOS = {
    "favorable": {"demanda": 0.90, "oferta": 1.10, "ruido": 0.0},
    "neutro": {"demanda": 1.00, "oferta": 1.00, "ruido": 0.01},
    "desfavorable": {"demanda": 1.10, "oferta": 0.90, "ruido": 0.0},
}

# Semilla del ruido del escenario neutro.
SEMILLA_ESCENARIOS = 2024

# ---------------------------------------------------------------------------
# Prueba con las semanas reales (codigo/evaluacion_historica.py, §15.4)
# ---------------------------------------------------------------------------

# Corte entre los datos con que se construye la política y los datos con que
# se la prueba: las distribuciones se ajustan sólo con lo anterior a esta
# fecha (2022–2023) y se prueba con las semanas reales desde esta fecha (2024).
# Así la política es agnóstica a los datos de prueba: no los vio al definirse.
# Si se cambia: cambia qué años se usan para ajustar y cuáles para probar.
INICIO_PRUEBA_HISTORICA = "2024-01-01"

# Nivel de confianza de los intervalos de los KPIs (0.95 = 95%).
# Si se cambia: un nivel mayor da intervalos más anchos.
CONFIANZA_INTERVALOS = 0.95

# ---------------------------------------------------------------------------
# Heurística C (codigo/heuristica_c.py, METODOLOGIA.md §6)
# ---------------------------------------------------------------------------

# Largo (en días) de cada bloque en que se resuelve el calendario el lunes
# en el intento 1 de la heurística C (descartado, ver heuristica_c.py):
# (4, 3) = primero lunes a jueves y después viernes a domingo. Cada bloque
# parte, en cada escenario, del inventario que dejó el anterior. Deben sumar
# el horizonte (7). Fuente: interpretación nuestra del "4+3" de
# METODOLOGIA.md §6, que en realidad se refiere a las cantidades.
# Si se cambia: bloques más cortos son más rápidos de resolver, pero cada uno
# ve menos días hacia adelante. Con (7,) el calendario es el del modelo común.
BLOQUES_CALENDARIO_C = (4, 3)

# Días que ve cada mañana el modelo de cantidades: hoy y los siguientes, sin
# pasar del domingo. El mínimo terminal se exige al cierre de la ventana.
# Fuente: METODOLOGIA.md §6 ("ventanas de 3 días").
# Si se cambia: con 7 o más, el modelo de cantidades es el del modelo común.
VENTANA_CANTIDADES_C = 3

# ---------------------------------------------------------------------------
# Gurobi
# ---------------------------------------------------------------------------

# Tiempo máximo (segundos) por resolución de cada modelo. Si se alcanza, se
# usa la mejor solución encontrada hasta ese momento.
# Valores PROVISORIOS, elegidos sin criterio formal (ver PENDIENTES.md).
# Si se cambia: más tiempo da soluciones mejores o probadamente óptimas, pero
# la simulación tarda más (el modelo de cantidades se resuelve 7 veces por semana).
TIEMPO_LIMITE_CALENDARIO_S = 3600
TIEMPO_LIMITE_CANTIDADES_S = 600

# Gap de optimalidad relativo con que Gurobi se detiene (0.01 = 1%).
# Valor PROVISORIO (ver PENDIENTES.md). Si se cambia: menor gap = solución más
# cercana al óptimo, pero más tiempo.
GAP_MIP = 0.01

# Si es True, Gurobi imprime su log de resolución en la consola.
MOSTRAR_LOG_GUROBI = False
