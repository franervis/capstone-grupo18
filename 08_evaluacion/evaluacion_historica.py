"""Prueba con las semanas reales de 2024, intervalos de confianza y escenarios
favorables, neutros y desfavorables.

1. **Política agnóstica a los datos de prueba.** Las distribuciones de demanda
   y oferta se ajustan sólo con los datos anteriores a
   `cfg.INICIO_PRUEBA_HISTORICA` (2022–2023). La política se define con eso y
   nunca ve 2024 antes de vivirlo.
2. **Escenario real.** Se le hace vivir las 52 semanas reales de 2024, con la
   oferta y la demanda que efectivamente ocurrieron. `ejecutar_semana` se las
   revela un día a la vez, igual que con las semanas simuladas.
3. **Intervalos de confianza** de cada KPI: intervalo t de Student para el
   promedio semanal, con nivel `cfg.CONFIANZA_INTERVALOS`.
4. **Escenarios favorables, neutros y desfavorables.** Las semanas se separan
   en tercios según su holgura: oferta total de la semana menos demanda total.
   El tercio con más holgura es favorable, el del medio neutro y el con menos
   desfavorable.

Correr desde la raíz del repo (unos segundos):

    python -m codigo.evaluacion_historica

Guarda `resultados/caso_base_historico_2024.csv` (una fila por variante y
semana real) y `resultados/caso_base_intervalos.csv` (intervalos por variante
y escenario, para las semanas reales y las 60 simuladas).
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from codigo import caso_base, cargador_datos, distribuciones
from codigo import configuracion as cfg
from codigo.horizonte_rodante import Insumos
from codigo.simulacion import SemanaReal, cargar_semanas

ARCHIVO_HISTORICO = "resultados/caso_base_historico_2024.csv"
ARCHIVO_INTERVALOS = "resultados/caso_base_intervalos.csv"

ESCENARIOS = ("favorable", "neutro", "desfavorable")
KPIS_INTERVALOS = ["costo_operacional", "costo_total", "fill_rate_red", "unidades_faltantes",
                   "unidades_vencidas", "deficit_terminal", "km", "utilizacion_flota"]


# ---------------------------------------------------------------------------
# Semanas reales y ajuste sin los datos de prueba
# ---------------------------------------------------------------------------


def preparar_insumos(hasta=cfg.INICIO_PRUEBA_HISTORICA):
    """Como `horizonte_rodante.preparar_insumos`, pero ajusta las
    distribuciones sólo con los datos anteriores a `hasta`.

    Devuelve: un objeto Insumos.
    """
    corte = pd.Timestamp(hasta)
    demanda = cargador_datos.cargar_demanda_historica()
    oferta = cargador_datos.cargar_oferta_historica()
    nodos = cargador_datos.cargar_hospitales()
    return Insumos(
        parametros=cargador_datos.cargar_parametros(),
        hospitales=cargador_datos.tabla_hospitales(),
        nombre_banco=nodos.loc[nodos["tipo"] == "Banco", "nombre"].item(),
        capacidad_banco=cargador_datos.capacidad_banco(),
        flota=cargador_datos.cargar_flota(),
        distancias=cargador_datos.cargar_distancias(),
        distribuciones_demanda=distribuciones.ajustar_distribuciones_demanda(demanda[demanda["fecha"] < corte]),
        distribuciones_oferta=distribuciones.ajustar_distribuciones_oferta(oferta[oferta["fecha"] < corte]),
        demanda_historica=demanda[demanda["fecha"] < corte],
    )


def semanas_historicas(insumos, desde=cfg.INICIO_PRUEBA_HISTORICA, hasta=None):
    """Semanas reales completas (de lunes a domingo) de los datos.

    Recibe:
        insumos: Insumos (define el orden de los hospitales).
        desde: la primera semana es la del primer lunes desde esta fecha.
        hasta: fecha límite (incluida) para el domingo de la última semana;
            None = hasta el final de los datos.
    Devuelve: lista de SemanaReal en orden cronológico, numeradas 0, 1, 2, ...
    """
    horizonte = insumos.parametros["horizonte"]
    demanda = cargador_datos.cargar_demanda_historica().pivot(index="fecha", columns="hospital", values="demanda")
    demanda = demanda[list(insumos.hospitales["nombre"])]
    oferta = cargador_datos.cargar_oferta_historica().set_index("fecha")["oferta"]
    ultimo = min(demanda.index.max(), oferta.index.max())
    if hasta is not None:
        ultimo = min(ultimo, pd.Timestamp(hasta))

    lunes = pd.Timestamp(desde)
    lunes += pd.Timedelta(days=(cfg.DIA_INICIO_SEMANA - lunes.dayofweek) % 7)
    semanas = []
    while lunes + pd.Timedelta(days=horizonte - 1) <= ultimo:
        dias = pd.date_range(lunes, periods=horizonte, freq="D")
        semanas.append(SemanaReal(lunes, oferta.loc[dias].to_numpy(dtype=int),
                                  demanda.loc[dias].to_numpy(dtype=int), indice=len(semanas)))
        lunes += pd.Timedelta(days=7)
    return semanas


# ---------------------------------------------------------------------------
# Escenarios construidos: favorable, neutro y desfavorable
# ---------------------------------------------------------------------------


def escenario_construido(semanas, nombre, semilla=cfg.SEMILLA_ESCENARIOS):
    """Copia de las semanas con la demanda y la oferta transformadas según
    `cfg.ESCENARIOS_CONSTRUIDOS[nombre]`: se multiplican por su factor y, si el
    escenario tiene ruido, por un factor al azar en [1 − ruido, 1 + ruido]
    distinto cada día (y hospital). Se redondea al entero más cercano, sin bajar de 0.

    Recibe: semanas (lista de SemanaReal), nombre ("favorable", "neutro",
        "desfavorable" o "real" = sin cambios) y la semilla del ruido.
    Devuelve: lista de SemanaReal con las mismas fechas e índices.
    """
    if nombre == "real":
        return list(semanas)
    factores = cfg.ESCENARIOS_CONSTRUIDOS[nombre]
    rng = np.random.default_rng(semilla)
    ruido = factores["ruido"]
    salida = []
    for s in semanas:
        fd = factores["demanda"] * (rng.uniform(1 - ruido, 1 + ruido, s.demanda.shape) if ruido else 1.0)
        fo = factores["oferta"] * (rng.uniform(1 - ruido, 1 + ruido, s.oferta.shape) if ruido else 1.0)
        salida.append(SemanaReal(s.fecha_lunes, np.maximum(0, np.rint(s.oferta * fo)).astype(int),
                                 np.maximum(0, np.rint(s.demanda * fd)).astype(int), indice=s.indice))
    return salida


# ---------------------------------------------------------------------------
# Semanas favorables, neutras y desfavorables por holgura, e intervalos
# ---------------------------------------------------------------------------


def clasificar_semanas(semanas):
    """Separa las semanas en tercios según su holgura (oferta total menos
    demanda total de la semana).

    Devuelve: DataFrame indexado por semana (indice) con oferta, demanda,
        holgura y escenario ("favorable", "neutro" o "desfavorable").
    """
    tabla = pd.DataFrame({
        "semana": [s.indice for s in semanas],
        "oferta_semana": [int(s.oferta.sum()) for s in semanas],
        "demanda_semana": [int(s.demanda.sum()) for s in semanas],
    }).set_index("semana")
    tabla["holgura"] = tabla["oferta_semana"] - tabla["demanda_semana"]
    # Tercios por orden de holgura (empates por número de semana), así cada grupo tiene ~1/3.
    orden = tabla.sort_values(["holgura"], kind="stable").index
    tercio = np.array_split(np.array(orden), 3)
    for nombre, semanas_grupo in zip(("desfavorable", "neutro", "favorable"), tercio):
        tabla.loc[semanas_grupo, "escenario"] = nombre
    return tabla


def intervalo(valores, confianza=cfg.CONFIANZA_INTERVALOS):
    """Intervalo t de Student para el promedio de `valores`.

    Devuelve: (promedio, inferior, superior, n). Con una sola semana el
        intervalo no se puede calcular y queda NaN.
    """
    x = np.asarray(valores, dtype=float)
    n = len(x)
    promedio = x.mean()
    if n < 2:
        return promedio, np.nan, np.nan, n
    margen = stats.t.ppf((1 + confianza) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return promedio, promedio - margen, promedio + margen, n


def tabla_intervalos(tabla, clasificacion, conjunto, kpis_=None):
    """Intervalos de cada KPI por variante y escenario (y para todas las semanas).

    Recibe: tabla con una fila por variante y semana (columnas variante y
        semana), la clasificación de `clasificar_semanas` y el nombre del conjunto.
    Devuelve: DataFrame largo: conjunto, variante, escenario, kpi, promedio,
        inferior, superior, semanas.
    """
    kpis_ = kpis_ or KPIS_INTERVALOS
    tabla = tabla.join(clasificacion["escenario"], on="semana")
    filas = []
    for variante, grupo in tabla.groupby("variante", sort=False):
        for escenario in ("todas",) + ESCENARIOS:
            datos = grupo if escenario == "todas" else grupo[grupo["escenario"] == escenario]
            for kpi in kpis_:
                promedio, inferior, superior, n = intervalo(datos[kpi])
                filas.append({"conjunto": conjunto, "variante": variante, "escenario": escenario, "kpi": kpi,
                              "promedio": promedio, "inferior": inferior, "superior": superior, "semanas": n})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Corrida
# ---------------------------------------------------------------------------


def correr_caso_base(insumos, semanas):
    """Corre las 8 variantes del caso base y la variante con escenarios."""
    return pd.concat([caso_base.correr_variantes(insumos, semanas),
                      caso_base.correr_escenarios(insumos, semanas)], ignore_index=True)


def main():
    from codigo.horizonte_rodante import preparar_insumos as preparar_insumos_completos

    # 1–2. Política ajustada con 2022–2023 y probada con las semanas reales de 2024.
    insumos = preparar_insumos()
    semanas = semanas_historicas(insumos)
    clasificacion = clasificar_semanas(semanas)
    historico = correr_caso_base(insumos, semanas)
    historico = historico.join(clasificacion[["holgura", "escenario"]], on="semana")
    historico.to_csv(ARCHIVO_HISTORICO, index=False, encoding="utf-8")

    # 3–4. Intervalos por escenario, para las semanas reales y para las 60 simuladas.
    simulado = pd.read_csv(caso_base.ARCHIVO_RESULTADOS)
    semanas_simuladas = cargar_semanas(caso_base.ARCHIVO_SEMANAS, preparar_insumos_completos())
    intervalos = pd.concat([
        tabla_intervalos(historico.drop(columns=["holgura", "escenario"]), clasificacion, "historico_2024"),
        tabla_intervalos(simulado, clasificar_semanas(semanas_simuladas), "simulado_60"),
    ], ignore_index=True)
    intervalos.to_csv(ARCHIVO_INTERVALOS, index=False, encoding="utf-8")

    print("Semanas reales de 2024 por escenario (holgura = oferta − demanda de la semana):")
    print(clasificacion.groupby("escenario", sort=False)["holgura"].agg(["count", "min", "max"]).to_string())
    pd.set_option("display.width", 250)
    for kpi in ("costo_operacional", "fill_rate_red"):
        vista = intervalos[(intervalos["conjunto"] == "historico_2024") & (intervalos["kpi"] == kpi)]
        formato = (lambda v: f"{v:.3f}") if kpi == "fill_rate_red" else (lambda v: f"{v / 1e6:.2f} M")
        vista = vista.assign(texto=[f"{formato(p)} [{formato(a)}, {formato(b)}]"
                                    for p, a, b in zip(vista["promedio"], vista["inferior"], vista["superior"])])
        print(f"\n{kpi}, semanas reales de 2024 (promedio [intervalo {cfg.CONFIANZA_INTERVALOS:.0%}]):")
        print(vista.pivot(index="variante", columns="escenario", values="texto")[["todas", *ESCENARIOS]].to_string())
    print(f"\nGuardado en {ARCHIVO_HISTORICO} ({len(historico)} filas) y {ARCHIVO_INTERVALOS} ({len(intervalos)} filas).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
