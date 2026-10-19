"""Indicadores de desempeño (KPIs) de una política.

Los KPIs se calculan con lo que realmente pasó en cada semana simulada, no con
lo que planificó el modelo. Todo sale de lo que devuelve
`simulacion.simular_semana`: la semana real, el calendario y los registros
diarios de `horizonte_rodante.actualizar_estado`. Por eso sirven igual para el
modelo común, las heurísticas y el caso base.

Hay tres niveles:

1. Por semana: `kpis_semana` (KPIs de la red) y `detalle_hospitales` (demanda
   y faltante de cada hospital). Ambos devuelven diccionarios planos, pensados
   para agregarse al resumen de `simular_semana`.
2. Por hospital, sobre un conjunto de semanas: `kpis_hospitales` y `equidad`.
3. Entre semanas: `resumir_semanas` (promedio, desviación y percentiles por
   mes y para el año).

Las tasas (fill rate, wastage rate, oferta no admitida, utilización,
participaciones) van como
fracción entre 0 y 1, no como porcentaje.
"""

import numpy as np
import pandas as pd

from codigo import modelo_comun

# Prefijos de las columnas por hospital; el sufijo es la posición del hospital
# en Insumos.hospitales con dos dígitos (demanda_h00, demanda_h01, ...).
PREFIJO_DEMANDA = "demanda_h"
PREFIJO_FALTANTE = "faltante_h"

# KPIs de la red que se resumen entre semanas (claves de `kpis_semana`).
COLUMNAS_KPI = [
    "costo_operacional",
    "participacion_transporte",
    "participacion_holding",
    "participacion_shortage",
    "participacion_wastage",
    "deficit_terminal",
    "fill_rate_red",
    "unidades_faltantes",
    "unidades_vencidas",
    "wastage_rate",
    "oferta_no_admitida",
    "tasa_oferta_no_admitida",
    "km",
    "km_por_unidad",
    "utilizacion_flota",
    "visitas",
]


def _dividir(numerador, denominador):
    """numerador / denominador; NaN si el denominador es 0."""
    return float(numerador) / float(denominador) if denominador else float("nan")


def kpis_semana(semana, calendario, registros, insumos, estado_inicial=None):
    """KPIs de la red para una semana simulada.

    Recibe: lo que usa y devuelve `simulacion.simular_semana`:
        semana (SemanaReal), calendario (SolucionCalendario), registros (lista
        de registros diarios; el último trae "estado_final") e insumos.
    Devuelve: diccionario con
        - costo_operacional y sus componentes (costo_transporte, costo_holding,
          costo_shortage, costo_wastage), SIN la penalización terminal;
        - participacion_*: fracción del costo operacional de cada componente;
        - deficit_terminal: unidades bajo el mínimo terminal al cierre del domingo;
        - fill_rate_red, demanda_total, unidades_faltantes;
        - unidades_vencidas, wastage_rate;
        - oferta_total, oferta_no_admitida y tasa_oferta_no_admitida (fracción
          de la oferta de la semana que no se admitió en el banco);
        - km, unidades_despachadas, km_por_unidad, utilizacion_flota, visitas.
    Supone: la semana partió con el inventario inicial de
        `modelo_comun.inventario_inicial` (así lo hace `simular_semana`), salvo
        que se entregue `estado_inicial` (semanas encadenadas): entonces se usa ese.
    """
    par = insumos.parametros
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))

    demanda = int(semana.demanda.sum())
    faltante = int(sum(r["faltante"].sum() for r in registros))
    vencidas = int(sum(r["vencidas_banco"] + r["vencidas_hospitales"].sum() for r in registros))
    noche_banco = int(sum(r["noche_banco"] for r in registros))
    noche_hospitales = int(sum(r["noche_hospitales"].sum() for r in registros))
    admitida = int(sum(r["admitida"] for r in registros))
    oferta = int(semana.oferta.sum())
    despachada = int(sum(r["despachada"] for r in registros))
    km = float(calendario.distancia_km)

    # Costo operacional: la función objetivo sin la penalización terminal.
    costo_transporte = par["costo_km"] * km
    costo_holding = par["costo_holding_banco"] * noche_banco + par["costo_holding_hospital"] * noche_hospitales
    costo_shortage = par["costo_shortage"] * faltante
    costo_wastage = par["costo_wastage"] * vencidas
    costo_operacional = costo_transporte + costo_holding + costo_shortage + costo_wastage

    # Inventario final: unidades que faltan para el mínimo terminal.
    final = registros[-1]["estado_final"]
    mu_b, mu_i = modelo_comun.minimos_terminales(insumos)
    deficit = max(0.0, mu_b - final.inventario_banco.sum())
    deficit += np.maximum(0.0, mu_i - final.inventario_hospitales.sum(axis=1)).sum()

    # Todo lo que entró al sistema: inventario inicial + oferta admitida.
    if estado_inicial is None:
        inicial_banco, inicial_hospitales = modelo_comun.inventario_inicial(insumos)
    else:
        inicial_banco, inicial_hospitales = estado_inicial.inventario_banco, estado_inicial.inventario_hospitales
    entradas = int(inicial_banco.sum() + inicial_hospitales.sum()) + admitida

    # Capacidad de los vehículos que salieron (w = 1): los que tienen alguna visita.
    salidas = {(t, k) for (t, _i, k) in calendario.visitas}
    capacidad_usada = sum(theta[k] for (_t, k) in salidas)

    return {
        "costo_operacional": float(costo_operacional),
        "costo_transporte": float(costo_transporte),
        "costo_holding": float(costo_holding),
        "costo_shortage": float(costo_shortage),
        "costo_wastage": float(costo_wastage),
        "participacion_transporte": _dividir(costo_transporte, costo_operacional),
        "participacion_holding": _dividir(costo_holding, costo_operacional),
        "participacion_shortage": _dividir(costo_shortage, costo_operacional),
        "participacion_wastage": _dividir(costo_wastage, costo_operacional),
        "deficit_terminal": float(deficit),
        "fill_rate_red": 1 - _dividir(faltante, demanda),
        "demanda_total": demanda,
        "unidades_faltantes": faltante,
        "unidades_vencidas": vencidas,
        "wastage_rate": _dividir(vencidas, entradas),
        "oferta_total": oferta,
        "oferta_no_admitida": oferta - admitida,
        "tasa_oferta_no_admitida": _dividir(oferta - admitida, oferta),
        "km": km,
        "unidades_despachadas": despachada,
        "km_por_unidad": _dividir(km, despachada),
        "utilizacion_flota": _dividir(despachada, capacidad_usada),
        "visitas": len(calendario.visitas),
    }


def detalle_hospitales(semana, registros):
    """Demanda y faltante de cada hospital en una semana simulada.

    Devuelve: diccionario plano con una clave por hospital y dato
        (demanda_h00, faltante_h00, demanda_h01, ...), en el orden de
        Insumos.hospitales. Es el insumo de `kpis_hospitales`.
    """
    demanda = semana.demanda.sum(axis=0)
    faltante = sum(r["faltante"] for r in registros)
    detalle = {}
    for i in range(len(demanda)):
        detalle[f"{PREFIJO_DEMANDA}{i:02d}"] = int(demanda[i])
        detalle[f"{PREFIJO_FALTANTE}{i:02d}"] = int(faltante[i])
    return detalle


def kpis_hospitales(tabla, insumos):
    """Fill rate de cada hospital sobre un conjunto de semanas.

    Se calcula con los totales del conjunto (faltantes totales sobre demanda
    total), no como promedio de los fill rate semanales. Supuesto vigente, por
    revisar.

    Recibe: tabla = DataFrame con una fila por semana y las columnas de
        `detalle_hospitales`; insumos.
    Devuelve: DataFrame con una fila por hospital: hospital, demanda,
        faltante y fill_rate (NaN si el hospital no tuvo demanda).
    """
    nombres = list(insumos.hospitales["nombre"])
    demanda = np.array([tabla[f"{PREFIJO_DEMANDA}{i:02d}"].sum() for i in range(len(nombres))], dtype=float)
    faltante = np.array([tabla[f"{PREFIJO_FALTANTE}{i:02d}"].sum() for i in range(len(nombres))], dtype=float)
    fill_rate = np.full(len(nombres), np.nan)
    con_demanda = demanda > 0
    fill_rate[con_demanda] = 1 - faltante[con_demanda] / demanda[con_demanda]
    return pd.DataFrame({
        "hospital": nombres,
        "demanda": demanda.astype(int),
        "faltante": faltante.astype(int),
        "fill_rate": fill_rate,
    })


def equidad(tabla_hospitales):
    """KPIs de equidad a partir de la tabla de `kpis_hospitales`.

    Devuelve: diccionario con fill_rate_minimo, hospital_fill_rate_minimo,
        fill_rate_maximo y brecha_fill_rate (máximo − mínimo). Los hospitales
        sin demanda no se consideran.
    """
    con_demanda = tabla_hospitales.dropna(subset=["fill_rate"])
    peor = con_demanda.loc[con_demanda["fill_rate"].idxmin()]
    maximo = float(con_demanda["fill_rate"].max())
    return {
        "fill_rate_minimo": float(peor["fill_rate"]),
        "hospital_fill_rate_minimo": peor["hospital"],
        "fill_rate_maximo": maximo,
        "brecha_fill_rate": maximo - float(peor["fill_rate"]),
    }


def completar_tabla(tabla):
    """Prepara una tabla semanal ya guardada para resumirla.

    Las corridas guardadas antes de que existiera `tasa_oferta_no_admitida`
    traen `oferta_total` y `oferta_no_admitida`, pero no la tasa: aquí se
    calcula. Los KPIs que la corrida no guardó (por ejemplo, la oferta en las
    corridas con semanas reales de 2024) se omiten.

    Devuelve: (copia de la tabla, KPIs de COLUMNAS_KPI que tiene).
    """
    tabla = tabla.copy()
    if "tasa_oferta_no_admitida" not in tabla and {"oferta_total", "oferta_no_admitida"} <= set(tabla.columns):
        tabla["tasa_oferta_no_admitida"] = tabla["oferta_no_admitida"] / tabla["oferta_total"].replace(0, np.nan)
    return tabla, [c for c in COLUMNAS_KPI if c in tabla]


def _estadisticos(tabla, columnas):
    return pd.DataFrame({
        "promedio": tabla[columnas].mean(),
        "desviacion": tabla[columnas].std(ddof=1),
        "p5": tabla[columnas].quantile(0.05),
        "p95": tabla[columnas].quantile(0.95),
        "semanas": len(tabla),
    })


def resumir_semanas(tabla, columnas=None):
    """Resume los KPIs entre semanas, por mes y para el año.

    Recibe: tabla = DataFrame con una fila por semana, la columna
        "fecha_lunes" y las columnas de `kpis_semana`; columnas = KPIs a
        resumir (por defecto, los de COLUMNAS_KPI que tenga la tabla; ver
        `completar_tabla`).
    Devuelve: DataFrame indexado por (periodo, kpi), donde periodo es el mes
        (1–12) o "año", con las columnas promedio, desviacion, p5, p95 y
        semanas. Los estadísticos son sobre los valores semanales.
    Supone: todos los meses tienen el mismo número de semanas; así el año
        pondera los meses por igual. Si no, falla.
    """
    if columnas is None:
        tabla, columnas = completar_tabla(tabla)
    mes = pd.to_datetime(tabla["fecha_lunes"]).dt.month
    if mes.value_counts().nunique() > 1:
        raise ValueError("Los meses tienen distinto número de semanas; el resumen del año no los ponderaría por igual.")
    partes = {int(m): _estadisticos(grupo, columnas) for m, grupo in tabla.groupby(mes)}
    partes["año"] = _estadisticos(tabla, columnas)
    return pd.concat(partes, names=["periodo", "kpi"])
