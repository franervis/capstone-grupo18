"""Comparación del caso base oficial con las heurísticas A y B en el AÑO
SIMULADO ENCADENADO, el conjunto con que se comparan las políticas
(decisión del grupo del 2026-10-07).

No corre ningún modelo: usa los resultados de `correr_anio_simulado`. Verifica
que las tres políticas vivieron las mismas 52 semanas con la misma demanda, y
compara semana a semana (diferencias pareadas con intervalo del 95%).

Desde la raíz del repo (segundos):

    python -m codigo.comparacion_anio_simulado

Guarda en resultados/:
- comparacion_anio_simulado_kpis.csv: cada KPI por política (promedio, IC 95%, p5, p95).
- comparacion_anio_simulado_pareada.csv: diferencias A − caso base, B − caso
  base y B − A por KPI (promedio, IC 95% y semanas en que gana la primera).
- comparacion_anio_simulado_hospitales.csv: fill rate de cada hospital por política.
- comparacion_anio_simulado_meses.csv: costo operacional y fill rate por mes y política.
"""

import sys

import numpy as np
import pandas as pd

from codigo import kpis
from codigo.evaluacion_historica import intervalo
from codigo.horizonte_rodante import preparar_insumos

POLITICAS = {
    "Caso base agnóstico": "resultados/caso_base_agnostico_anio_simulado_2024.csv",
    "Heurística A": "resultados/heuristica_a_anio_simulado_2024.csv",
    "Heurística B (zonas compactas)": "resultados/heuristica_b_compacta_anio_simulado_2024.csv",
}
PARES = [("Heurística A", "Caso base agnóstico"), ("Heurística B (zonas compactas)", "Caso base agnóstico"),
         ("Heurística B (zonas compactas)", "Heurística A")]
# KPI: True si más alto es mejor.
KPIS = {
    "costo_operacional": False, "costo_total": False, "costo_transporte": False, "costo_holding_banco": False,
    "costo_holding_hospitales": False, "costo_shortage": False, "costo_wastage": False, "penalizacion_terminal": False,
    "participacion_transporte": False, "participacion_holding": False, "participacion_shortage": False,
    "deficit_terminal": False, "fill_rate_red": True, "unidades_faltantes": False, "unidades_vencidas": False,
    "oferta_no_admitida": False, "tasa_oferta_no_admitida": False, "km": False, "km_por_unidad": False, "utilizacion_flota": True, "visitas": False,
    "unidades_despachadas": True,
}
ARCHIVO_KPIS = "resultados/comparacion_anio_simulado_kpis.csv"
ARCHIVO_PAREADA = "resultados/comparacion_anio_simulado_pareada.csv"
ARCHIVO_HOSPITALES = "resultados/comparacion_anio_simulado_hospitales.csv"
ARCHIVO_MESES = "resultados/comparacion_anio_simulado_meses.csv"


def cargar():
    """{política: DataFrame indexado por semana}; falla si no vivieron lo mismo."""
    tablas = {n: kpis.completar_tabla(pd.read_csv(r))[0].sort_values("semana").set_index("semana")
              for n, r in POLITICAS.items()}
    base = next(iter(tablas.values()))
    for nombre, tabla in tablas.items():
        if len(tabla) != 52 or list(tabla.index) != list(base.index):
            raise ValueError(f"{nombre} no tiene las mismas 52 semanas.")
        if not np.array_equal(tabla["demanda_total"].to_numpy(), base["demanda_total"].to_numpy()):
            raise ValueError(f"{nombre} no vivió la misma demanda.")
    return tablas


def comparar(tablas, insumos):
    """Arma las cuatro tablas de la comparación (ver el docstring del módulo)."""
    filas_kpi, filas_par = [], []
    for nombre, t in tablas.items():
        for kpi in KPIS:
            p, a, b, n = intervalo(t[kpi])
            filas_kpi.append({"politica": nombre, "kpi": kpi, "promedio": p, "inferior": a, "superior": b,
                              "p5": t[kpi].quantile(0.05), "p95": t[kpi].quantile(0.95), "semanas": n})
    for primera, segunda in PARES:
        for kpi, mayor_mejor in KPIS.items():
            d = tablas[primera][kpi] - tablas[segunda][kpi]
            p, a, b, n = intervalo(d)
            gana = (d > 0) if mayor_mejor else (d < 0)
            filas_par.append({"politica": primera, "contra": segunda, "kpi": kpi, "diferencia_promedio": p,
                              "inferior": a, "superior": b, "semanas": n, "semanas_gana_politica": int(gana.sum()),
                              "significativa": not (a <= 0 <= b)})
    hospitales = pd.concat([kpis.kpis_hospitales(t.reset_index(), insumos).assign(politica=n) for n, t in tablas.items()],
                           ignore_index=True)[["politica", "hospital", "demanda", "faltante", "fill_rate"]]
    meses = pd.concat([t.assign(mes=pd.to_datetime(t["fecha_lunes"]).dt.month).groupby("mes")[["costo_operacional", "fill_rate_red"]]
                       .mean().reset_index().assign(politica=n) for n, t in tablas.items()], ignore_index=True)
    return pd.DataFrame(filas_kpi), pd.DataFrame(filas_par), hospitales, meses[["politica", "mes", "costo_operacional", "fill_rate_red"]]


def main():
    insumos = preparar_insumos()
    tablas = cargar()
    tabla_kpis, pareada, hospitales, meses = comparar(tablas, insumos)
    tabla_kpis.to_csv(ARCHIVO_KPIS, index=False, encoding="utf-8")
    pareada.to_csv(ARCHIVO_PAREADA, index=False, encoding="utf-8")
    hospitales.to_csv(ARCHIVO_HOSPITALES, index=False, encoding="utf-8")
    meses.to_csv(ARCHIVO_MESES, index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    v = tabla_kpis[tabla_kpis["kpi"].isin(["costo_operacional", "fill_rate_red", "unidades_faltantes", "km"])]
    print(v.pivot(index="kpi", columns="politica", values="promedio").round(4).to_string())
    v = pareada[pareada["kpi"].isin(["costo_operacional", "fill_rate_red"])]
    print(v[["politica", "contra", "kpi", "diferencia_promedio", "inferior", "superior", "semanas_gana_politica"]].round(4).to_string(index=False))
    for n, g in hospitales.groupby("politica", sort=False):
        e = kpis.equidad(g)
        print(f"{n}: peor hospital {e['hospital_fill_rate_minimo']} ({e['fill_rate_minimo']:.3f}), brecha {e['brecha_fill_rate']:.3f}")
    print(f"\nGuardado en {ARCHIVO_KPIS}, {ARCHIVO_PAREADA}, {ARCHIVO_HOSPITALES} y {ARCHIVO_MESES}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
