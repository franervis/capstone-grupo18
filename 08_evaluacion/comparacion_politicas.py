"""Comparación del caso base con las heurísticas en las 60 semanas simuladas
(METODOLOGIA.md §15.7).

No corre ningún modelo: usa los resultados ya guardados en `resultados/`. Todas
las políticas vivieron las mismas 60 semanas (misma oferta y demanda día a día),
así que se pueden comparar **semana a semana** (diferencias pareadas):

- por política: promedio e intervalo del 95% de cada KPI;
- por par heurística – caso base: promedio e intervalo del 95% de la
  diferencia semana a semana, y en cuántas semanas gana la heurística;
- equidad (fill rate por hospital sobre el conjunto de semanas);

todo para el conjunto completo y para las semanas favorables, neutras y
desfavorables (tercios de holgura, `evaluacion_historica.clasificar_semanas`).

Correr desde la raíz del repo (segundos):

    python -m codigo.comparacion_politicas

Guarda `resultados/comparacion_politicas.csv`, `resultados/comparacion_pareada.csv`
y `resultados/comparacion_equidad.csv`.
"""

import sys

import numpy as np
import pandas as pd

from codigo import caso_base, caso_base_agnostico, kpis
from codigo.evaluacion_historica import ESCENARIOS, clasificar_semanas, intervalo
from codigo.horizonte_rodante import preparar_insumos
from codigo.simulacion import cargar_semanas

HEURISTICAS = {
    "Heurística A": "resultados/heuristica_a_semanas_2024_5xmes.csv",
    "Heurística B (zonas compactas)": "resultados/heuristica_b_compacta_semanas_2024_5xmes.csv",
}
# El caso base oficial (agnóstico, §15) y su sensibilidad a β.
CASOS_BASE = {
    "Caso base agnóstico β 28% (principal)": "agnostico_28",
    "Caso base agnóstico β 25%": "agnostico_25",
    "Caso base agnóstico β 35%": "agnostico_35",
}
# KPI: True si más alto es mejor.
KPIS = {
    "costo_operacional": False, "costo_total": False, "fill_rate_red": True, "unidades_faltantes": False,
    "deficit_terminal": False, "km": False, "km_por_unidad": False, "utilizacion_flota": True,
}
ARCHIVO_POLITICAS = "resultados/comparacion_politicas.csv"
ARCHIVO_PAREADA = "resultados/comparacion_pareada.csv"
ARCHIVO_EQUIDAD = "resultados/comparacion_equidad.csv"


def cargar_politicas():
    """{nombre: DataFrame con una fila por semana}, indexado por semana."""
    tablas = {nombre: pd.read_csv(ruta) for nombre, ruta in HEURISTICAS.items()}
    resultados = pd.read_csv(caso_base_agnostico.ARCHIVO_SIMULADAS)
    tablas.update({nombre: resultados[resultados["variante"] == v] for nombre, v in CASOS_BASE.items()})
    return {nombre: t.set_index("semana").sort_index() for nombre, t in tablas.items()}


def grupos(clasificacion):
    """{escenario: semanas}, incluido "todas"."""
    salida = {"todas": list(clasificacion.index)}
    salida.update({e: list(clasificacion.index[clasificacion["escenario"] == e]) for e in ESCENARIOS})
    return salida


def comparar(politicas, clasificacion, insumos):
    """Arma las tres tablas de la comparación (ver el docstring del módulo)."""
    semanas = grupos(clasificacion)
    nombres = list(insumos.hospitales["nombre"])
    filas_pol, filas_par, filas_eq = [], [], []
    for escenario, indices in semanas.items():
        for nombre, tabla in politicas.items():
            datos = tabla.loc[indices]
            for kpi in KPIS:
                p, a, b, n = intervalo(datos[kpi])
                filas_pol.append({"politica": nombre, "escenario": escenario, "kpi": kpi,
                                  "promedio": p, "inferior": a, "superior": b, "semanas": n})
            por_hospital = kpis.kpis_hospitales(datos.reset_index(), insumos).set_index("hospital")
            equidad = kpis.equidad(por_hospital.reset_index())
            filas_eq.append({"politica": nombre, "escenario": escenario, **equidad,
                             "fill_rate_san_jose_de_maipo": por_hospital.loc["Hospital San José de Maipo", "fill_rate"]})
        for heuristica in HEURISTICAS:
            for base in CASOS_BASE:
                h, c = politicas[heuristica].loc[indices], politicas[base].loc[indices]
                for kpi, mayor_mejor in KPIS.items():
                    diferencia = h[kpi] - c[kpi]
                    p, a, b, n = intervalo(diferencia)
                    gana = (diferencia > 0) if mayor_mejor else (diferencia < 0)
                    significativa = not (a <= 0 <= b) if n > 1 else False
                    filas_par.append({"heuristica": heuristica, "caso_base": base, "escenario": escenario, "kpi": kpi,
                                      "diferencia_promedio": p, "inferior": a, "superior": b, "semanas": n,
                                      "semanas_gana_heuristica": int(gana.sum()), "significativa": significativa})
    return pd.DataFrame(filas_pol), pd.DataFrame(filas_par), pd.DataFrame(filas_eq)


def _texto(p, a, b, kpi):
    if kpi in ("fill_rate_red", "utilizacion_flota"):
        return f"{100 * p:.1f}% [{100 * a:.1f}; {100 * b:.1f}]"
    if kpi.startswith("costo"):
        return f"{p / 1e6:.2f} M [{a / 1e6:.2f}; {b / 1e6:.2f}]"
    return f"{p:,.0f} [{a:,.0f}; {b:,.0f}]"


def main():
    insumos = preparar_insumos()
    politicas = cargar_politicas()
    semanas = cargar_semanas(caso_base.ARCHIVO_SEMANAS, insumos)
    indices = [s.indice for s in semanas]
    for nombre, tabla in politicas.items():
        if list(tabla.index) != indices:
            raise ValueError(f"{nombre} no tiene las mismas 60 semanas.")
        if not np.array_equal(tabla["demanda_total"].to_numpy(), [s.demanda.sum() for s in semanas]):
            raise ValueError(f"{nombre} no vivió la misma demanda.")
    clasificacion = clasificar_semanas(semanas)
    por_politica, pareada, equidad = comparar(politicas, clasificacion, insumos)
    por_politica.to_csv(ARCHIVO_POLITICAS, index=False, encoding="utf-8")
    pareada.to_csv(ARCHIVO_PAREADA, index=False, encoding="utf-8")
    equidad.to_csv(ARCHIVO_EQUIDAD, index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print("Semanas simuladas por escenario:", clasificacion["escenario"].value_counts().to_dict())
    for kpi in ("costo_operacional", "fill_rate_red"):
        v = por_politica[por_politica["kpi"] == kpi]
        v = v.assign(texto=[_texto(p, a, b, kpi) for p, a, b in zip(v["promedio"], v["inferior"], v["superior"])])
        print(f"\n{kpi} (promedio [intervalo 95%]):")
        print(v.pivot(index="politica", columns="escenario", values="texto")[["todas", *ESCENARIOS]].loc[list(politicas)].to_string())
    v = pareada[(pareada["kpi"] == "costo_operacional")]
    v = v.assign(texto=[f"{p / 1e6:+.2f} M [{a / 1e6:+.2f}; {b / 1e6:+.2f}] {g}/{n}"
                        for p, a, b, g, n in zip(v["diferencia_promedio"], v["inferior"], v["superior"],
                                                 v["semanas_gana_heuristica"], v["semanas"])])
    print("\nCosto operacional: heurística − caso base, semana a semana (negativo = la heurística cuesta menos; "
          "al final, semanas en que gana la heurística):")
    print(v.pivot(index=["heuristica", "caso_base"], columns="escenario", values="texto")[["todas", *ESCENARIOS]].to_string())
    print("\nEquidad:")
    print(equidad.pivot(index="politica", columns="escenario", values="fill_rate_minimo")[["todas", *ESCENARIOS]]
          .loc[list(politicas)].map(lambda x: f"{100 * x:.1f}%").to_string())
    print(f"\nGuardado en {ARCHIVO_POLITICAS}, {ARCHIVO_PAREADA} y {ARCHIVO_EQUIDAD}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
