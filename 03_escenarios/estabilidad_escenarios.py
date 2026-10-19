"""Prueba de estabilidad del número de escenarios (Kaut y Wallace, 2003).

Pregunta: ¿alcanzan 30 escenarios? Un número de escenarios alcanza cuando el
resultado deja de depender de qué escenarios se sortearon. Para medirlo, el
calendario de la heurística B se resuelve varias veces para la misma semana,
con distintos números de escenarios y distintos sorteos (semillas), y se mira:

1. Dentro de la muestra: cuánto cambia el costo que el propio modelo estima
   (valor objetivo) entre sorteos. Si con N escenarios casi no cambia, el
   modelo ya no depende de la suerte del sorteo.
2. Fuera de la muestra: cada calendario se evalúa en simulación, con las
   mismas semanas "reales" para todos (números aleatorios comunes). Si el
   costo simulado con 30 escenarios es igual al de 50, más escenarios no
   mejoran las decisiones.

Solo cambia el número de escenarios del CALENDARIO. El modelo de cantidades de
cada día usa siempre 30, como en la política, para aislar el efecto.

No modifica el código del proyecto. Desde la raíz del repo:

    caffeinate -i python -m codigo.estabilidad_escenarios            # corre (o sigue) la prueba
    python -m codigo.estabilidad_escenarios --grafico                # solo el gráfico y la tabla

Guarda cada resultado apenas termina, así que si se corta, se vuelve a lanzar
el mismo comando y sigue donde quedó:
- resultados/estabilidad_dentro_muestra.csv: una fila por (semana, N, semilla).
- resultados/estabilidad_fuera_muestra.csv: una fila por (semana, N, semilla, realización).
- resultados/estabilidad_escenarios.png y resultados/estabilidad_resumen.csv: el gráfico y la tabla del anexo.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codigo import configuracion as cfg
from codigo import heuristica_b
from codigo.escenarios import generar_escenarios
from codigo.horizonte_rodante import estado_inicial_semana, preparar_insumos
from codigo.simulacion import cargar_semanas, generar_semanas_reales, simular_semana

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
ARCHIVO_DENTRO = RESULTADOS / "estabilidad_dentro_muestra.csv"
ARCHIVO_FUERA = RESULTADOS / "estabilidad_fuera_muestra.csv"
ARCHIVO_SEMANAS = "01_datos/semanas_simuladas_consecutivas_2024.csv"   # año simulado (conjunto oficial)

# Semanas del año simulado: una normal (la primera) y la de mayor escasez (oferta − demanda más negativa).
SEMANAS = [0, 37]
TAMANOS = [5, 10, 20, 30, 50]
SEMILLAS = [0, 1, 2, 3, 4]
REALIZACIONES = 8
SEMILLA_BASE = 2024


def estado_lunes(indice, insumos):
    """Inventario con que parte la semana: el inicial estándar (10% de la
    capacidad, edad 0) en todas las semanas de la prueba.

    No se usa el inventario que tenía la B ese lunes en el año simulado porque
    `resultados/heuristica_b_compacta_anio_simulado_2024.csv` guarda el detalle
    por hospital solo de la última semana. Para esta prueba basta con que todos
    los calendarios de una misma semana partan igual."""
    return estado_inicial_semana(insumos)


def resolver_calendario(semana, estado, n, semilla, insumos):
    """Calendario de la B con n escenarios de un sorteo dado. La oferta del lunes
    es la real de la semana, como en la política."""
    rng = np.random.default_rng(np.random.SeedSequence([SEMILLA_BASE, 1, semana.indice, n, semilla]))
    escenarios = generar_escenarios(7, semana.fecha_lunes.month, insumos.distribuciones_demanda, n,
                                    distribuciones_oferta=insumos.distribuciones_oferta,
                                    oferta_hoy=int(semana.oferta[0]), rng=rng)
    return heuristica_b.resolver_modelo_calendario(escenarios, estado, insumos)


def evaluar(calendario, semana, estado, realizaciones, insumos):
    """Costo simulado del calendario en `realizaciones` semanas reales, iguales para
    todos los calendarios (mismo sorteo de la realidad y de los escenarios diarios)."""
    reales = generar_semanas_reales(semana.fecha_lunes, realizaciones, insumos,
                                    np.random.default_rng(np.random.SeedSequence([SEMILLA_BASE, 2, semana.indice])))
    filas = []
    for j, real in enumerate(reales):
        # Para que la oferta del lunes coincida con la que vio el calendario.
        real.oferta[0] = semana.oferta[0]
        rng = np.random.default_rng(np.random.SeedSequence([SEMILLA_BASE, 3, semana.indice, j]))
        resumen, _, _ = simular_semana(real, insumos, cfg.N_ESCENARIOS_DEFAULT, rng, estado_inicial=estado,
                                       resolver_calendario=lambda *_: calendario)
        filas.append({"realizacion": j, "costo_operacional": resumen["costo_operacional"],
                      "costo_total": resumen["costo_total"], "fill_rate_red": resumen["fill_rate_red"]})
    return filas


def guardar(fila_o_filas, ruta):
    filas = fila_o_filas if isinstance(fila_o_filas, list) else [fila_o_filas]
    pd.DataFrame(filas).to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")


def correr(semanas_idx, tamanos, semillas, realizaciones):
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    insumos = preparar_insumos()
    semanas = {s.indice: s for s in cargar_semanas(ARCHIVO_SEMANAS, insumos)}
    hechas = set()
    if ARCHIVO_DENTRO.exists():
        hechas = set(map(tuple, pd.read_csv(ARCHIVO_DENTRO)[["semana", "n_escenarios", "semilla"]].to_numpy().tolist()))
    pendientes = [(w, n, r) for w in semanas_idx for n in tamanos for r in semillas if (w, n, r) not in hechas]
    print(f"{len(hechas)} corridas hechas, {len(pendientes)} pendientes.", flush=True)
    for w, n, r in pendientes:
        semana, estado = semanas[w], estado_lunes(w, insumos)
        inicio = time.time()
        cal = resolver_calendario(semana, estado, n, r, insumos)
        t_cal = time.time() - inicio
        filas = evaluar(cal, semana, estado, realizaciones, insumos)
        info = cal.info_resolucion
        # Primero lo de fuera de la muestra: si se corta entre los dos, la corrida se repite completa.
        guardar([{"semana": w, "n_escenarios": n, "semilla": r, **f} for f in filas], ARCHIVO_FUERA)
        guardar({"semana": w, "fecha_lunes": semana.fecha_lunes.date(), "n_escenarios": n, "semilla": r,
                 "valor_objetivo": info["valor_objetivo"], "gap": info["gap"], "optimo": info["optimo"],
                 "km": cal.distancia_km, "visitas": len(cal.visitas), "tiempo_calendario_s": t_cal,
                 "tiempo_total_s": time.time() - inicio}, ARCHIVO_DENTRO)
        costo = np.mean([f["costo_total"] for f in filas])
        print(f"semana {w:2d}, N={n:2d}, semilla {r}: objetivo {info['valor_objetivo']:,.0f} (gap {info['gap']:.2%}), "
              f"simulado {costo:,.0f}; calendario {t_cal:.0f} s, total {time.time() - inicio:.0f} s", flush=True)


def resumen_y_grafico():
    import matplotlib.pyplot as plt

    dentro = pd.read_csv(ARCHIVO_DENTRO)
    fuera = pd.read_csv(ARCHIVO_FUERA)
    # Costo simulado de cada calendario = promedio sobre las realizaciones (las mismas para todos).
    # Se compara con el costo total (incluye la penalización del mínimo al cierre), que es lo mismo que mide el valor objetivo.
    por_cal = fuera.groupby(["semana", "n_escenarios", "semilla"])["costo_total"].mean().rename("simulado").reset_index()
    t = dentro.merge(por_cal, on=["semana", "n_escenarios", "semilla"])
    filas = []
    for (w, n), g in t.groupby(["semana", "n_escenarios"]):
        filas.append({"semana": w, "n_escenarios": n, "sorteos": len(g),
                      "objetivo_promedio": g["valor_objetivo"].mean(),
                      "objetivo_variacion_pct": 100 * g["valor_objetivo"].std(ddof=1) / g["valor_objetivo"].mean() if len(g) > 1 else np.nan,
                      "simulado_promedio": g["simulado"].mean(),
                      "simulado_variacion_pct": 100 * g["simulado"].std(ddof=1) / g["simulado"].mean() if len(g) > 1 else np.nan,
                      "tiempo_calendario_s": g["tiempo_calendario_s"].mean()})
    resumen = pd.DataFrame(filas)
    resumen.to_csv(RESULTADOS / "estabilidad_resumen.csv", index=False, encoding="utf-8")
    print(resumen.round(2).to_string(index=False))

    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11})
    nombre = {0: "Semana normal (1 de enero)", 37: "Semana de escasez (16 de septiembre)"}
    color = {0: "#3E74D6", 37: "#C8323A"}
    paneles = [("objetivo_variacion_pct", "a) Costo estimado por el modelo", "Variación entre sorteos (%)"),
               ("simulado_variacion_pct", "b) Costo simulado de la semana", "Variación entre sorteos (%)"),
               ("tiempo_calendario_s", "c) Tiempo de resolución del calendario", "Segundos")]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=200)
    for ax, (col, titulo, eje_y) in zip(ejes, paneles):
        for w, g in resumen.groupby("semana"):
            ax.plot(g["n_escenarios"], g[col], color=color.get(w, "#5C5C5C"), lw=2, marker="o", ms=6, label=nombre.get(w, f"Semana {w}"))
        ax.axvline(30, color="#8C8C8C", lw=1, ls="--")
        ax.set_xscale("log"); ax.set_xticks(sorted(resumen["n_escenarios"].unique()))
        ax.set_xticklabels([str(v) for v in sorted(resumen["n_escenarios"].unique())]); ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.text(30, ax.get_ylim()[1] * 0.97, " 30 (usado)", color="#5C5C5C", fontsize=9, va="top")
        ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Número de escenarios"); ax.set_ylabel(eje_y)
        ax.grid(axis="y", color="#E6E6E6"); ax.spines[["top", "right"]].set_visible(False)
    ejes[0].axhline(1, color="#8C8C8C", lw=0.8, ls=":")
    ejes[0].text(5, 1.05, "tolerancia de Gurobi (1%)", fontsize=8.5, color="#5C5C5C", va="bottom")
    ejes[2].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, -0.04, "Variación entre sorteos = desviación estándar del costo entre 5 sorteos independientes de escenarios, como % del promedio. "
             "Calendario de la heurística B; en (b), cada calendario se evalúa en las mismas 8 semanas simuladas,\ncon 30 escenarios en el modelo diario. "
             "Ambas semanas parten con el inventario inicial estándar (10% de la capacidad).", fontsize=8.5, color="#5C5C5C")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "estabilidad_escenarios.png", bbox_inches="tight", facecolor="white")
    print("Gráfico en", RESULTADOS / "estabilidad_escenarios.png")


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--semanas", type=int, nargs="+", default=SEMANAS)
    parser.add_argument("--tamanos", type=int, nargs="+", default=TAMANOS)
    parser.add_argument("--semillas", type=int, nargs="+", default=SEMILLAS)
    parser.add_argument("--realizaciones", type=int, default=REALIZACIONES)
    parser.add_argument("--grafico", action="store_true", help="solo arma la tabla y el gráfico con lo que ya corrió")
    args = parser.parse_args(argumentos)
    if not args.grafico:
        correr(args.semanas, args.tamanos, args.semillas, args.realizaciones)
    resumen_y_grafico()
    return 0


if __name__ == "__main__":
    sys.exit(main())
