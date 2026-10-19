"""Corre una política sobre las 52 semanas reales de 2024 ENCADENADAS: cada
lunes parte con el inventario real que dejó el domingo anterior, con sus
edades (METODOLOGIA.md §15.4), tal cual o en un escenario construido
(favorable, neutro o desfavorable, §15.4).

Todas las políticas usan lo mismo: distribuciones ajustadas solo con 2022–2023
(`evaluacion_historica.preparar_insumos`), las 52 semanas reales de 2024, 30
escenarios del modelo y semilla 2024, y la primera semana parte con el 10% de
la capacidad.

Desde la raíz del repo:

    python -m codigo.correr_encadenado caso_base                          # caso base oficial (agnóstico), todo, segundos
    python -m codigo.correr_encadenado b --escenario desfavorable         # heurística B en un escenario
    python -m codigo.correr_encadenado a --escenario real --minutos 50    # para a los 50 min
    python -m codigo.correr_encadenado caso_base_registro                 # variantes exploradas (P95, etc.)
    python -m codigo.correr_encadenado b --resumir                        # solo arma los KPIs de una corrida hecha

Al completar las 52 semanas, cada heurística deja tres archivos, con el mismo
formato que los de las 60 semanas: `..._encadenado_2024.csv` (una fila por
semana), `..._kpis_resumen_encadenado_2024.csv` (KPIs por mes y año, con
intervalo del 95%) y `..._kpis_hospitales_encadenado_2024.csv` (fill rate por hospital).

Las heurísticas guardan cada semana apenas termina (con el inventario final),
así que si se cortan, al volver a correr el mismo comando siguen desde la semana
siguiente. `--minutos` deja de empezar semanas nuevas pasado ese tiempo.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codigo import caso_base, evaluacion_historica
from codigo import configuracion as cfg
from codigo.modelo_comun import EstadoSistema
from codigo.simulacion import simular_semanas_encadenadas

SEMILLA = 2024
ARCHIVOS = {
    "caso_base_registro": "resultados/caso_base_encadenado_2024.csv",
    "a": "resultados/heuristica_a_encadenado_2024.csv",
    "b": "resultados/heuristica_b_compacta_encadenado_2024.csv",
}
ESCENARIOS = ("real", "favorable", "neutro", "desfavorable")


def archivo(politica, escenario):
    """Archivo de resultados de una heurística en un escenario (el real conserva su nombre)."""
    ruta = ARCHIVOS[politica]
    return ruta if escenario == "real" else ruta.replace("_encadenado_2024", f"_encadenado_{escenario}_2024")


def resolvedores_heuristica(nombre):
    if nombre == "a":
        from codigo import heuristica_a
        return {"resolver_calendario": heuristica_a.resolver_modelo_calendario}
    from codigo import heuristica_b
    return {"resolver_calendario": heuristica_b.resolver_modelo_calendario}


def _estado_desde_fila(fila):
    return EstadoSistema(dia=0, inventario_banco=np.array(json.loads(fila["estado_final_banco"]), dtype=int),
                         inventario_hospitales=np.array(json.loads(fila["estado_final_hospitales"]), dtype=int))


def correr_caso_base_registro(insumos, semanas):
    variantes = [(caso_base.nombre_variante(c, a), caso_base.politica(c, a), 1) for c, a in cfg.VARIANTES_CASO_BASE]
    variantes.append(("escenarios", caso_base.politica_escenarios(), cfg.N_ESCENARIOS_DEFAULT))
    tablas = []
    for nombre, pol, n in variantes:
        tabla = simular_semanas_encadenadas(semanas, insumos, n, SEMILLA, **pol)
        tabla.insert(0, "variante", nombre)
        tablas.append(tabla)
    tabla = pd.concat(tablas, ignore_index=True)
    tabla.to_csv(ARCHIVOS["caso_base_registro"], index=False, encoding="utf-8")
    return tabla


def correr_heuristica(nombre, insumos, semanas, minutos=None, escenario="real"):
    ruta = Path(archivo(nombre, escenario))
    hechas = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()
    estado = _estado_desde_fila(hechas.iloc[-1]) if len(hechas) else None
    pendientes = semanas[len(hechas):]
    if not pendientes:
        print(f"Las {len(semanas)} semanas ya están en {ruta}.")
        return hechas
    print(f"Heurística {nombre.upper()}, escenario {escenario}: {len(hechas)} semanas hechas; parte en la semana {pendientes[0].indice} "
          f"({pendientes[0].fecha_lunes.date()}).", flush=True)
    inicio = time.time()

    class Parar(Exception):
        pass

    def guardar(resumen, estado_final):
        fila = dict(resumen)
        fila["estado_final_banco"] = json.dumps(estado_final.inventario_banco.tolist())
        fila["estado_final_hospitales"] = json.dumps(estado_final.inventario_hospitales.tolist())
        pd.DataFrame([fila]).to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")
        print(f"  semana {resumen['semana']:2d} ({pd.Timestamp(resumen['fecha_lunes']).date()}): "
              f"costo operacional {resumen['costo_operacional']:,.0f}, {resumen['tiempo_s']:.0f} s", flush=True)
        if minutos is not None and time.time() - inicio > 60 * minutos:
            raise Parar

    try:
        simular_semanas_encadenadas(pendientes, insumos, cfg.N_ESCENARIOS_DEFAULT, SEMILLA, estado_inicial=estado,
                                    al_terminar_semana=guardar, **resolvedores_heuristica(nombre))
    except Parar:
        print(f"Se cumplieron los {minutos} minutos; para seguir, correr el mismo comando.")
    return pd.read_csv(ruta)


def archivos_kpis(ruta):
    """Rutas del resumen de KPIs y del fill rate por hospital de una corrida
    encadenada, con el mismo formato que las de las 60 semanas
    (`heuristica_x_kpis_resumen_...` y `heuristica_x_kpis_hospitales_...`)."""
    ruta = str(ruta)
    return ruta.replace("_encadenado", "_kpis_resumen_encadenado"), ruta.replace("_encadenado", "_kpis_hospitales_encadenado")


def resumir(ruta, insumos):
    """Resume una corrida encadenada completa en dos archivos, como los de las 60 semanas:

    - resumen: por mes (el del lunes de cada semana) y para el año, cada KPI de
      `kpis.COLUMNAS_KPI` que tenga la tabla (ver `kpis.completar_tabla`)
      con promedio, desviación, p5, p95, número de semanas
      e intervalo del 95% (inferior, superior). A diferencia de
      `kpis.resumir_semanas`, el año pondera las 52 semanas por igual, porque
      en 2024 los meses tienen 4 o 5 semanas.
    - hospitales: fill rate de cada hospital sobre las semanas (`kpis.kpis_hospitales`).

    Devuelve: (resumen, hospitales) como DataFrames.
    """
    from codigo import kpis
    from codigo.evaluacion_historica import intervalo

    tabla, columnas = kpis.completar_tabla(pd.read_csv(ruta))
    tabla["mes"] = pd.to_datetime(tabla["fecha_lunes"]).dt.month
    filas = []
    for periodo, grupo in [*tabla.groupby("mes"), ("año", tabla)]:
        for kpi in columnas:
            x = grupo[kpi].astype(float)
            promedio, inferior, superior, n = intervalo(x)
            filas.append({"periodo": periodo, "kpi": kpi, "promedio": promedio, "desviacion": x.std(ddof=1),
                          "p5": x.quantile(0.05), "p95": x.quantile(0.95), "semanas": n,
                          "inferior": inferior, "superior": superior})
    resumen = pd.DataFrame(filas)
    hospitales = kpis.kpis_hospitales(tabla, insumos)
    ruta_resumen, ruta_hospitales = archivos_kpis(ruta)
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8")
    hospitales.to_csv(ruta_hospitales, index=False, encoding="utf-8")
    print(f"Resumen en {ruta_resumen} y {ruta_hospitales}.")
    return resumen, hospitales


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("politica", choices=["caso_base", "caso_base_registro", "a", "b"])
    parser.add_argument("--escenario", choices=ESCENARIOS, default="real", help="solo para las heurísticas")
    parser.add_argument("--minutos", type=float, default=None, help="no empezar semanas nuevas pasado este tiempo")
    parser.add_argument("--resumir", action="store_true", help="solo armar los archivos de KPIs de una corrida ya hecha")
    args = parser.parse_args(argumentos)
    if args.politica == "caso_base":
        from codigo import caso_base_agnostico
        return caso_base_agnostico.main()
    insumos = evaluacion_historica.preparar_insumos()
    semanas = evaluacion_historica.semanas_historicas(insumos)
    if args.politica == "caso_base_registro":
        tabla = correr_caso_base_registro(insumos, semanas)
        resumen = tabla.groupby("variante", sort=False)[["costo_operacional", "fill_rate_red", "unidades_vencidas",
                                                         "inventario_final_banco", "inventario_final_hospitales"]].mean()
        pd.set_option("display.width", 200)
        print(resumen.round(3).to_string())
        print(f"Guardado en {ARCHIVOS['caso_base_registro']} ({len(tabla)} filas).")
    else:
        ruta = archivo(args.politica, args.escenario)
        if not args.resumir:
            semanas = evaluacion_historica.escenario_construido(semanas, args.escenario)
            correr_heuristica(args.politica, insumos, semanas, args.minutos, args.escenario)
        # Al completar las 52 semanas (o con --resumir), arma los archivos de KPIs.
        if Path(ruta).exists() and len(pd.read_csv(ruta)) == len(evaluacion_historica.semanas_historicas(insumos)):
            resumir(ruta, insumos)
        elif args.resumir:
            print(f"{ruta} no tiene las 52 semanas todavía.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
