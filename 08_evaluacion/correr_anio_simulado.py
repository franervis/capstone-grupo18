"""Corre una política sobre el AÑO SIMULADO ENCADENADO: 52 semanas simuladas
consecutivas de 2024 (lunes 1 de enero a domingo 29 de diciembre), donde cada
lunes parte con el inventario real que dejó el domingo anterior, con sus edades.

Es el conjunto con que se comparan las políticas (decisión del grupo,
2026-10-07).

- Las semanas están en `01_datos/semanas_simuladas_consecutivas_2024.csv`.
  Se sortearon una vez con `simulacion.generar_semanas_reales` y la semilla
  2024: oferta Normal del mes del lunes y demanda de la distribución de cada
  hospital (Tabla A.3), ajustadas con 2022–2024 (`horizonte_rodante.preparar_insumos`).
  No son las semanas históricas de 2024 (esas las usa `correr_encadenado`).
- Solo la primera semana parte con el 10% de cada capacidad (edad 0).
- 30 escenarios del modelo por resolución (1 en el caso base) y semilla 2024:
  la semana n usa `generador_modelo_semana(2024, n)`, como en `simular_semanas_encadenadas`.
- Guarda cada semana apenas termina, con el inventario final. Si se corta, al
  volver a correr el mismo comando sigue desde la semana siguiente.

Desde la raíz del repo (Mac enchufado y con la tapa abierta):

    python -m codigo.correr_anio_simulado caso_base   # caso base oficial (agnóstico), segundos
    python -m codigo.correr_anio_simulado a           # heurística A, ~20 min
    python -m codigo.correr_anio_simulado b           # heurística B (zonas compactas), ~5 h
    python -m codigo.correr_anio_simulado b --minutos 60   # no empezar semanas nuevas pasada 1 h

Salidas, para la política X (caso_base_agnostico, heuristica_a o heuristica_b_compacta):
- resultados/X_anio_simulado_2024.csv: una fila por semana (costos, KPIs,
  control de Gurobi, inventario al empezar y terminar, oferta no admitida,
  edades del inventario que pasa al lunes).
- resultados/X_anio_simulado_dias_2024.csv: una fila por día, con el
  inventario al cierre por edad, vencidas y oferta (para estudiar el vencimiento).
- Al completar el año: resultados/X_kpis_resumen_anio_simulado_2024.csv y
  resultados/X_kpis_hospitales_anio_simulado_2024.csv (como los de `correr_encadenado`).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from codigo import configuracion as cfg
from codigo import horizonte_rodante, kpis
from codigo.evaluacion_historica import intervalo
from codigo.modelo_comun import EstadoSistema
from codigo.simulacion import cargar_semanas, simular_semanas_encadenadas

SEMILLA = 2024
ARCHIVO_SEMANAS = "01_datos/semanas_simuladas_consecutivas_2024.csv"
N_SEMANAS = 52
POLITICAS = {
    "caso_base": "caso_base_agnostico",
    "a": "heuristica_a",
    "b": "heuristica_b_compacta",
}


def archivos(politica):
    """(semanas, días) de una política."""
    base = f"resultados/{POLITICAS[politica]}_anio_simulado"
    return Path(f"{base}_2024.csv"), Path(f"{base}_dias_2024.csv")


def resolvedores(politica):
    """(resolvedores, número de escenarios del modelo) de una política."""
    if politica == "caso_base":
        from codigo import caso_base_agnostico
        return caso_base_agnostico.politica(), 1
    if politica == "a":
        from codigo import heuristica_a
        return {"resolver_calendario": heuristica_a.resolver_modelo_calendario}, cfg.N_ESCENARIOS_DEFAULT
    from codigo import heuristica_b
    return {"resolver_calendario": heuristica_b.resolver_modelo_calendario}, cfg.N_ESCENARIOS_DEFAULT


def archivos_kpis(politica):
    """(resumen de KPIs, fill rate por hospital) de una política."""
    base = f"resultados/{POLITICAS[politica]}"
    return Path(f"{base}_kpis_resumen_anio_simulado_2024.csv"), Path(f"{base}_kpis_hospitales_anio_simulado_2024.csv")


def resumir(politica, insumos):
    """Resume el año completo igual que `correr_encadenado.resumir`: por mes (el
    del lunes) y para el año, cada KPI de `kpis.COLUMNAS_KPI` (los que tenga
    la tabla; ver `kpis.completar_tabla`) con promedio,
    desviación, p5, p95, número de semanas e intervalo del 95%; el año pondera
    las 52 semanas por igual. Además, el fill rate de cada hospital."""
    tabla, columnas = kpis.completar_tabla(pd.read_csv(archivos(politica)[0]))
    tabla["mes"] = pd.to_datetime(tabla["fecha_lunes"]).dt.month
    filas = []
    for periodo, grupo in [*tabla.groupby("mes"), ("año", tabla)]:
        for kpi in columnas:
            x = grupo[kpi].astype(float)
            promedio, inferior, superior, n = intervalo(x)
            filas.append({"periodo": periodo, "kpi": kpi, "promedio": promedio, "desviacion": x.std(ddof=1),
                          "p5": x.quantile(0.05), "p95": x.quantile(0.95), "semanas": n,
                          "inferior": inferior, "superior": superior})
    ruta_resumen, ruta_hospitales = archivos_kpis(politica)
    pd.DataFrame(filas).to_csv(ruta_resumen, index=False, encoding="utf-8")
    kpis.kpis_hospitales(tabla, insumos).to_csv(ruta_hospitales, index=False, encoding="utf-8")
    print(f"Resumen en {ruta_resumen} y {ruta_hospitales}.")


def cargar_anio(insumos):
    """Las 52 semanas simuladas consecutivas, validadas."""
    semanas = cargar_semanas(ARCHIVO_SEMANAS, insumos)
    if len(semanas) != N_SEMANAS or any(s.fecha_lunes.dayofweek != 0 for s in semanas):
        raise ValueError(f"{ARCHIVO_SEMANAS} no tiene {N_SEMANAS} semanas que partan en lunes.")
    if any((b.fecha_lunes - a.fecha_lunes).days != 7 for a, b in zip(semanas, semanas[1:])):
        raise ValueError(f"Las semanas de {ARCHIVO_SEMANAS} no son consecutivas.")
    return semanas


def _estado_desde_fila(fila):
    return EstadoSistema(dia=0, inventario_banco=np.array(json.loads(fila["estado_final_banco"]), dtype=int),
                         inventario_hospitales=np.array(json.loads(fila["estado_final_hospitales"]), dtype=int))


def correr(politica, insumos, semanas, minutos=None):
    ruta, ruta_dias = archivos(politica)
    hechas = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()
    estado = _estado_desde_fila(hechas.iloc[-1]) if len(hechas) else None
    pendientes = semanas[len(hechas):]
    if not pendientes:
        print(f"Las {len(semanas)} semanas ya están en {ruta}.")
        return
    print(f"Política {politica}: {len(hechas)} semanas hechas; parte en la semana {pendientes[0].indice} "
          f"({pendientes[0].fecha_lunes.date()}).", flush=True)
    L = insumos.parametros["shelf_life"]
    pols, n_escenarios = resolvedores(politica)
    if politica == "b":
        from codigo import heuristica_b
        heuristica_b.particion_compacta(insumos)  # arma las zonas una vez (~1-2 min)

    # Registro diario de la realidad: se envuelve la actualización del inventario.
    dias = []
    original = horizonte_rodante.actualizar_estado

    def actualizar_y_registrar(estado_dia, decisiones, oferta_real, demanda_real, insumos_):
        manana, resultado = original(estado_dia, decisiones, oferta_real, demanda_real, insumos_)
        dias.append({
            "dia": estado_dia.dia,
            "oferta": resultado["oferta"],
            "admitida": resultado["admitida"],
            "no_admitida": resultado["oferta"] - resultado["admitida"],
            "despachada": resultado["despachada"],
            "demanda": int(np.sum(demanda_real)),
            "faltante": int(resultado["faltante"].sum()),
            "vencidas_banco": resultado["vencidas_banco"],
            "vencidas_hospitales": int(resultado["vencidas_hospitales"].sum()),
            # Inventario que pasa la noche, por la edad que tendrá mañana (1..L-1).
            **{f"banco_edad{a}": int(manana.inventario_banco[a]) for a in range(1, L)},
            **{f"hosp_edad{a}": int(manana.inventario_hospitales[:, a].sum()) for a in range(1, L)},
        })
        return manana, resultado

    inicio = time.time()

    class Parar(Exception):
        pass

    def guardar(resumen, estado_final):
        fila = dict(resumen)
        fila["fecha_lunes"] = pd.Timestamp(fila["fecha_lunes"]).date().isoformat()
        for a in range(L):
            fila[f"final_edad{a}"] = int(estado_final.inventario_banco[a] + estado_final.inventario_hospitales[:, a].sum())
        fila["estado_final_banco"] = json.dumps(estado_final.inventario_banco.tolist())
        fila["estado_final_hospitales"] = json.dumps(estado_final.inventario_hospitales.tolist())
        pd.DataFrame([fila]).to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")
        pd.DataFrame([{"semana": resumen["semana"], "fecha_lunes": fila["fecha_lunes"], **d} for d in dias]).to_csv(
            ruta_dias, mode="a", header=not ruta_dias.exists(), index=False, encoding="utf-8")
        dias.clear()
        print(f"  semana {resumen['semana']:2d} ({fila['fecha_lunes']}): vencidas {resumen['unidades_vencidas']}, "
              f"no admitida {fila['oferta_no_admitida']}, fill {resumen['fill_rate_red']:.1%}, "
              f"costo operacional {resumen['costo_operacional']:,.0f}, "
              f"óptimas {resumen['calendario_optimo']}/{resumen['cantidades_optimas']}, {resumen['tiempo_s']:.0f} s",
              flush=True)
        if minutos is not None and time.time() - inicio > 60 * minutos:
            raise Parar

    horizonte_rodante.actualizar_estado = actualizar_y_registrar
    try:
        simular_semanas_encadenadas(pendientes, insumos, n_escenarios, SEMILLA, estado_inicial=estado,
                                    al_terminar_semana=guardar, **pols)
    except Parar:
        print(f"Se cumplieron los {minutos} minutos; para seguir, correr el mismo comando.")
    finally:
        horizonte_rodante.actualizar_estado = original


def main(argumentos=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("politica", choices=list(POLITICAS))
    parser.add_argument("--minutos", type=float, default=None, help="no empezar semanas nuevas pasado este tiempo")
    parser.add_argument("--resumir", action="store_true", help="solo armar los archivos de KPIs de una corrida ya hecha")
    args = parser.parse_args(argumentos)
    insumos = horizonte_rodante.preparar_insumos()
    semanas = cargar_anio(insumos)
    if not args.resumir:
        correr(args.politica, insumos, semanas, args.minutos)
    ruta, _ = archivos(args.politica)
    if ruta.exists() and len(pd.read_csv(ruta)) == N_SEMANAS:
        resumir(args.politica, insumos)
    elif args.resumir:
        print(f"{ruta} no tiene las {N_SEMANAS} semanas todavía.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
