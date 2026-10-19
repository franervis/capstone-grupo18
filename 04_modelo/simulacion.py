"""Simulación de evaluación: corre la política sobre semanas "reales" simuladas.

Una semana simulada funciona así (ver METODOLOGIA.md, §11):
0. Antes del lunes se sortea la "realidad" de los 7 días (oferta y demanda)
   y se guarda. El modelo no la ve.
1. Se corre `ejecutar_semana`, que revela la realidad un día a la vez: la
   oferta al inicio del día y la demanda después de despachar.
2. Al final se suman los costos y unidades de la semana: un resultado por semana.

Las semanas "reales" se generan con las mismas distribuciones que los
escenarios (Normal del mes del lunes para la oferta, independiente cada día; distribución
de cada hospital para la demanda), pero con un generador aleatorio distinto,
así que son independientes de los escenarios del modelo. Para comparar
políticas, todas deben correr sobre las mismas semanas reales.

Los KPIs (`codigo/kpis.py`, METODOLOGIA.md §14) se agregan al resumen de
cada semana en `simular_semana`. El caso base NO está aquí todavía
(PENDIENTES.md).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from codigo import configuracion as cfg
from codigo import modelo_comun
from codigo import kpis
from codigo.escenarios import generar_escenarios
from codigo.horizonte_rodante import ejecutar_semana, estado_inicial_semana


@dataclass
class SemanaReal:
    """La "realidad" de una semana.

    Atributos:
        fecha_lunes: lunes de la semana.
        oferta: arreglo de largo 7 con la oferta real de cada día.
        demanda: arreglo (7, n_hospitales) en el orden de Insumos.hospitales.
        indice: número de la semana dentro de su conjunto (0, 1, 2, ...).
    """

    fecha_lunes: pd.Timestamp
    oferta: np.ndarray
    demanda: np.ndarray
    indice: int = 0


def generadores(semilla=cfg.SEMILLA):
    """Devuelve dos generadores independientes derivados de `semilla`:
    (para semanas reales, para escenarios del modelo)."""
    real, modelo = np.random.SeedSequence(semilla).spawn(2)
    return np.random.default_rng(real), np.random.default_rng(modelo)


def generar_semanas_reales(fecha_lunes, n_semanas, insumos, rng):
    """Sortea `n_semanas` semanas "reales" independientes que parten en `fecha_lunes`.

    Supuesto vigente (PENDIENTES.md): los 7 días usan el mes del lunes.

    Recibe: fecha_lunes, n_semanas (int), insumos, rng (numpy Generator).
    Devuelve: lista de SemanaReal.
    """
    fecha_lunes = pd.Timestamp(fecha_lunes)
    horizonte = insumos.parametros["horizonte"]
    sorteo = generar_escenarios(
        horizonte,
        fecha_lunes.month,
        insumos.distribuciones_demanda,
        n_semanas,
        distribuciones_oferta=insumos.distribuciones_oferta,
        rng=rng,
    )
    columnas = [sorteo.hospitales.index(nombre) for nombre in insumos.hospitales["nombre"]]
    return [
        SemanaReal(fecha_lunes, sorteo.oferta[s].copy(), sorteo.demanda[s][:, columnas].copy(), indice=s)
        for s in range(n_semanas)
    ]


def simular_semana(semana, insumos, n_escenarios=cfg.N_ESCENARIOS_DEFAULT, rng=None, estado_inicial=None, **resolvedores):
    """Corre la política sobre una semana real y resume lo que pasó.

    Recibe: semana (SemanaReal), insumos, n_escenarios, rng (numpy
        Generator para los escenarios del modelo) y, opcionalmente,
        resolver_calendario / resolver_cantidades (ver `ejecutar_semana`).
        estado_inicial: inventario con que parte el lunes (EstadoSistema). Por
            defecto, el 10% de la capacidad con edad 0, como si la semana fuera
            independiente; en semanas encadenadas, el que dejó la semana anterior.
    Devuelve: (resumen, calendario, registros). El resumen es un diccionario
        con costos y unidades de la semana, calculados con lo que realmente pasó:
        costo de transporte, holding (banco y hospitales), shortage, wastage,
        penalización terminal y total; km; demanda total, faltante y
        unidades vencidas.
    """
    par = insumos.parametros
    dia = lambda fecha: (pd.Timestamp(fecha) - semana.fecha_lunes).days

    inicial = estado_inicial_semana(insumos) if estado_inicial is None else estado_inicial
    calendario, registros = ejecutar_semana(
        semana.fecha_lunes,
        inicial,
        observar_oferta=lambda fecha: int(semana.oferta[dia(fecha)]),
        observar_demanda=lambda fecha: semana.demanda[dia(fecha)],
        insumos=insumos,
        n_escenarios=n_escenarios,
        rng=rng,
        **resolvedores,
    )

    faltante = sum(r["faltante"].sum() for r in registros)
    vencidas = sum(r["vencidas_banco"] + r["vencidas_hospitales"].sum() for r in registros)
    noche_banco = sum(r["noche_banco"] for r in registros)
    noche_hospitales = sum(r["noche_hospitales"].sum() for r in registros)

    # Penalización terminal con el inventario real que pasa a la semana siguiente.
    final = registros[-1]["estado_final"]
    mu_b, mu_i = modelo_comun.minimos_terminales(insumos)
    deficit = max(0.0, mu_b - final.inventario_banco.sum())
    deficit += np.maximum(0.0, mu_i - final.inventario_hospitales.sum(axis=1)).sum()

    costos = {
        "costo_transporte": par["costo_km"] * calendario.distancia_km,
        "costo_holding_banco": par["costo_holding_banco"] * noche_banco,
        "costo_holding_hospitales": par["costo_holding_hospital"] * noche_hospitales,
        "costo_shortage": par["costo_shortage"] * faltante,
        "costo_wastage": par["costo_wastage"] * vencidas,
        "penalizacion_terminal": par["costo_shortage"] * deficit,
    }
    resumen = {
        "fecha_lunes": semana.fecha_lunes,
        **costos,
        "costo_total": sum(costos.values()),
        "km": calendario.distancia_km,
        "demanda_total": int(semana.demanda.sum()),
        "unidades_faltantes": int(faltante),
        "unidades_vencidas": int(vencidas),
        "deficit_terminal": float(deficit),
        "calendario_optimo": calendario.info_resolucion["optimo"],
        "calendario_gap": calendario.info_resolucion["gap"],
        # Esfuerzo real del calendario (unidades de trabajo de Gurobi, no dependen del reloj).
        # Si es muy bajo para el tiempo que tomó, el computador se suspendió a mitad de la corrida.
        "calendario_tiempo_s": calendario.info_resolucion.get("tiempo_s"),
        "calendario_trabajo": calendario.info_resolucion.get("trabajo"),
        # False si algún modelo diario se cortó por tiempo antes del óptimo (por ejemplo,
        # si el computador se suspendió a mitad de la corrida). Esas semanas hay que repetirlas.
        # Si una decisión no trae información de Gurobi (p. ej., una heurística sin modelo), cuenta como óptima.
        "cantidades_optimas": all(r["decisiones"].info_resolucion.get("optimo", True) for r in registros),
    }

    # KPIs (METODOLOGIA.md §14): KPIs de la red y demanda/faltante por hospital.
    resumen.update(kpis.kpis_semana(semana, calendario, registros, insumos, inicial))
    resumen.update(kpis.detalle_hospitales(semana, registros))
    return resumen, calendario, registros


def generar_conjunto_anual(semanas_por_mes, insumos, anio=2024, semilla=cfg.SEMILLA):
    """Sortea un conjunto de semanas reales que cubre todo el año: `semanas_por_mes`
    semanas para cada mes, todas partiendo el primer lunes del mes.

    Da lo mismo qué semana del mes se use: para el modelo, lo único que depende
    de la fecha es el mes del lunes (Normal de la oferta); la demanda y el
    inventario inicial son iguales todo el año.

    Las semanas quedan numeradas de 0 en adelante, mes por mes: las del mes m
    (1–12) son las semanas semanas_por_mes·(m−1) hasta semanas_por_mes·m − 1.
    Dentro de cada mes, las primeras n semanas forman un subconjunto anidado.

    Devuelve: lista de SemanaReal.
    """
    rng_real, _ = generadores(semilla)
    semanas = []
    for mes in range(1, 13):
        primer_dia = pd.Timestamp(anio, mes, 1)
        lunes = primer_dia + pd.Timedelta(days=(cfg.DIA_INICIO_SEMANA - primer_dia.dayofweek) % 7)
        for semana in generar_semanas_reales(lunes, semanas_por_mes, insumos, rng_real):
            semana.indice = len(semanas)
            semanas.append(semana)
    return semanas


def generador_modelo_semana(semilla, indice):
    """Generador de los escenarios del modelo para la semana `indice`.

    Cada semana tiene su propio generador, derivado de la semilla y del número
    de la semana. Así los escenarios de una semana no dependen de qué otras
    semanas se corrieron antes. Por ejemplo, las primeras 30 semanas de un
    conjunto de 100 dan lo mismo que correr solo esas 30, y distintas
    personas pueden correr partes del conjunto por separado.
    """
    _, modelo = np.random.SeedSequence(semilla).spawn(2)
    return np.random.default_rng(modelo.spawn(indice + 1)[indice])


def guardar_semanas(semanas, ruta, insumos):
    """Guarda un conjunto de semanas reales en un CSV, con una fila por semana
    y día: semana, fecha_lunes, dia, oferta y una columna de demanda por hospital."""
    filas = []
    for semana in semanas:
        for dia in range(len(semana.oferta)):
            fila = {"semana": semana.indice, "fecha_lunes": semana.fecha_lunes.date(), "dia": dia,
                    "oferta": int(semana.oferta[dia])}
            fila.update({nombre: int(semana.demanda[dia, i]) for i, nombre in enumerate(insumos.hospitales["nombre"])})
            filas.append(fila)
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8")


def cargar_semanas(ruta, insumos):
    """Carga un conjunto de semanas reales guardado con `guardar_semanas`.

    Devuelve: lista de SemanaReal, en orden de semana.
    """
    df = pd.read_csv(ruta, encoding="utf-8")
    nombres = list(insumos.hospitales["nombre"])
    semanas = []
    for indice, grupo in df.groupby("semana", sort=True):
        grupo = grupo.sort_values("dia")
        semanas.append(SemanaReal(
            fecha_lunes=pd.Timestamp(grupo["fecha_lunes"].iloc[0]),
            oferta=grupo["oferta"].to_numpy(dtype=int),
            demanda=grupo[nombres].to_numpy(dtype=int),
            indice=int(indice),
        ))
    return semanas


def simular_semanas(semanas, insumos, n_escenarios=cfg.N_ESCENARIOS_DEFAULT, semilla=cfg.SEMILLA, **resolvedores):
    """Corre la política sobre un conjunto de semanas reales ya generado
    (por ejemplo, cargado con `cargar_semanas`).

    Recibe: semanas (lista de SemanaReal), insumos, n_escenarios, semilla
        (para los escenarios del modelo) y, opcionalmente,
        resolver_calendario / resolver_cantidades. Con las mismas semanas y la
        misma semilla, todas las políticas enfrentan exactamente lo mismo.
    Devuelve: DataFrame con una fila por semana (ver `simular_semana`), con
        la columna "semana" y el tiempo de cada una.
    """
    filas = []
    for semana in semanas:
        inicio = pd.Timestamp.now()
        resumen = simular_semana(semana, insumos, n_escenarios,
                                 generador_modelo_semana(semilla, semana.indice), **resolvedores)[0]
        resumen["semana"] = semana.indice
        resumen["tiempo_s"] = (pd.Timestamp.now() - inicio).total_seconds()
        filas.append(resumen)
    return pd.DataFrame(filas)


def simular_semanas_encadenadas(semanas, insumos, n_escenarios=cfg.N_ESCENARIOS_DEFAULT, semilla=cfg.SEMILLA,
                                estado_inicial=None, al_terminar_semana=None, **resolvedores):
    """Corre la política sobre semanas CONSECUTIVAS: cada lunes parte con el
    inventario real que dejó el domingo anterior, con sus edades (lo que sobró no
    se bota). Solo la primera semana parte con `estado_inicial` (por defecto, el
    10% de la capacidad con edad 0).

    Recibe: semanas (lista de SemanaReal, cada lunes 7 días después del anterior),
        insumos, n_escenarios, semilla (escenarios del modelo, uno por semana como
        en `simular_semanas`), estado_inicial, al_terminar_semana (función
        opcional (resumen, estado_final) que se llama al cerrar cada semana, por
        ejemplo para guardar avance) y resolver_calendario / resolver_cantidades.
    Devuelve: DataFrame con una fila por semana, como `simular_semanas`, más el
        inventario con que partió y terminó cada semana.
    """
    import dataclasses

    estado = estado_inicial_semana(insumos) if estado_inicial is None else estado_inicial
    filas, lunes_anterior = [], None
    for semana in semanas:
        if lunes_anterior is not None and semana.fecha_lunes != lunes_anterior + pd.Timedelta(days=7):
            raise ValueError(f"La semana del {semana.fecha_lunes.date()} no sigue a la anterior.")
        lunes_anterior = semana.fecha_lunes
        inicial = dataclasses.replace(estado, dia=0)
        inicio = pd.Timestamp.now()
        resumen, _, registros = simular_semana(semana, insumos, n_escenarios, generador_modelo_semana(semilla, semana.indice),
                                               estado_inicial=inicial, **resolvedores)
        estado = registros[-1]["estado_final"]
        resumen.update({
            "semana": semana.indice,
            "tiempo_s": (pd.Timestamp.now() - inicio).total_seconds(),
            "inventario_inicial_banco": int(inicial.inventario_banco.sum()),
            "inventario_inicial_hospitales": int(inicial.inventario_hospitales.sum()),
            "inventario_final_banco": int(estado.inventario_banco.sum()),
            "inventario_final_hospitales": int(estado.inventario_hospitales.sum()),
        })
        filas.append(resumen)
        if al_terminar_semana is not None:
            al_terminar_semana(resumen, estado)
    return pd.DataFrame(filas)


def simular(fecha_lunes, n_semanas, insumos, n_escenarios=cfg.N_ESCENARIOS_DEFAULT, semilla=cfg.SEMILLA, **resolvedores):
    """Simulación de evaluación completa: sortea `n_semanas` semanas reales
    que parten en `fecha_lunes` y corre la política sobre cada una.

    Recibe: fecha_lunes, n_semanas (int), insumos, n_escenarios, semilla y,
        opcionalmente, resolver_calendario / resolver_cantidades. Con la misma
        semilla, distintas políticas corren sobre las mismas semanas reales.
    Devuelve: DataFrame con una fila por semana (ver `simular_semana`).
    """
    rng_real, _ = generadores(semilla)
    semanas = generar_semanas_reales(fecha_lunes, n_semanas, insumos, rng_real)
    return simular_semanas(semanas, insumos, n_escenarios, semilla, **resolvedores)
