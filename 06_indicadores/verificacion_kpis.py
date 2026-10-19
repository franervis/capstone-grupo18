"""Verificación de los KPIs (METODOLOGIA.md §14).

Se corre desde la raíz del repo con:

    python -m codigo.verificacion_kpis

Imprime OK / FALLA para cada prueba y termina con código de salida 1 si
alguna falla. Hay que correrlo cada vez que alguien cambie `codigo/kpis.py` o
lo que entrega `simulacion.simular_semana`. No resuelve ningún modelo de
Gurobi, así que toma segundos.

Tres pruebas:
1. Caso calculado a mano: una semana inventada de 2 hospitales y 2 días, con
   todos los KPIs calculados a mano.
2. Hospitales y resumen entre semanas: el fill rate por hospital se calcula
   con los totales del conjunto de semanas, el resumen entrega los
   estadísticos correctos y las tablas guardadas sin la tasa de oferta no
   admitida se completan.
3. Consistencia con la simulación: en una semana simulada con una política
   simple (sin optimización), los KPIs calzan con el resumen de
   `simular_semana`.
"""

import sys

import numpy as np
import pandas as pd

from codigo import kpis, modelo_comun
from codigo.horizonte_rodante import Insumos
from codigo.modelo_comun import DecisionesDia, EstadoSistema, SolucionCalendario
from codigo.simulacion import SemanaReal, generadores, generar_semanas_reales, simular_semana
from codigo.verificacion import _iguales, _insumos_chicos


def _comparar(obtenido, esperado):
    """Lista de diferencias entre dos diccionarios de KPIs (vacía si calzan)."""
    problemas = []
    for clave, valor in esperado.items():
        if isinstance(valor, str):
            calza = obtenido[clave] == valor
        else:
            calza = _iguales(obtenido[clave], valor)
        if not calza:
            problemas.append(f"{clave}: {obtenido[clave]} en vez de {valor}")
    return problemas


# ---------------------------------------------------------------------------
# 1. Caso calculado a mano
# ---------------------------------------------------------------------------


def _caso_a_mano():
    """Semana inventada: 2 hospitales (A y B), 2 vehículos y 2 días."""
    insumos = Insumos(
        parametros={
            "shelf_life": 5, "horizonte": 2,
            "inventario_inicial_banco_factor": 0.1, "inventario_inicial_hospital_factor": 0.1,
            "costo_km": 10, "costo_holding_banco": 1, "costo_holding_hospital": 2,
            "costo_shortage": 100, "costo_wastage": 50,
        },
        hospitales=pd.DataFrame({"nombre": ["A", "B"], "capacidad_maxima": [50, 20]}),
        nombre_banco="Banco",
        capacidad_banco=100,
        flota=pd.DataFrame({"vehiculo": [0, 1], "capacidad": [40, 40]}),
        distancias=None,
        distribuciones_demanda={},
        distribuciones_oferta={},
    )
    semana = SemanaReal(
        fecha_lunes=pd.Timestamp("2024-09-02"),
        oferta=np.array([20, 22]),  # el día 1 se admiten 17 de 22
        demanda=np.array([[10, 4], [10, 6]]),  # A pide 20 en la semana; B pide 10
    )
    # Día 0: el vehículo 0 visita a A y B. Día 1: el 0 visita a A y el 1 a B.
    calendario = SolucionCalendario(visitas={(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)}, distancia_km=30.0)
    registros = [
        {
            "decisiones": DecisionesDia(oferta_admitida=20, despachos={(0, 0, 0): 15, (1, 0, 0): 5}),
            "admitida": 20, "despachada": 20,
            "faltante": np.array([2, 0]),
            "vencidas_banco": 1, "vencidas_hospitales": np.array([0, 0]),
            "noche_banco": 12, "noche_hospitales": np.array([3, 2]),
        },
        {
            "decisiones": DecisionesDia(oferta_admitida=17, despachos={(0, 0, 1): 10, (1, 1, 0): 6}),
            "admitida": 17, "despachada": 16,
            "faltante": np.array([3, 1]),
            "vencidas_banco": 0, "vencidas_hospitales": np.array([0, 2]),
            "noche_banco": 8, "noche_hospitales": np.array([1, 4]),
            # Quedan 4 unidades en el banco, 5 en A y 1 en B.
            "estado_final": EstadoSistema(
                dia=2,
                inventario_banco=np.array([0, 4, 0, 0, 0]),
                inventario_hospitales=np.array([[0, 5, 0, 0, 0], [0, 1, 0, 0, 0]]),
            ),
        },
    ]
    return insumos, semana, calendario, registros


def prueba_caso_a_mano():
    """Todos los KPIs de una semana inventada, contra el cálculo a mano."""
    insumos, semana, calendario, registros = _caso_a_mano()
    esperado = {
        # Costos: transporte 10·30; holding 1·(12+8) + 2·(3+2+1+4); shortage 100·6; wastage 50·3.
        "costo_transporte": 300,
        "costo_holding": 40,
        "costo_shortage": 600,
        "costo_wastage": 150,
        "costo_operacional": 1090,
        "participacion_transporte": 300 / 1090,
        "participacion_holding": 40 / 1090,
        "participacion_shortage": 600 / 1090,
        "participacion_wastage": 150 / 1090,
        # Mínimos: banco 10, A 5, B 2. Quedan 4, 5 y 1: faltan 6 + 0 + 1.
        "deficit_terminal": 7,
        # Faltan 2+0+3+1 = 6 de una demanda de 30.
        "fill_rate_red": 0.8,
        "demanda_total": 30,
        "unidades_faltantes": 6,
        # Vencen 1 + 2 = 3. Entraron: inicial 10 + 5 + 2 = 17, más 20 + 17 admitidas.
        "unidades_vencidas": 3,
        "wastage_rate": 3 / 54,
        # Se ofrecen 20 + 22 = 42 y se admiten 20 + 17 = 37: quedan fuera 5.
        "oferta_total": 42,
        "oferta_no_admitida": 5,
        "tasa_oferta_no_admitida": 5 / 42,
        # 36 unidades despachadas en 30 km. Salieron 3 vehículos de 40: capacidad 120.
        "km": 30,
        "unidades_despachadas": 36,
        "km_por_unidad": 30 / 36,
        "utilizacion_flota": 36 / 120,
        "visitas": 4,
    }
    problemas = _comparar(kpis.kpis_semana(semana, calendario, registros, insumos), esperado)

    detalle = kpis.detalle_hospitales(semana, registros)
    problemas += _comparar(detalle, {"demanda_h00": 20, "faltante_h00": 5, "demanda_h01": 10, "faltante_h01": 1})

    # A: 1 − 5/20 = 0.75. B: 1 − 1/10 = 0.9.
    hospitales = kpis.kpis_hospitales(pd.DataFrame([detalle]), insumos)
    if not np.allclose(hospitales["fill_rate"], [0.75, 0.9]):
        problemas.append(f"fill rate por hospital {hospitales['fill_rate'].tolist()} en vez de [0.75, 0.9]")
    problemas += _comparar(kpis.equidad(hospitales), {
        "fill_rate_minimo": 0.75, "hospital_fill_rate_minimo": "A", "fill_rate_maximo": 0.9, "brecha_fill_rate": 0.15,
    })
    return not problemas, "; ".join(problemas) or "costo 1090, fill rate 0.80, wastage rate 3/54, oferta no admitida 5/42, utilización 0.30"


# ---------------------------------------------------------------------------
# 2. Hospitales y resumen entre semanas
# ---------------------------------------------------------------------------


def prueba_hospitales_y_resumen():
    """Fill rate por hospital con los totales del conjunto, y estadísticos entre semanas."""
    insumos = _caso_a_mano()[0]
    # A pide 1 y le falta 1 (fill rate 0) y después pide 9 y no le falta nada (fill rate 1).
    # Con los totales: 1 − 1/10 = 0.9, no el promedio semanal 0.5.
    tabla = pd.DataFrame([
        {"fecha_lunes": "2024-09-02", "costo_operacional": 100.0,
         "demanda_h00": 1, "faltante_h00": 1, "demanda_h01": 10, "faltante_h01": 0},
        {"fecha_lunes": "2024-09-02", "costo_operacional": 300.0,
         "demanda_h00": 9, "faltante_h00": 0, "demanda_h01": 10, "faltante_h01": 1},
    ])
    problemas = []

    hospitales = kpis.kpis_hospitales(tabla, insumos)
    if not np.allclose(hospitales["fill_rate"], [0.9, 0.95]):
        problemas.append(f"fill rate por hospital {hospitales['fill_rate'].tolist()} en vez de [0.9, 0.95]")
    problemas += _comparar(kpis.equidad(hospitales), {
        "fill_rate_minimo": 0.9, "hospital_fill_rate_minimo": "A", "brecha_fill_rate": 0.05,
    })

    # Costos 100 y 300: promedio 200, desviación 141.42, p5 = 110 y p95 = 290.
    resumen = kpis.resumir_semanas(tabla, columnas=["costo_operacional"])
    esperado = {"promedio": 200, "desviacion": 2 ** 0.5 * 100, "p5": 110, "p95": 290, "semanas": 2}
    for periodo in (9, "año"):
        fila = resumen.loc[(periodo, "costo_operacional")]
        problemas += [f"{periodo}: {p}" for p in _comparar(fila.to_dict(), esperado)]

    # Tabla guardada sin la tasa de oferta no admitida: se calcula, y los KPIs
    # que la tabla no tiene se omiten.
    guardada = tabla.assign(oferta_total=[100, 50], oferta_no_admitida=[20, 0])
    completa, columnas = kpis.completar_tabla(guardada)
    if not np.allclose(completa["tasa_oferta_no_admitida"], [0.2, 0.0]):
        problemas.append(f"tasa de oferta no admitida {completa['tasa_oferta_no_admitida'].tolist()} en vez de [0.2, 0.0]")
    if columnas != ["costo_operacional", "oferta_no_admitida", "tasa_oferta_no_admitida"]:
        problemas.append(f"KPIs disponibles {columnas}")

    # Meses con distinto número de semanas: el resumen del año debe fallar.
    desbalanceada = pd.concat([tabla, tabla.head(1).assign(fecha_lunes="2024-10-07")])
    try:
        kpis.resumir_semanas(desbalanceada, columnas=["costo_operacional"])
        problemas.append("no falló con meses de distinto número de semanas")
    except ValueError:
        pass
    return not problemas, "; ".join(problemas) or "fill rate de A = 0.90 (promedio semanal sería 0.50); promedio 200"


# ---------------------------------------------------------------------------
# 3. Consistencia con la simulación
# ---------------------------------------------------------------------------


def _calendario_simple(escenarios, estado, insumos):
    """Política de prueba (NO es el caso base): el primer vehículo visita
    todos los hospitales todos los días, en el orden de Insumos.hospitales."""
    hospitales = range(len(insumos.hospitales))
    k = insumos.flota["vehiculo"].iloc[0]
    delta = modelo_comun.matriz_distancias(insumos)
    nodos = [0] + [i + 1 for i in hospitales] + [0]
    largo = sum(delta[m, n] for m, n in zip(nodos[:-1], nodos[1:]))
    dias = range(estado.dia, estado.dia + escenarios.dias)
    return SolucionCalendario(
        visitas={(t, i, k) for t in dias for i in hospitales},
        rutas={(t, k): list(hospitales) for t in dias},
        distancia_km=largo * escenarios.dias,
        info_resolucion={"optimo": True, "gap": 0.0},
    )


def _cantidades_simples(escenarios, estado, calendario, insumos):
    """Política de prueba (NO es el caso base): admite toda la oferta que cabe
    en el banco y llena cada hospital visitado hasta su capacidad, con las
    unidades más viejas primero, mientras quede carga en el vehículo."""
    par = insumos.parametros
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    C_i = insumos.hospitales["capacidad_maxima"].to_numpy()

    banco = estado.inventario_banco.astype(int).copy()
    admitida = int(min(escenarios.oferta[0, 0], insumos.capacidad_banco - banco.sum()))
    banco[0] += admitida

    despachos, carga = {}, {k: 0 for k in theta}
    for (t, i, k) in sorted(calendario.visitas):
        if t != estado.dia:
            continue
        espacio = int(C_i[i] - estado.inventario_hospitales[i].sum())
        for a in range(par["shelf_life"] - 1, -1, -1):
            unidades = int(min(espacio, banco[a], theta[k] - carga[k]))
            if unidades > 0:
                despachos[i, k, a] = unidades
                banco[a] -= unidades
                carga[k] += unidades
                espacio -= unidades
    return DecisionesDia(oferta_admitida=admitida, despachos=despachos)


def prueba_consistencia_simulacion():
    """Los KPIs de una semana simulada calzan con el resumen de `simular_semana`."""
    ins = _insumos_chicos(["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], n_vehiculos=2)
    rng_real, rng_modelo = generadores(5)
    semana = generar_semanas_reales(pd.Timestamp("2024-09-02"), 1, ins, rng_real)[0]
    resumen, calendario, registros = simular_semana(
        semana, ins, n_escenarios=1, rng=rng_modelo,
        resolver_calendario=_calendario_simple, resolver_cantidades=_cantidades_simples,
    )
    k = kpis.kpis_semana(semana, calendario, registros, ins)
    detalle = kpis.detalle_hospitales(semana, registros)

    problemas = _comparar(k, {
        "costo_operacional": resumen["costo_total"] - resumen["penalizacion_terminal"],
        "deficit_terminal": resumen["deficit_terminal"],
        "demanda_total": resumen["demanda_total"],
        "unidades_faltantes": resumen["unidades_faltantes"],
        "unidades_vencidas": resumen["unidades_vencidas"],
        "km": resumen["km"],
        "fill_rate_red": 1 - resumen["unidades_faltantes"] / resumen["demanda_total"],
    })
    participaciones = sum(k[f"participacion_{c}"] for c in ("transporte", "holding", "shortage", "wastage"))
    if not _iguales(participaciones, 1):
        problemas.append(f"las participaciones del costo suman {participaciones}")
    n_h = len(ins.hospitales)
    if sum(detalle[f"{kpis.PREFIJO_DEMANDA}{i:02d}"] for i in range(n_h)) != k["demanda_total"]:
        problemas.append("la demanda por hospital no suma la demanda total")
    if sum(detalle[f"{kpis.PREFIJO_FALTANTE}{i:02d}"] for i in range(n_h)) != k["unidades_faltantes"]:
        problemas.append("el faltante por hospital no suma el faltante total")
    admitida = sum(r["admitida"] for r in registros)
    if k["oferta_total"] != semana.oferta.sum() or k["oferta_no_admitida"] != k["oferta_total"] - admitida:
        problemas.append(f"oferta {k['oferta_total']} / no admitida {k['oferta_no_admitida']} no calzan con los registros")
    if not 0 <= k["utilizacion_flota"] <= 1:
        problemas.append(f"utilización de la flota {k['utilizacion_flota']} fuera de [0, 1]")
    return not problemas, "; ".join(problemas) or (
        f"costo operacional {k['costo_operacional']:,.0f} = costo total − penalización; "
        f"fill rate {k['fill_rate_red']:.3f}"
    )


PRUEBAS = [
    ("1. KPIs de una semana calculados a mano", prueba_caso_a_mano),
    ("2. Fill rate por hospital y resumen entre semanas", prueba_hospitales_y_resumen),
    ("3. Consistencia con el resumen de la simulación", prueba_consistencia_simulacion),
]


def main():
    fallas = 0
    for nombre, prueba in PRUEBAS:
        try:
            ok, detalle = prueba()
        except Exception as error:  # una prueba que se cae cuenta como falla
            ok, detalle = False, f"error: {error!r}"
        fallas += not ok
        print(f"[{'OK   ' if ok else 'FALLA'}] {nombre}: {detalle}")
    print(f"\n{len(PRUEBAS) - fallas} de {len(PRUEBAS)} pruebas OK.")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
