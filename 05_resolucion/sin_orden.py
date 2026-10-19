"""Modelo común sin la decisión del orden de la ruta.

El modelo original decide tres cosas junto con el inventario:
 (A) a qué hospitales va cada vehículo cada día (variables y),
 (B) en qué orden los visita (variables x: arcos de ruta),
 (C) cuánto lleva a cada uno.
Este modelo mantiene (A) y (C) exactamente iguales. Reemplaza (B) por una
TABLA ya calculada con Held-Karp: para cada grupo posible de hospitales, los
km del mejor orden. El modelo elige libremente qué hospitales van juntos; el
costo de transporte lo calcula a partir del grupo elegido, usando la tabla.

Así se aisla el efecto de la decisión del orden. Si este modelo se resuelve
rápido, el orden era la causa; si sigue lento, no lo era.

Supuesto: distancias simétricas (en nuestros datos tienen UN par asimétrico
muy chico, lo que afecta a lo más 0,14 km en una ruta de ~100 km).
"""

import gurobipy as gp
from gurobipy import GRB

from codigo import configuracion as cfg
from codigo import modelo_comun, generacion_columnas
from codigo.modelo_comun import SolucionCalendario


def resolver_sin_orden(escenarios, estado_inicial, insumos):
    n_h = len(insumos.hospitales)
    hospitales = list(range(n_h))
    vehiculos = list(insumos.flota["vehiculo"])
    dias = range(escenarios.dias)
    alfa = insumos.parametros["costo_km"]
    tabla = generacion_columnas.costos_rutas(insumos, hospitales)
    # Grupos no vacíos indexados por un entero; incluye todas las combinaciones.
    grupos = list(tabla.keys())
    km_grupo = [tabla[g]["km"] for g in grupos]
    pertenece = {(i, r): (i in grupos[r]) for i in hospitales for r in range(len(grupos))}

    modelo = gp.Model("sin_orden")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)

    # (A) y (C): iguales al modelo común.
    y = modelo.addVars(hospitales, vehiculos, dias, vtype=GRB.BINARY, name="y")
    w = modelo.addVars(vehiculos, dias, vtype=GRB.BINARY, name="w")
    for k in vehiculos:
        for t in dias:
            for i in hospitales:
                modelo.addConstr(y[i, k, t] <= w[k, t])

    # (B'): el modelo elige UN grupo por (vehículo, día), que debe calzar con sus visitas.
    z = modelo.addVars(range(len(grupos)), vehiculos, dias, vtype=GRB.BINARY, name="z")
    for k in vehiculos:
        for t in dias:
            modelo.addConstr(gp.quicksum(z[r, k, t] for r in range(len(grupos))) == w[k, t], name=f"un_grupo[{k},{t}]")
            for i in hospitales:
                # y[i,k,t] = 1 si y solo si i está en el grupo elegido para (k, t).
                modelo.addConstr(y[i, k, t] == gp.quicksum(z[r, k, t] for r in range(len(grupos)) if pertenece[i, r]),
                                 name=f"calza[{i},{k},{t}]")

    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado_inicial, insumos,
        visita=lambda i, k, t: y[i, k, t], hoy_comun=False,
    )
    transporte = alfa * gp.quicksum(km_grupo[r] * z[r, k, t] for r in range(len(grupos)) for k in vehiculos for t in dias)
    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)
    info = modelo_comun._resolver(modelo)

    # Reconstruir la SolucionCalendario con el orden de Held-Karp.
    hoy = estado_inicial.dia
    visitas, rutas, km = set(), {}, 0.0
    for k in vehiculos:
        for t in dias:
            if w[k, t].X < 0.5:
                continue
            r = next(r for r in range(len(grupos)) if z[r, k, t].X > 0.5)
            visitas |= {(hoy + t, i, k) for i in grupos[r]}
            rutas[hoy + t, k] = list(tabla[grupos[r]]["orden"])
            km += tabla[grupos[r]]["km"]
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)
