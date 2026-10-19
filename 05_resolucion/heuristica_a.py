"""Heurística A: separar el modelo de calendario en inventario y ruteo.

Sólo cambia el modelo de calendario; el modelo de cantidades (cada día) es el
mismo del modelo común, porque ya no tiene ruteo.

Paso 1 — Inventario: el mismo modelo de calendario del modelo común
    (`modelo_comun._agregar_inventario`) con las variables de visita y, pero
    SIN variables de ruteo (x, w, rho) y SIN costo de transporte. Decide qué
    hospitales visita cada vehículo cada día y las cantidades por escenario.
Paso 2 — Ruteo: con las visitas fijas, para cada vehículo y día se busca el
    orden de visita más corto (problema del vendedor viajero, resuelto al
    óptimo con Gurobi).

El costo de la heurística es justamente que el paso 1 decide sin saber nada
del ruteo.

Uso (misma firma que el modelo común):

    from codigo import heuristica_a
    simular(..., resolver_calendario=heuristica_a.resolver_modelo_calendario)
"""

import gurobipy as gp
from gurobipy import GRB

from codigo import configuracion as cfg
from codigo import modelo_comun
from codigo.modelo_comun import SolucionCalendario


def resolver_inventario(escenarios, estado_inicial, insumos):
    """Paso 1: visitas y cantidades sin ruteo ni costo de transporte.

    Devuelve: (conjunto de visitas (t, i, k), información de Gurobi).
    """
    hospitales = range(len(insumos.hospitales))
    vehiculos = list(insumos.flota["vehiculo"])
    dias = range(escenarios.dias)

    modelo = gp.Model("heuristica_a_inventario")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)
    y = modelo.addVars(hospitales, vehiculos, dias, vtype=GRB.BINARY, name="y")
    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado_inicial, insumos,
        visita=lambda i, k, t: y[i, k, t], hoy_comun=False,
    )
    modelo.setObjective(costo_inventario, GRB.MINIMIZE)
    info = modelo_comun._resolver(modelo)

    hoy = estado_inicial.dia
    visitas = {(hoy + t, i, k) for i in hospitales for k in vehiculos for t in dias if y[i, k, t].X > 0.5}
    return visitas, info


def ruta_mas_corta(hospitales_visitados, delta):
    """Paso 2 para un vehículo y un día: orden de visita más corto que sale
    del banco, pasa por todos los hospitales dados y vuelve al banco.

    Recibe: lista de hospitales i; delta = `modelo_comun.matriz_distancias`
        (posición 0 = banco, i + 1 = hospital i).
    Devuelve: (lista de hospitales en orden de visita, km de la ruta).
    """
    if not hospitales_visitados:
        return [], 0.0
    if len(hospitales_visitados) == 1:
        i = hospitales_visitados[0]
        return [i], delta[0, i + 1] + delta[i + 1, 0]

    nodos = [0] + [i + 1 for i in hospitales_visitados]
    n_h = len(hospitales_visitados)
    modelo = gp.Model("heuristica_a_ruteo")
    modelo.Params.OutputFlag = 0
    x = modelo.addVars([(m, n) for m in nodos for n in nodos if m != n], vtype=GRB.BINARY, name="x")
    rho = modelo.addVars(nodos[1:], lb=1, ub=n_h, name="rho")
    for m in nodos:
        modelo.addConstr(x.sum(m, "*") == 1, name=f"sale[{m}]")
        modelo.addConstr(x.sum("*", m) == 1, name=f"entra[{m}]")
    for m in nodos[1:]:
        for n in nodos[1:]:
            if m != n:
                modelo.addConstr(rho[m] - rho[n] + n_h * x[m, n] <= n_h - 1, name=f"mtz[{m},{n}]")
    modelo.setObjective(gp.quicksum(delta[m, n] * x[m, n] for (m, n) in x), GRB.MINIMIZE)
    modelo.optimize()
    if modelo.Status != GRB.OPTIMAL:
        raise RuntimeError(f"El ruteo no llegó al óptimo (estado {modelo.Status}).")

    ruta, actual = [], 0
    while True:
        siguiente = next(n for n in nodos if n != actual and x[actual, n].X > 0.5)
        if siguiente == 0:
            break
        ruta.append(siguiente - 1)
        actual = siguiente
    return ruta, modelo.ObjVal


def resolver_modelo_calendario(escenarios, estado_inicial, insumos):
    """Heurística A completa (paso 1 + paso 2). Misma firma y resultado que
    `modelo_comun.resolver_modelo_calendario`.

    Devuelve: SolucionCalendario. En info_resolucion queda la información del
        paso 1 (Gurobi) y el valor objetivo total (inventario + transporte).
    """
    visitas, info = resolver_inventario(escenarios, estado_inicial, insumos)
    delta = modelo_comun.matriz_distancias(insumos)

    rutas, km = {}, 0.0
    for t, k in sorted({(t, k) for (t, _, k) in visitas}):
        hospitales_visitados = sorted(i for (dia, i, kk) in visitas if dia == t and kk == k)
        ruta, largo = ruta_mas_corta(hospitales_visitados, delta)
        rutas[t, k] = ruta
        km += largo

    info = dict(info)
    info["valor_objetivo_inventario"] = info["valor_objetivo"]
    info["valor_objetivo"] = info["valor_objetivo"] + insumos.parametros["costo_km"] * km
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)
