"""Modelo sin la decisión del orden PERO con zonas fijas por vehículo
(las de la heurística B). Cada vehículo k tiene su zona Z_k y solo se
consideran grupos r ⊆ Z_k; para cada grupo se precalcula con Held-Karp su
mejor orden y sus km.

Idea: a escala completa (22 hospitales), enumerar los 2^22 ≈ 4,19 M grupos
no es viable. Con zonas, solo hay Σ_k (2^{|Z_k|} - 1) grupos. Con 5 zonas de
≈ 4-5 hospitales cada una, son del orden de 160, trivial.

El resto del modelo es idéntico al `sin_orden.resolver_sin_orden`: mantiene
las variables de visita y, w y todo el inventario del modelo común, y usa
una variable z[r, k, t] por grupo-vehículo-día (solo donde r ⊆ Z_k).

Si el objetivo de este modelo da lo mismo que el óptimo del modelo común
(libre), las zonas no pierden nada; si da peor, se mide la pérdida."""

import gurobipy as gp
from gurobipy import GRB

from codigo import configuracion as cfg
from codigo import modelo_comun, generacion_columnas
from codigo.modelo_comun import SolucionCalendario


def resolver_sin_orden_zonas(escenarios, estado_inicial, insumos, zonas):
    """Resuelve el modelo sin orden con zonas fijas por vehículo.

    Recibe:
        escenarios, estado_inicial, insumos: los habituales.
        zonas: lista de listas; `zonas[j]` son los hospitales i del vehículo j
            (como las devuelve `heuristica_b.particion_compacta`).
    """
    n_h = len(insumos.hospitales)
    hospitales = list(range(n_h))
    vehiculos = list(insumos.flota["vehiculo"])
    assert len(zonas) == len(vehiculos), "una zona por vehículo"
    dias = range(escenarios.dias)
    alfa = insumos.parametros["costo_km"]

    # Grupos por vehículo: subconjuntos no vacíos de su zona, con su mejor orden (Held-Karp).
    tabla_por_vehiculo = [generacion_columnas.costos_rutas(insumos, zonas[j]) for j in range(len(vehiculos))]
    grupos_por_vehiculo = [list(t.keys()) for t in tabla_por_vehiculo]

    modelo = gp.Model("sin_orden_zonas")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)

    y = modelo.addVars(hospitales, vehiculos, dias, vtype=GRB.BINARY, name="y")
    w = modelo.addVars(vehiculos, dias, vtype=GRB.BINARY, name="w")
    for k_idx, k in enumerate(vehiculos):
        for t in dias:
            for i in hospitales:
                modelo.addConstr(y[i, k, t] <= w[k, t])
                if i not in zonas[k_idx]:
                    # Fuera de su zona, el vehículo no visita.
                    modelo.addConstr(y[i, k, t] == 0, name=f"fuera_zona[{i},{k},{t}]")

    # z[r, k, t] solo para grupos r de la zona del vehículo k.
    z = {}
    for k_idx, k in enumerate(vehiculos):
        for r_idx in range(len(grupos_por_vehiculo[k_idx])):
            for t in dias:
                z[r_idx, k, t] = modelo.addVar(vtype=GRB.BINARY, name=f"z[{r_idx},{k},{t}]")
        for t in dias:
            modelo.addConstr(gp.quicksum(z[r_idx, k, t] for r_idx in range(len(grupos_por_vehiculo[k_idx]))) == w[k, t],
                             name=f"un_grupo[{k},{t}]")
            for i in zonas[k_idx]:
                modelo.addConstr(y[i, k, t] == gp.quicksum(z[r_idx, k, t]
                                                           for r_idx in range(len(grupos_por_vehiculo[k_idx]))
                                                           if i in grupos_por_vehiculo[k_idx][r_idx]),
                                 name=f"calza[{i},{k},{t}]")

    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado_inicial, insumos,
        visita=lambda i, k, t: y[i, k, t], hoy_comun=False,
    )
    transporte = alfa * gp.quicksum(
        tabla_por_vehiculo[k_idx][grupos_por_vehiculo[k_idx][r_idx]]["km"] * z[r_idx, k, t]
        for k_idx, k in enumerate(vehiculos)
        for r_idx in range(len(grupos_por_vehiculo[k_idx]))
        for t in dias
    )
    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)
    info = modelo_comun._resolver(modelo)

    hoy = estado_inicial.dia
    visitas, rutas, km = set(), {}, 0.0
    for k_idx, k in enumerate(vehiculos):
        for t in dias:
            if w[k, t].X < 0.5:
                continue
            r_idx = next(r for r in range(len(grupos_por_vehiculo[k_idx])) if z[r, k, t].X > 0.5)
            grupo = grupos_por_vehiculo[k_idx][r_idx]
            visitas |= {(hoy + t, i, k) for i in grupo}
            rutas[hoy + t, k] = list(tabla_por_vehiculo[k_idx][grupo]["orden"])
            km += tabla_por_vehiculo[k_idx][grupo]["km"]
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)
