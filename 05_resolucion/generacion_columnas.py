"""Modelo de calendario por generación de columnas (EN DESARROLLO).

Diseño: `feedback_presentacion/plan_generacion_columnas.md`. Reformula el modelo
de calendario del modelo común: en vez de armar las rutas arco por arco, el
maestro elige entre RUTAS completas, cada una con su costo de transporte ya
calculado. No modifica ningún otro archivo del proyecto.

Paso 0 (este archivo, por ahora): costo y orden óptimo de cada ruta, con
Held-Karp.

Una ruta es un grupo de hospitales que visita un vehículo en un día, saliendo
del banco y volviendo a él. Su costo es fijo (supuesto G1 del plan): depende
solo de las distancias, no del día ni de cuánto se lleve.
"""

import itertools

import numpy as np

from codigo import modelo_comun


def held_karp(delta, hospitales):
    """Orden óptimo y largo de la ruta más corta para TODOS los grupos de
    `hospitales`, con programación dinámica (Held-Karp).

    La ruta sale del banco (posición 0 de `delta`), visita todos los hospitales
    del grupo una vez y vuelve al banco. Admite distancias asimétricas.

    Idea: mejor[S][j] = largo del camino más corto que sale del banco, visita
    exactamente los hospitales del conjunto S y termina en j (j ∈ S). Se
    construye para conjuntos cada vez más grandes:
        mejor[S][j] = min_{h ∈ S − {j}} mejor[S − {j}][h] + δ[h][j],
    y la ruta del grupo S cuesta min_j mejor[S][j] + δ[j][banco].
    Hace del orden de 2^n · n² operaciones (n = número de hospitales): con 10
    hospitales son ~10^5, en vez de probar los 10! ≈ 3,6 millones de órdenes.

    Recibe:
        delta: matriz de distancias (n_h + 1)², banco en la posición 0 y el
            hospital i en la posición i + 1 (`modelo_comun.matriz_distancias`).
        hospitales: lista de hospitales i (posiciones de Insumos.hospitales).
    Devuelve: diccionario {grupo (tupla ordenada de hospitales i): (orden de
        visita (lista de hospitales i), km)}, con los 2^n − 1 grupos no vacíos.
    """
    n = len(hospitales)
    nodo = [i + 1 for i in hospitales]  # posición de cada hospital en delta
    infinito = float("inf")
    # mejor[mascara][j] y de_donde[mascara][j]; la máscara indica qué hospitales (bits) se visitaron.
    mejor = np.full((1 << n, n), infinito)
    de_donde = np.full((1 << n, n), -1, dtype=int)
    for j in range(n):
        mejor[1 << j][j] = delta[0, nodo[j]]
    for mascara in range(1, 1 << n):
        for j in range(n):
            if not (mascara >> j) & 1 or mejor[mascara][j] == infinito:
                continue
            for siguiente in range(n):
                if (mascara >> siguiente) & 1:
                    continue
                nueva = mascara | (1 << siguiente)
                largo = mejor[mascara][j] + delta[nodo[j], nodo[siguiente]]
                if largo < mejor[nueva][siguiente]:
                    mejor[nueva][siguiente] = largo
                    de_donde[nueva][siguiente] = j

    rutas = {}
    for mascara in range(1, 1 << n):
        cierre = [mejor[mascara][j] + delta[nodo[j], 0] if (mascara >> j) & 1 else infinito for j in range(n)]
        ultimo = int(np.argmin(cierre))
        km = cierre[ultimo]
        # Reconstruir el orden hacia atrás.
        orden, actual, resto = [], ultimo, mascara
        while actual != -1:
            orden.append(hospitales[actual])
            anterior = de_donde[resto][actual]
            resto &= ~(1 << actual)
            actual = anterior
        orden.reverse()
        grupo = tuple(sorted(hospitales[j] for j in range(n) if (mascara >> j) & 1))
        rutas[grupo] = (orden, float(km))
    return rutas


def costos_rutas(insumos, hospitales=None):
    """Tabla de rutas de la instancia: para cada grupo de hospitales, su orden
    óptimo, sus km y su costo de transporte c_r = costo_km × km.

    Se calcula una sola vez, antes y por separado del horizonte rodante.

    Recibe: insumos; hospitales (opcional): lista de hospitales i a considerar
        (por defecto, todos los de la instancia).
    Devuelve: diccionario {grupo: {"orden": [...], "km": float, "costo": float}}.
    """
    if hospitales is None:
        hospitales = list(range(len(insumos.hospitales)))
    alfa = insumos.parametros["costo_km"]
    delta = modelo_comun.matriz_distancias(insumos)
    return {grupo: {"orden": orden, "km": km, "costo": alfa * km}
            for grupo, (orden, km) in held_karp(delta, hospitales).items()}


def ruta_por_fuerza_bruta(delta, grupo):
    """Orden óptimo probando todos los órdenes (solo para verificar Held-Karp)."""
    largo = lambda orden: (delta[0, orden[0] + 1] + sum(delta[a + 1, b + 1] for a, b in zip(orden[:-1], orden[1:]))
                           + delta[orden[-1] + 1, 0])
    mejor = min(itertools.permutations(grupo), key=largo)
    return list(mejor), float(largo(mejor))


# ---------------------------------------------------------------------------
# Paso 2: el maestro por rutas
# ---------------------------------------------------------------------------

import dataclasses

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from codigo import configuracion as cfg
from codigo.modelo_comun import SolucionCalendario


def resolver_maestro(escenarios, estado_inicial, insumos, rutas, relajado=False):
    """Modelo de calendario por rutas (plan §5), con las rutas dadas.

    Variables nuevas: lambda[r, t] = cuántos vehículos hacen la ruta r el día t,
    y q[i, r, a, t, s] = lo que la ruta r lleva al hospital i. Todo el
    inventario es EL MISMO del modelo común: se construye con
    `modelo_comun._agregar_inventario`, sin modificarlo, tratando cada ruta
    como si fuera un "vehículo" para efectos de las cantidades.

    Restricciones propias de las rutas:
    - flota: Σ_r lambda[r, t] ≤ número de vehículos, cada día;
    - capacidad de cada ruta: lo que lleva ≤ θ · lambda[r, t], en cada escenario.

    Supone (plan §6): vehículos idénticos (G2), costo de ruta fijo (G1).

    Recibe:
        escenarios, estado_inicial, insumos: como en modelo_comun.
        rutas: lista de grupos (tuplas de hospitales i); su costo y orden salen
            de `costos_rutas`.
        relajado: si es True, resuelve la versión continua (para los precios
            sombra de la generación de columnas).
    Devuelve: (SolucionCalendario o None si relajado, modelo de Gurobi resuelto,
        diccionario de variables lambda).
    """
    capacidades = set(insumos.flota["capacidad"])
    if len(capacidades) != 1:
        raise ValueError("El maestro por rutas supone vehículos idénticos (supuesto G2).")
    theta = capacidades.pop()
    n_vehiculos = len(insumos.flota)
    tabla = costos_rutas(insumos, sorted({i for r in rutas for i in r}))
    dias = range(escenarios.dias)
    rutas = [tuple(sorted(r)) for r in rutas]

    modelo = gp.Model("maestro_rutas")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)
    tipo = GRB.CONTINUOUS if relajado else GRB.INTEGER
    lam = modelo.addVars(range(len(rutas)), dias, lb=0, ub=n_vehiculos, vtype=tipo, name="lambda")

    # Cada ruta hace de "vehículo" para las cantidades. Su tope total por escenario
    # (θ · número de vehículos) nunca es activo: la capacidad real va abajo, con lambda.
    flota_rutas = pd.DataFrame({"vehiculo": range(len(rutas)), "capacidad": theta * n_vehiculos})
    insumos_rutas = dataclasses.replace(insumos, flota=flota_rutas)
    miembros = [set(r) for r in rutas]
    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado_inicial, insumos_rutas,
        # Σ_a q[i, r] ≤ θ_r · lambda: con θ_r = θ · n_vehiculos sería flojo, así que se corrige abajo.
        visita=lambda i, r, t: lam[r, t] if i in miembros[r] else 0, hoy_comun=False,
    )
    modelo.update()

    # Cantidades por ruta (las crea _agregar_inventario con el nombre q[i,r,a,t,s]).
    q = {}
    for v in modelo.getVars():
        if v.VarName.startswith("q["):
            i, r, a, t, s = (int(z) for z in v.VarName[2:-1].split(","))
            q.setdefault((r, t, s), []).append(v)
    # Capacidad de cada ruta: lo que lleva ≤ θ · lambda (en cada escenario).
    for (r, t, s), vs in q.items():
        modelo.addConstr(gp.quicksum(vs) <= theta * lam[r, t], name=f"cap_ruta[{r},{t},{s}]")
    # Flota.
    for t in dias:
        modelo.addConstr(lam.sum("*", t) <= n_vehiculos, name=f"flota[{t}]")

    if relajado:
        for v in modelo.getVars():
            v.VType = GRB.CONTINUOUS

    transporte = gp.quicksum(tabla[rutas[r]]["costo"] * lam[r, t] for r in range(len(rutas)) for t in dias)
    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)
    if relajado:
        # Versión continua: no tiene gap; se resuelve directo (los precios sombra quedan en .Pi).
        modelo.optimize()
        if modelo.Status != GRB.OPTIMAL:
            raise RuntimeError(f"El maestro continuo no llegó al óptimo (estado {modelo.Status}).")
        return None, modelo, lam
    info = modelo_comun._resolver(modelo)

    # Solución en el formato de siempre: cada ruta elegida se asigna a un vehículo (idénticos).
    hoy = estado_inicial.dia
    visitas, rutas_sol, km = set(), {}, 0.0
    vehiculos = list(insumos.flota["vehiculo"])
    for t in dias:
        k = 0
        for r in range(len(rutas)):
            for _copia in range(int(round(lam[r, t].X))):
                vehiculo = vehiculos[k]; k += 1
                rutas_sol[hoy + t, vehiculo] = list(tabla[rutas[r]]["orden"])
                visitas |= {(hoy + t, i, vehiculo) for i in rutas[r]}
                km += tabla[rutas[r]]["km"]
    return SolucionCalendario(visitas=visitas, rutas=rutas_sol, distancia_km=km, info_resolucion=info), modelo, lam


# ---------------------------------------------------------------------------
# Pasos 3 a 5: subproblema, ciclo y solución entera
# ---------------------------------------------------------------------------

import time


def _beneficios_por_unidad(modelo, rutas, n_dias, n_escenarios, edades):
    """Para cada (hospital i, día t, escenario s): cuánto bajaría el costo del
    maestro continuo por cada unidad que una ruta NUEVA le lleve al hospital i,
    según los precios sombra (lo mejor entre las edades a).

    Se obtiene del costo reducido que Gurobi calcula para la variable q de la
    ruta individual {i} (que siempre está en el maestro), descontando los
    precios sombra de las restricciones propias de esa ruta (visita, capacidad
    del "vehículo" y capacidad de la ruta), que una ruta nueva no comparte.
    Lo que queda es el efecto de una unidad en el inventario compartido
    (banco, hospital y capacidad del hospital), igual para cualquier ruta.

    Devuelve: diccionario {(i, t, s): valor ≤ 0} (más negativo = más conveniente).
    """
    individual = {r[0]: indice for indice, r in enumerate(rutas) if len(r) == 1}
    beneficio = {}
    for i, rho in individual.items():
        for t in range(n_dias):
            for s in range(n_escenarios):
                pi_propios = (modelo.getConstrByName(f"visita[{i},{rho},{t},{s}]").Pi
                              + modelo.getConstrByName(f"cap_vehiculo[{rho},{t},{s}]").Pi
                              + modelo.getConstrByName(f"cap_ruta[{rho},{t},{s}]").Pi)
                beneficio[i, t, s] = min(modelo.getVarByName(f"q[{i},{rho},{a},{t},{s}]").RC + pi_propios
                                         for a in edades)
    return beneficio


def subproblema(modelo, rutas, tabla, theta, n_dias, n_escenarios, edades, maximo_por_dia):
    """Busca rutas nuevas con costo reducido negativo (plan §4, paso 3).

    Costo reducido de la ruta S el día t:
        c_S − μ_t + Σ_s θ · min(0, min_{i ∈ S} beneficio[i, t, s]),
    donde μ_t es el precio sombra de la flota del día t: en cada escenario, la
    ruta lleva sus θ unidades al hospital del grupo donde más valen.

    Con pocos hospitales revisa TODOS los grupos de `tabla` (sin límite de
    tamaño). Devuelve: lista de (costo reducido, grupo), las `maximo_por_dia`
    mejores de cada día con costo reducido < 0, sin repetir grupos.
    """
    beneficio = _beneficios_por_unidad(modelo, rutas, n_dias, n_escenarios, edades)
    conocidas = set(rutas)
    nuevas = {}
    for t in range(n_dias):
        mu = modelo.getConstrByName(f"flota[{t}]").Pi
        candidatas = []
        for grupo, datos in tabla.items():
            if grupo in conocidas:
                continue
            valor = datos["costo"] - mu + sum(theta * min(0.0, min(beneficio[i, t, s] for i in grupo))
                                              for s in range(n_escenarios))
            if valor < -1e-6:
                candidatas.append((valor, grupo))
        for valor, grupo in sorted(candidatas)[:maximo_por_dia]:
            if grupo not in nuevas or valor < nuevas[grupo]:
                nuevas[grupo] = valor
    return sorted((v, g) for g, v in nuevas.items())


def generar_columnas(escenarios, estado_inicial, insumos, hospitales=None, maximo_por_dia=5,
                     maximo_iteraciones=200, registro=None):
    """Ciclo de generación de columnas sobre el maestro continuo (plan §4, pasos 1 a 4).

    Parte con las rutas de un solo hospital y agrega en cada iteración las
    rutas con costo reducido negativo, hasta que no quede ninguna.

    Devuelve: diccionario con las rutas finales, el valor del maestro continuo
        (cota inferior del problema con todas las rutas), las iteraciones y el
        tiempo, y el historial por iteración.
    """
    if hospitales is None:
        hospitales = list(range(len(insumos.hospitales)))
    tabla = costos_rutas(insumos, hospitales)
    theta = int(insumos.flota["capacidad"].iloc[0])
    edades = range(insumos.parametros["shelf_life"])
    rutas = [(i,) for i in hospitales]
    historial, inicio = [], time.time()
    for iteracion in range(1, maximo_iteraciones + 1):
        _, modelo, _ = resolver_maestro(escenarios, estado_inicial, insumos, rutas, relajado=True)
        nuevas = subproblema(modelo, rutas, tabla, theta, escenarios.dias, escenarios.n_escenarios, edades, maximo_por_dia)
        fila = {"iteracion": iteracion, "rutas": len(rutas), "valor_continuo": modelo.ObjVal,
                "nuevas": len(nuevas), "mejor_costo_reducido": nuevas[0][0] if nuevas else 0.0,
                "tiempo_s": time.time() - inicio}
        historial.append(fila)
        if registro:
            registro(fila)
        if not nuevas:
            break
        rutas += [g for _, g in nuevas]
    return {"rutas": rutas, "valor_continuo": historial[-1]["valor_continuo"], "iteraciones": len(historial),
            "convergio": not nuevas, "tiempo_s": time.time() - inicio, "historial": historial}


def resolver_modelo_calendario(escenarios, estado_inicial, insumos, maximo_por_dia=5):
    """Modelo de calendario por generación de columnas, con la misma firma que
    `modelo_comun.resolver_modelo_calendario` (plan §4, pasos 1 a 6).

    1. Generación de columnas sobre el maestro continuo (rutas útiles + cota).
    2. Maestro ENTERO con las rutas generadas (price-and-branch, paso 5).
    Devuelve: SolucionCalendario; en info_resolucion quedan además las rutas
        generadas, la cota del maestro continuo y el tiempo de cada parte.
    """
    cg = generar_columnas(escenarios, estado_inicial, insumos, maximo_por_dia=maximo_por_dia)
    sol, _, _ = resolver_maestro(escenarios, estado_inicial, insumos, cg["rutas"])
    sol.info_resolucion.update({
        "rutas_generadas": len(cg["rutas"]), "iteraciones_cg": cg["iteraciones"], "convergio_cg": cg["convergio"],
        "cota_continua_cg": cg["valor_continuo"], "tiempo_cg_s": cg["tiempo_s"],
    })
    return sol
