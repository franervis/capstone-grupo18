"""Modelo de calendario del modelo común con mejoras exactas opcionales (EXPERIMENTAL).

Responde al comentario del profesor (`feedback_presentacion/comentarios_presentacion.md`,
punto 4): antes de usar heurísticas, intentar acelerar el modelo sin cambiar el
problema. NO reemplaza a `codigo/modelo_comun.py`: construye el mismo modelo de
calendario (mismas variables, mismo inventario importado de `modelo_comun`) y le
agrega, según las opciones, mejoras que no cambian el óptimo:

- simetria = "w":        w[0,t] ≥ w[1,t] ≥ ... (los vehículos salen en orden).
- simetria = "visitas":  Σ_i y[i,0,t] ≥ Σ_i y[i,1,t] ≥ ... (los vehículos se ordenan
                         por número de visitas; también ordena w).
  Válidas porque los 5 vehículos son idénticos (capacidad 143): toda solución se
  puede renumerar para cumplir el orden sin cambiar su costo.
- desigualdades = True:  cortes de ciclos de 2 (x[i,j] + x[j,i] ≤ y[i]) y despacho
                         acotado por min(θ_k, C_i) en vez de θ_k.
- subtours = "mtz" (original) o "lazy": elimina MTZ (rho) y agrega cortes de
                         eliminación de subtours solo cuando una solución entera
                         tiene un ciclo que no pasa por el banco (lazy constraints).
- parametros_gurobi:     dict de parámetros extra (Symmetry, Cuts, MIPFocus, ...).

Para el análisis por partes (estas SÍ cambian el problema, solo para medir):
- cantidades_continuas = True: q y u continuas.
"""

import gurobipy as gp
from gurobipy import GRB

from codigo import configuracion as cfg
from codigo import modelo_comun
from codigo.modelo_comun import SolucionCalendario


def topes_utiles(escenarios, insumos):
    """Opción a (`resultados_generacion_columnas.md` §6): para cada hospital i,
    día t y escenario s, lo máximo que le puede servir recibir ese día:

        U[i,t,s] = mín(θ, C_i, demanda de i en los días t .. t+L−1 del escenario s
                                (sin pasar del domingo) + μ_i si la ventana llega al domingo)

    Una unidad que llega el día t vence, a más tardar, al cierre del día t+L−1,
    así que más que eso no se alcanza a consumir (salvo lo que quede para el
    mínimo terminal μ_i del domingo). Son números conocidos: la demanda de cada
    escenario es un dato del modelo.
    """
    from codigo.modelo_comun import _demanda_por_hospital, minimos_terminales
    L = insumos.parametros["shelf_life"]
    theta = int(insumos.flota["capacidad"].max())
    C_i = insumos.hospitales["capacidad_maxima"].to_numpy()
    _, mu_i = minimos_terminales(insumos)
    D = _demanda_por_hospital(escenarios, insumos)  # (S, T, n_h)
    S, T, n_h = D.shape
    U = {}
    for s in range(S):
        for t in range(T):
            fin = min(t + L - 1, T - 1)
            for i in range(n_h):
                util = D[s, t:fin + 1, i].sum() + (mu_i[i] if fin == T - 1 else 0.0)
                U[i, t, s] = float(min(theta, C_i[i], util))
    return U


def resolver(escenarios, estado_inicial, insumos, simetria=None, desigualdades=False, subtours="mtz",
             parametros_gurobi=None, cantidades_continuas=False, inicio=None, topes_de_lo_util=False):
    n_h = len(insumos.hospitales)
    hospitales, vehiculos, dias = range(n_h), list(insumos.flota["vehiculo"]), range(escenarios.dias)
    alfa = insumos.parametros["costo_km"]
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    C_i = insumos.hospitales["capacidad_maxima"].to_numpy()
    delta = modelo_comun.matriz_distancias(insumos)
    nodo = lambda i: i + 1
    nodos = range(n_h + 1)

    modelo = gp.Model("calendario_mejorado")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)
    for nombre, valor in (parametros_gurobi or {}).items():
        modelo.setParam(nombre, valor)

    y = modelo.addVars(hospitales, vehiculos, dias, vtype=GRB.BINARY, name="y")
    w = modelo.addVars(vehiculos, dias, vtype=GRB.BINARY, name="w")
    x = modelo.addVars([(m, n, k, t) for m in nodos for n in nodos if m != n for k in vehiculos for t in dias],
                       vtype=GRB.BINARY, name="x")

    for k in vehiculos:
        for t in dias:
            for i in hospitales:
                modelo.addConstr(gp.quicksum(x[nodo(i), n, k, t] for n in nodos if n != nodo(i)) == y[i, k, t])
                modelo.addConstr(gp.quicksum(x[m, nodo(i), k, t] for m in nodos if m != nodo(i)) == y[i, k, t])
                modelo.addConstr(y[i, k, t] <= w[k, t])
            modelo.addConstr(gp.quicksum(x[0, n, k, t] for n in nodos if n != 0) == w[k, t])
            modelo.addConstr(gp.quicksum(x[m, 0, k, t] for m in nodos if m != 0) == w[k, t])

    if subtours == "mtz":
        rho = modelo.addVars(hospitales, vehiculos, dias, lb=1, ub=n_h, name="rho")
        for k in vehiculos:
            for t in dias:
                for i in hospitales:
                    for j in hospitales:
                        if i != j:
                            modelo.addConstr(rho[i, k, t] - rho[j, k, t] + n_h * x[nodo(i), nodo(j), k, t] <= n_h - 1)
    elif subtours != "lazy":
        raise ValueError(subtours)

    # Ruptura de simetría (los vehículos son idénticos).
    if simetria is not None:
        if len(set(theta.values())) != 1:
            raise ValueError("La ruptura de simetría solo es válida si todos los vehículos son iguales.")
        for a, b in zip(vehiculos[:-1], vehiculos[1:]):
            for t in dias:
                modelo.addConstr(w[a, t] >= w[b, t])
                if simetria == "visitas":
                    modelo.addConstr(y.sum("*", a, t) >= y.sum("*", b, t))

    # Desigualdades válidas.
    if desigualdades:
        for k in vehiculos:
            for t in dias:
                for i in hospitales:
                    for j in hospitales:
                        if i < j:
                            modelo.addConstr(x[nodo(i), nodo(j), k, t] + x[nodo(j), nodo(i), k, t] <= y[i, k, t])
                            modelo.addConstr(x[nodo(i), nodo(j), k, t] + x[nodo(j), nodo(i), k, t] <= y[j, k, t])

    transporte = alfa * gp.quicksum(delta[m, n] * x[m, n, k, t] for (m, n, k, t) in x)
    cota_despacho = (lambda i, k: min(theta[k], int(C_i[i]))) if desigualdades else None
    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado_inicial, insumos,
        visita=lambda i, k, t: y[i, k, t], hoy_comun=False,
    )
    if desigualdades:
        # Despacho a un hospital acotado por min(θ_k, C_i): nunca puede llegar más que su capacidad.
        modelo.update()
        q = {}
        for v in modelo.getVars():
            if v.VarName.startswith("q["):
                i, k, a, t, s = (int(z) for z in v.VarName[2:-1].split(","))
                q.setdefault((i, k, t, s), []).append(v)
        for (i, k, t, s), vs in q.items():
            modelo.addConstr(gp.quicksum(vs) <= cota_despacho(i, k) * y[i, k, t])

    if topes_de_lo_util:
        # Opción a: lo que se lleva a i el día t (por vehículo) ≤ U[i,t,s] · y[i,k,t].
        U = topes_utiles(escenarios, insumos)
        modelo.update()
        qv = {}
        for v in modelo.getVars():
            if v.VarName.startswith("q["):
                i, k, a, t, s = (int(z) for z in v.VarName[2:-1].split(","))
                qv.setdefault((i, k, t, s), []).append(v)
        for (i, k, t, s), vs in qv.items():
            modelo.addConstr(gp.quicksum(vs) <= U[i, t, s] * y[i, k, t])

    if cantidades_continuas:
        modelo.update()
        for v in modelo.getVars():
            if v.VarName.split("[")[0] in ("q", "u", "r", "r_hoy", "q_hoy"):
                v.VType = GRB.CONTINUOUS

    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)

    if inicio is not None:
        inicio(modelo, y, w, x)

    if subtours == "lazy":
        modelo.Params.LazyConstraints = 1
        modelo._x, modelo._pares = x, [(k, t) for k in vehiculos for t in dias]

        def cortar(m, donde):
            if donde != GRB.Callback.MIPSOL:
                return
            valores = m.cbGetSolution(m._x)
            for (k, t) in m._pares:
                sig = {a: b for (a, b, kk, tt), v in valores.items() if kk == k and tt == t and v > 0.5}
                vistos = {0}
                actual = 0
                while actual in sig and sig[actual] not in vistos:  # recorrer la ruta desde el banco
                    actual = sig[actual]; vistos.add(actual)
                for inicio_ciclo in list(sig):
                    if inicio_ciclo in vistos:
                        continue
                    ciclo, actual = [inicio_ciclo], sig[inicio_ciclo]
                    while actual != inicio_ciclo and actual in sig and actual not in ciclo:
                        ciclo.append(actual); actual = sig[actual]
                    vistos |= set(ciclo)
                    if len(ciclo) >= 2:
                        m.cbLazy(gp.quicksum(m._x[a, b, k, t] for a in ciclo for b in ciclo if a != b) <= len(ciclo) - 1)

        modelo.optimize(cortar)
        if modelo.SolCount == 0:
            raise RuntimeError(f"Sin solución factible (estado {modelo.Status}).")
        info = {"estado": modelo.Status, "optimo": modelo.Status == GRB.OPTIMAL, "valor_objetivo": modelo.ObjVal,
                "cota": modelo.ObjBound, "gap": modelo.MIPGap, "tiempo_s": modelo.Runtime, "trabajo": modelo.Work}
    else:
        info = modelo_comun._resolver(modelo)

    hoy = estado_inicial.dia
    visitas = {(hoy + t, i, k) for i in hospitales for k in vehiculos for t in dias if y[i, k, t].X > 0.5}
    rutas, km = {}, 0.0
    for k in vehiculos:
        for t in dias:
            if w[k, t].X < 0.5:
                continue
            ruta, actual = [], 0
            while True:
                siguiente = next(n for n in nodos if n != actual and x[actual, n, k, t].X > 0.5)
                km += delta[actual, siguiente]
                if siguiente == 0:
                    break
                ruta.append(siguiente - 1)
                actual = siguiente
            rutas[hoy + t, k] = ruta
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)
