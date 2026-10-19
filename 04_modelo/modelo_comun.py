"""Modelo común del SIRP: modelo de calendario y modelo de cantidades.

Implementa la formulación de la §10 de METODOLOGIA.md, sin heurísticas ni
simplificaciones. Las heurísticas se construyen sobre este archivo.

NOMENCLATURA: los nombres del código son los mismos símbolos de la
formulación (tabla completa en METODOLOGIA.md §10.1). Cada símbolo es único.

    Índices:     i, j hospital (0..21) · b banco · k vehículo · t día (0 = lunes)
                 a edad (0..L-1) · s escenario
    Parámetros:  delta (δ) distancia · alfa (α) costo km · beta (β) holding banco
                 gamma (γ) holding hospital · pi (π) shortage · omega (ω) wastage
                 theta (θ) capacidad vehículo · C capacidad (C_b = insumos.capacidad_banco) · phi (φ) inventario
                 inicial · mu (μ) mínimo terminal · D demanda · O oferta
                 L vida útil · p probabilidad de escenario
    Variables:   y visita · x arco · w salida del vehículo · rho (ρ) posición MTZ
                 r oferta admitida · q despacho · B / B_cierre inventario del banco
                 V / V_cierre inventario del hospital · u consumo · z faltante
                 e déficit terminal

Ambos modelos comparten la misma dinámica de inventario
(`_agregar_inventario`). Diferencias:

- Modelo de calendario (lunes): y, x, w y rho son decisiones. Todas las
  cantidades dependen del escenario (dos etapas).
- Modelo de cantidades (cada día): y está fijo según el calendario, no hay
  costo de transporte (es constante) y las decisiones de HOY (r y q del día
  en curso) son iguales en todos los escenarios (no anticipatividad).

Dentro de un modelo, t recorre los días que quedan; el día de la semana es
t_semana = día de hoy + t.
"""

import math
from dataclasses import dataclass, field

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from codigo import configuracion as cfg


# ---------------------------------------------------------------------------
# Estructuras que intercambian el modelo y el orquestador
# ---------------------------------------------------------------------------


@dataclass
class EstadoSistema:
    """Inventario observado al inicio de un día, ANTES de admitir oferta y despachar.

    Atributos:
        dia: índice del día dentro de la semana (0 = lunes).
        inventario_banco: arreglo de largo L (shelf_life); la posición a es
            el número de unidades con edad a en el banco.
        inventario_hospitales: arreglo (n_hospitales, L), misma convención;
            la fila i es el hospital i (orden de Insumos.hospitales).
    """

    dia: int
    inventario_banco: np.ndarray
    inventario_hospitales: np.ndarray


@dataclass
class SolucionCalendario:
    """Solución del modelo de calendario, fija durante toda la semana.

    Atributos:
        visitas: conjunto de tuplas (t, i, k): el vehículo k visita al
            hospital i el día t de la semana (0 = lunes).
        rutas: diccionario {(t, k): [i1, i2, ...]} con los hospitales en el
            orden de visita. Toda ruta sale del banco y vuelve a él.
        distancia_km: km totales de la semana.
        info_resolucion: estado, gap y tiempo de Gurobi.
    """

    visitas: set = field(default_factory=set)
    rutas: dict = field(default_factory=dict)
    distancia_km: float = 0.0
    info_resolucion: dict = field(default_factory=dict)


@dataclass
class DecisionesDia:
    """Decisiones que se implementan en el día en curso.

    Atributos:
        oferta_admitida: r del día (unidades de la oferta que entran al banco).
        despachos: diccionario {(i, k, a): unidades}, es decir q del día: el
            vehículo k lleva al hospital i esas unidades de edad a.
        info_resolucion: estado, gap y tiempo de Gurobi.
    """

    oferta_admitida: int = 0
    despachos: dict = field(default_factory=dict)
    info_resolucion: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parámetros derivados de los insumos
# ---------------------------------------------------------------------------


def inventario_inicial(insumos):
    """φ: inventario con que parte la semana (supuesto vigente, PENDIENTES.md).

    El 10% de la capacidad de cada lugar (`inventario_inicial_*_factor`),
    redondeado hacia arriba, todo con edad 0.

    Devuelve: (inventario_banco, inventario_hospitales) con la convención
        de EstadoSistema.
    """
    par = insumos.parametros
    L = par["shelf_life"]
    banco = np.zeros(L, dtype=int)
    banco[0] = math.ceil(par["inventario_inicial_banco_factor"] * insumos.capacidad_banco)
    hospitales = np.zeros((len(insumos.hospitales), L), dtype=int)
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    hospitales[:, 0] = np.ceil(par["inventario_inicial_hospital_factor"] * C).astype(int)
    return banco, hospitales


def minimos_terminales(insumos):
    """μ: mínimo de inventario al cierre del domingo (§7 de METODOLOGIA.md).

    Devuelve: (mu_b, arreglo mu_i por hospital). Pueden ser fraccionarios.
    """
    par = insumos.parametros
    C = insumos.hospitales["capacidad_maxima"].to_numpy(dtype=float)
    return (
        par["inventario_inicial_banco_factor"] * insumos.capacidad_banco,
        par["inventario_inicial_hospital_factor"] * C,
    )


def matriz_distancias(insumos):
    """δ como arreglo (n_h + 1)²: la posición 0 es el banco b y la posición
    i + 1 es el hospital i. Sólo se usa dentro del ruteo."""
    nombres = [insumos.nombre_banco] + list(insumos.hospitales["nombre"])
    return insumos.distancias.loc[nombres, nombres].to_numpy(dtype=float)


def _demanda_por_hospital(escenarios, insumos):
    """D como arreglo (S, T, n_h), con el último eje en el orden de Insumos.hospitales."""
    columnas = [escenarios.hospitales.index(nombre) for nombre in insumos.hospitales["nombre"]]
    return escenarios.demanda[:, :, columnas]


def _configurar(modelo, tiempo_limite):
    modelo.Params.OutputFlag = 1 if cfg.MOSTRAR_LOG_GUROBI else 0
    modelo.Params.TimeLimit = tiempo_limite
    modelo.Params.MIPGap = cfg.GAP_MIP


def _resolver(modelo):
    """Optimiza y devuelve la información de resolución. Falla si no hay
    ninguna solución factible (no debería pasar: el modelo siempre admite
    no despachar nada y pagar shortage)."""
    modelo.optimize()
    if modelo.SolCount == 0:
        raise RuntimeError(f"Gurobi no encontró solución factible (estado {modelo.Status}).")
    return {
        "estado": modelo.Status,
        "optimo": modelo.Status == GRB.OPTIMAL,
        "valor_objetivo": modelo.ObjVal,
        "cota": modelo.ObjBound,
        "gap": modelo.MIPGap,
        "tiempo_s": modelo.Runtime,
        # Unidades de trabajo de Gurobi: miden el esfuerzo de cómputo sin depender del reloj.
        # Si el computador se suspende, tiempo_s sigue corriendo pero trabajo no.
        "trabajo": modelo.Work,
    }


def _es_cero(valor):
    """True si `valor` es la constante 0 (no una variable)."""
    return isinstance(valor, (int, float)) and valor == 0


# ---------------------------------------------------------------------------
# Dinámica de inventario (§10.4, común a ambos modelos)
# ---------------------------------------------------------------------------


def _agregar_inventario(modelo, escenarios, estado, insumos, visita, hoy_comun, inicio_por_escenario=None):
    """Agrega las variables y restricciones de inventario (§10.4) y devuelve
    el costo esperado de inventario (holding + shortage + wastage +
    penalización terminal).

    Recibe:
        visita: función (i, k, t) -> variable binaria y, o constante 0/1.
            Si devuelve 0, no se crean variables q para (i, k, t).
        hoy_comun: si es True, r y q del día t = 0 son iguales en todos los
            escenarios (modelo de cantidades).
        inicio_por_escenario: opcional, (banco (S, L), hospitales (S, n_h, L)):
            inventario con que parte el día t = 0 en cada escenario s. Por
            defecto (None) todos los escenarios parten con el de `estado`.
            Lo usa la heurística C (calendario por bloques).
    Devuelve: (expresión de costo, diccionario con r y q de hoy si hoy_comun).
    """
    par = insumos.parametros
    L = par["shelf_life"]
    beta, gamma = par["costo_holding_banco"], par["costo_holding_hospital"]
    pi, omega = par["costo_shortage"], par["costo_wastage"]
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    C_i = insumos.hospitales["capacidad_maxima"].to_numpy()
    C_b = insumos.capacidad_banco
    mu_b, mu_i = minimos_terminales(insumos)
    O = escenarios.oferta
    D = _demanda_por_hospital(escenarios, insumos)
    n_escenarios, n_dias = escenarios.n_escenarios, escenarios.dias
    p = 1.0 / n_escenarios

    hospitales, vehiculos = range(len(insumos.hospitales)), list(insumos.flota["vehiculo"])
    edades, dias, escenarios_idx = range(L), range(n_dias), range(n_escenarios)
    if inicio_por_escenario is None:
        inicio_banco = np.broadcast_to(estado.inventario_banco, (n_escenarios, L))
        inicio_hospitales = np.broadcast_to(estado.inventario_hospitales, (n_escenarios, len(hospitales), L))
    else:
        inicio_banco, inicio_hospitales = inicio_por_escenario

    # r y q. Con hoy_comun, el día 0 usa una sola variable para todos los escenarios.
    if hoy_comun:
        r_hoy = modelo.addVar(vtype=GRB.INTEGER, lb=0, ub=float(O[0, 0]), name="r_hoy")
        q_hoy = {
            (i, k, a): modelo.addVar(vtype=GRB.INTEGER, lb=0, name=f"q_hoy[{i},{k},{a}]")
            for i in hospitales for k in vehiculos if not _es_cero(visita(i, k, 0)) for a in edades
        }
    r, q = {}, {}
    for s in escenarios_idx:
        for t in dias:
            if hoy_comun and t == 0:
                r[t, s] = r_hoy
            else:
                r[t, s] = modelo.addVar(vtype=GRB.INTEGER, lb=0, ub=float(O[s, t]), name=f"r[{t},{s}]")
            for i in hospitales:
                for k in vehiculos:
                    if _es_cero(visita(i, k, t)):
                        continue
                    for a in edades:
                        if hoy_comun and t == 0:
                            q[i, k, a, t, s] = q_hoy[i, k, a]
                        else:
                            q[i, k, a, t, s] = modelo.addVar(vtype=GRB.INTEGER, lb=0, name=f"q[{i},{k},{a},{t},{s}]")

    def llega(i, a, t, s):
        """Unidades de edad a que llegan al hospital i el día t (suma de q sobre vehículos)."""
        return gp.quicksum(q[i, k, a, t, s] for k in vehiculos if (i, k, a, t, s) in q)

    u = modelo.addVars(hospitales, edades, dias, escenarios_idx, vtype=GRB.INTEGER, lb=0, name="u")
    z = modelo.addVars(hospitales, dias, escenarios_idx, lb=0, name="z")
    B_cierre = modelo.addVars(edades, dias, escenarios_idx, lb=0, name="B_cierre")
    V_cierre = modelo.addVars(hospitales, edades, dias, escenarios_idx, lb=0, name="V_cierre")
    e_b = modelo.addVars(escenarios_idx, lb=0, name="e_b")
    e_i = modelo.addVars(hospitales, escenarios_idx, lb=0, name="e_i")

    for s in escenarios_idx:
        for t in dias:
            # B y V: inventario disponible en el día (tras admitir / recibir).
            B, V = {}, {}
            for a in edades:
                if t == 0:
                    arrastre_b = float(inicio_banco[s, a])
                else:
                    arrastre_b = B_cierre[a - 1, t - 1, s] if a >= 1 else 0
                B[a] = arrastre_b + (r[t, s] if a == 0 else 0)
                for i in hospitales:
                    if t == 0:
                        arrastre_i = float(inicio_hospitales[s, i, a])
                    else:
                        arrastre_i = V_cierre[i, a - 1, t - 1, s] if a >= 1 else 0
                    V[i, a] = arrastre_i + llega(i, a, t, s)

            for a in edades:
                sale = gp.quicksum(q[i, k, a, t, s] for i in hospitales for k in vehiculos if (i, k, a, t, s) in q)
                modelo.addConstr(B_cierre[a, t, s] == B[a] - sale, name=f"cierre_banco[{a},{t},{s}]")
                for i in hospitales:
                    modelo.addConstr(V_cierre[i, a, t, s] == V[i, a] - u[i, a, t, s], name=f"cierre_hosp[{i},{a},{t},{s}]")

            for i in hospitales:
                modelo.addConstr(u.sum(i, "*", t, s) + z[i, t, s] == float(D[s, t, i]), name=f"demanda[{i},{t},{s}]")
                modelo.addConstr(gp.quicksum(V[i, a] for a in edades) <= C_i[i], name=f"cap_hosp[{i},{t},{s}]")
                for k in vehiculos:
                    y = visita(i, k, t)
                    if not _es_cero(y):
                        modelo.addConstr(gp.quicksum(q[i, k, a, t, s] for a in edades) <= theta[k] * y, name=f"visita[{i},{k},{t},{s}]")
            modelo.addConstr(gp.quicksum(B[a] for a in edades) <= C_b, name=f"cap_banco[{t},{s}]")
            for k in vehiculos:
                carga = gp.quicksum(q[i, k, a, t, s] for i in hospitales for a in edades if (i, k, a, t, s) in q)
                modelo.addConstr(carga <= theta[k], name=f"cap_vehiculo[{k},{t},{s}]")

        # Mínimo terminal blando: sólo cuentan las unidades que pasan a la semana siguiente.
        ultimo = n_dias - 1
        modelo.addConstr(e_b[s] >= mu_b - gp.quicksum(B_cierre[a, ultimo, s] for a in edades if a <= L - 2), name=f"terminal_banco[{s}]")
        for i in hospitales:
            modelo.addConstr(e_i[i, s] >= mu_i[i] - gp.quicksum(V_cierre[i, a, ultimo, s] for a in edades if a <= L - 2), name=f"terminal_hosp[{i},{s}]")

    # Costo esperado: cada escenario con probabilidad p = 1 / S.
    costo = gp.LinExpr()
    for s in escenarios_idx:
        for t in dias:
            costo += beta * gp.quicksum(B_cierre[a, t, s] for a in edades if a <= L - 2)
            costo += gamma * gp.quicksum(V_cierre[i, a, t, s] for i in hospitales for a in edades if a <= L - 2)
            costo += pi * z.sum("*", t, s)
            costo += omega * (B_cierre[L - 1, t, s] + V_cierre.sum("*", L - 1, t, s))
        costo += pi * (e_b[s] + e_i.sum("*", s))
    costo = p * costo

    variables_hoy = {"r_hoy": r_hoy, "q_hoy": q_hoy} if hoy_comun else {}
    return costo, variables_hoy


# ---------------------------------------------------------------------------
# Modelo de calendario
# ---------------------------------------------------------------------------


def resolver_modelo_calendario(escenarios, estado_inicial, insumos):
    """Modelo de calendario (§10.2–10.4): decide qué hospitales visita cada
    vehículo cada día y en qué orden, junto con las cantidades por escenario.
    Sólo el calendario se usa después.

    Recibe:
        escenarios: Escenarios para los 7 días; el día 0 (lunes) trae la
            oferta real del lunes (supuesto vigente: ya se conoce).
        estado_inicial: EstadoSistema al inicio del lunes.
        insumos: Insumos.
    Devuelve: SolucionCalendario.
    """
    n_h = len(insumos.hospitales)
    hospitales, vehiculos, dias = range(n_h), list(insumos.flota["vehiculo"]), range(escenarios.dias)
    alfa = insumos.parametros["costo_km"]
    delta = matriz_distancias(insumos)
    # Nodos del ruteo: 0 = banco b, i + 1 = hospital i (sólo dentro de x).
    nodo = lambda i: i + 1
    nodos = range(n_h + 1)

    modelo = gp.Model("calendario")
    _configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)

    y = modelo.addVars(hospitales, vehiculos, dias, vtype=GRB.BINARY, name="y")
    w = modelo.addVars(vehiculos, dias, vtype=GRB.BINARY, name="w")
    x = modelo.addVars([(m, n, k, t) for m in nodos for n in nodos if m != n for k in vehiculos for t in dias],
                       vtype=GRB.BINARY, name="x")
    rho = modelo.addVars(hospitales, vehiculos, dias, lb=1, ub=n_h, name="rho")

    for k in vehiculos:
        for t in dias:
            for i in hospitales:
                modelo.addConstr(gp.quicksum(x[nodo(i), n, k, t] for n in nodos if n != nodo(i)) == y[i, k, t], name=f"sale[{i},{k},{t}]")
                modelo.addConstr(gp.quicksum(x[m, nodo(i), k, t] for m in nodos if m != nodo(i)) == y[i, k, t], name=f"entra[{i},{k},{t}]")
                modelo.addConstr(y[i, k, t] <= w[k, t], name=f"usa[{i},{k},{t}]")
            modelo.addConstr(gp.quicksum(x[0, n, k, t] for n in nodos if n != 0) == w[k, t], name=f"sale_banco[{k},{t}]")
            modelo.addConstr(gp.quicksum(x[m, 0, k, t] for m in nodos if m != 0) == w[k, t], name=f"vuelve_banco[{k},{t}]")
            for i in hospitales:
                for j in hospitales:
                    if i != j:
                        modelo.addConstr(
                            rho[i, k, t] - rho[j, k, t] + n_h * x[nodo(i), nodo(j), k, t] <= n_h - 1,
                            name=f"mtz[{i},{j},{k},{t}]",
                        )

    transporte = alfa * gp.quicksum(delta[m, n] * x[m, n, k, t] for (m, n, k, t) in x)
    costo_inventario, _ = _agregar_inventario(
        modelo, escenarios, estado_inicial, insumos,
        visita=lambda i, k, t: y[i, k, t], hoy_comun=False,
    )
    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)
    info = _resolver(modelo)

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
                ruta.append(siguiente - 1)  # de nodo a hospital i
                actual = siguiente
            rutas[hoy + t, k] = ruta
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)


# ---------------------------------------------------------------------------
# Modelo de cantidades
# ---------------------------------------------------------------------------


def resolver_modelo_cantidades(escenarios, estado, calendario, insumos):
    """Modelo de cantidades (§10.5): con el calendario fijo, decide cuánta
    oferta admitir y cuánto despachar desde hoy hasta el domingo. Sólo se
    devuelven las decisiones de hoy, que son iguales en todos los escenarios.

    Recibe:
        escenarios: Escenarios para los días restantes; el día t = 0 es hoy y
            trae la oferta real de hoy.
        estado: EstadoSistema observado hoy.
        calendario: SolucionCalendario (fija).
        insumos: Insumos.
    Devuelve: DecisionesDia.
    """
    modelo = gp.Model("cantidades")
    _configurar(modelo, cfg.TIEMPO_LIMITE_CANTIDADES_S)
    hoy = estado.dia

    def visita(i, k, t):
        """y fijo: 1 si el calendario dice que k visita a i el día hoy + t."""
        return 1 if (hoy + t, i, k) in calendario.visitas else 0

    costo_inventario, variables_hoy = _agregar_inventario(
        modelo, escenarios, estado, insumos, visita=visita, hoy_comun=True,
    )
    modelo.setObjective(costo_inventario, GRB.MINIMIZE)
    info = _resolver(modelo)

    despachos = {
        clave: int(round(var.X)) for clave, var in variables_hoy["q_hoy"].items() if var.X > 0.5
    }
    return DecisionesDia(
        oferta_admitida=int(round(variables_hoy["r_hoy"].X)),
        despachos=despachos,
        info_resolucion=info,
    )
