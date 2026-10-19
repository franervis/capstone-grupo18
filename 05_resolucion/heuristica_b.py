"""Heurística B: partición geográfica por vehículo.

A cada vehículo se le asigna un subconjunto fijo de hospitales (su zona) y
sólo puede visitar esos. Sólo cambia el modelo de calendario; el modelo de
cantidades es el del modelo común, que ya respeta el calendario (y por lo
tanto las zonas).

El modelo de calendario es el del modelo común con una sola diferencia: las
variables de ruteo (y, x, rho) sólo existen dentro de la zona de cada
vehículo. Los arcos x bajan de 17.710 a menos de 1.000 y desaparece la
simetría entre vehículos. A diferencia de la heurística A, el costo de
transporte sigue dentro del modelo.

- El bloque de ruteo de `modelo_comun.resolver_modelo_calendario` se copia
  aquí, restringido a las zonas.
- La dinámica de inventario, los costos y la resolución se IMPORTAN del modelo
  común (que no se modifica), para que la comparación sea justa.

Variantes:
- B1, zonas rígidas: `resolver_modelo_calendario`. Zonas compactas
  (`particion_compacta`): un MIP chico minimiza la distancia entre los
  hospitales de cada zona, con la demanda de las zonas dentro de ±10% del
  promedio.
- B2, zonas con frontera: `resolver_modelo_calendario_fronteras`. Cada zona
  incorpora los 2 hospitales de otras zonas más cercanos a ella, que quedan
  compartidos entre dos vehículos.
- Zonas a mano: `crear_resolver_calendario(zonas)`.
- Partición angular original (obsoleta, con un defecto; ver
  `particion_angular`): `resolver_modelo_calendario_angular`. Sólo para
  reproducir los resultados publicados el 2026-10-05.

Ojo: con la flota al ~91% de su capacidad media, en zonas rígidas la demanda
de una zona supera la capacidad de su vehículo en un 20–40% de los días
(ver `describir_zonas`). Es lo esperable de B1 y lo que B2 intenta mitigar.

Uso (misma firma que el modelo común):

    from codigo import heuristica_b
    simular(..., resolver_calendario=heuristica_b.resolver_modelo_calendario)
"""

import itertools

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from codigo import cargador_datos
from codigo import configuracion as cfg
from codigo import modelo_comun
from codigo.modelo_comun import SolucionCalendario


# ---------------------------------------------------------------------------
# Zonas
# ---------------------------------------------------------------------------


def _demanda_media(insumos):
    """Demanda media diaria de cada hospital i (orden de Insumos.hospitales)."""
    return np.array([
        insumos.distribuciones_demanda[nombre].distribucion.mean()
        for nombre in insumos.hospitales["nombre"]
    ])


def _orden_angular(insumos):
    """Hospitales i ordenados por su ángulo alrededor del banco."""
    nodos = cargador_datos.cargar_hospitales()
    banco = nodos.loc[nodos["nombre"] == insumos.nombre_banco].iloc[0]
    angulos = np.arctan2(
        insumos.hospitales["lat"].to_numpy() - banco["lat"],
        insumos.hospitales["lon"].to_numpy() - banco["lon"],
    )
    return [int(i) for i in np.argsort(angulos)]


def particion_angular(insumos):
    """Partición ORIGINAL (obsoleta): sectores angulares contiguos alrededor
    del banco, uno por vehículo, equilibrados por demanda media. Se conserva
    sólo para reproducir los resultados de la B en las 60 semanas publicados
    el 2026-10-05; usar `particion_compacta`.

    Defecto: corta la lista ordenada por ángulo, no el ángulo mismo, e ignora
    la distancia. Entre −12° y 145° no hay hospitales, así que una "zona
    contigua" puede abarcar casi 180° (San José de Maipo quedó con San José),
    y el ángulo de los hospitales pegados al banco es casi arbitrario.

    Busca exhaustivamente el punto de partida y los cortes que minimizan la
    demanda media de la zona más cargada.

    Devuelve: lista de zonas en orden angular; cada zona es una lista de
        hospitales i. La zona j es del j-ésimo vehículo de la flota.
    """
    n_zonas = len(insumos.flota)
    orden = _orden_angular(insumos)
    demanda = _demanda_media(insumos)
    n_h = len(orden)
    mejor = None
    for inicio in range(n_h):
        rotado = orden[inicio:] + orden[:inicio]
        acumulada = np.concatenate([[0.0], np.cumsum(demanda[rotado])])
        for cortes in itertools.combinations(range(1, n_h), n_zonas - 1):
            limites = (0, *cortes, n_h)
            carga = max(acumulada[b] - acumulada[a] for a, b in zip(limites, limites[1:]))
            if mejor is None or carga < mejor[0]:
                mejor = (carga, [rotado[a:b] for a, b in zip(limites, limites[1:])])
    return mejor[1]


# Tolerancia de la demanda de cada zona respecto al promedio entre zonas.
# Es un parámetro de diseño de la heurística, no un dato del problema.
# Valor: 0.10. Con 5 zonas y ~651 unidades/día, cada zona queda entre ~117 y
# ~143 unidades/día; el tope coincide con la capacidad de un vehículo (143).
# Si se cambia: menor = zonas más parejas en demanda pero menos compactas;
# mayor = zonas más compactas pero alguna puede superar la capacidad de su vehículo.
TOLERANCIA_DEMANDA_ZONAS = 0.10

# Límite de tiempo (segundos) del MIP que arma las zonas. En la instancia
# completa llega al óptimo en mucho menos.
TIEMPO_LIMITE_ZONAS_S = 300

_zonas_calculadas = {}


def _distancia_simetrica(insumos):
    """δ entre hospitales (n_h × n_h), promediando ida y vuelta (la matriz
    tiene un par asimétrico)."""
    delta = modelo_comun.matriz_distancias(insumos)[1:, 1:]
    return (delta + delta.T) / 2


def particion_compacta(insumos, tolerancia=TOLERANCIA_DEMANDA_ZONAS):
    """Zonas compactas con demanda parecida, una por vehículo.

    MIP: asigna cada hospital a una zona minimizando la suma de las
    distancias viales entre todos los pares de hospitales que quedan en la
    misma zona, con la demanda media de cada zona dentro de ±`tolerancia`
    del promedio entre zonas.

        x_iz ∈ {0,1}: el hospital i está en la zona z;  s_ij ≥ 0: i y j comparten zona
        min  Σ_{i<j} δ_ij · s_ij
        s.a. Σ_z x_iz = 1                        para todo i
             s_ij ≥ x_iz + x_jz − 1               para todo i < j y z
             (1 − tol)·d̄ ≤ Σ_i d_i x_iz ≤ (1 + tol)·d̄   para todo z, con d̄ = Σ_i d_i / n_zonas
             x_iz = 0 si z > i                    (rompe la simetría entre zonas)

    Se resuelve una sola vez por combinación de datos y tolerancia; las
    siguientes llamadas devuelven el resultado guardado.

    Devuelve: lista de zonas (listas de hospitales i); la zona j es del
        j-ésimo vehículo de la flota.
    """
    demanda = _demanda_media(insumos)
    n_zonas, n_h = len(insumos.flota), len(demanda)
    clave = (tuple(insumos.hospitales["nombre"]), tuple(np.round(demanda, 6)), n_zonas, tolerancia)
    if clave in _zonas_calculadas:
        return [list(zona) for zona in _zonas_calculadas[clave]]

    delta = _distancia_simetrica(insumos)
    promedio = demanda.sum() / n_zonas
    hospitales, zonas = range(n_h), range(n_zonas)
    pares = [(i, j) for i in hospitales for j in hospitales if i < j]

    modelo = gp.Model("heuristica_b_zonas")
    modelo.Params.OutputFlag = 1 if cfg.MOSTRAR_LOG_GUROBI else 0
    modelo.Params.TimeLimit = TIEMPO_LIMITE_ZONAS_S
    modelo.Params.MIPGap = 0
    x = modelo.addVars(hospitales, zonas, vtype=GRB.BINARY, name="x")
    s = modelo.addVars(pares, lb=0, name="s")
    for i in hospitales:
        modelo.addConstr(x.sum(i, "*") == 1, name=f"asigna[{i}]")
        for z in zonas:
            if z > i:
                x[i, z].UB = 0
    for (i, j) in pares:
        for z in zonas:
            modelo.addConstr(s[i, j] >= x[i, z] + x[j, z] - 1, name=f"juntos[{i},{j},{z}]")
    for z in zonas:
        carga = gp.quicksum(demanda[i] * x[i, z] for i in hospitales)
        modelo.addConstr(carga >= (1 - tolerancia) * promedio, name=f"carga_min[{z}]")
        modelo.addConstr(carga <= (1 + tolerancia) * promedio, name=f"carga_max[{z}]")
    modelo.setObjective(gp.quicksum(delta[i, j] * s[i, j] for (i, j) in pares), GRB.MINIMIZE)
    modelo.optimize()
    if modelo.SolCount == 0:
        raise RuntimeError(
            f"No hay zonas con demanda dentro de ±{tolerancia:.0%} del promedio "
            f"(estado {modelo.Status}); aumentar la tolerancia."
        )
    if modelo.Status != GRB.OPTIMAL:
        print(f"Aviso: las zonas no llegaron al óptimo (gap {modelo.MIPGap:.1%}).")

    resultado = [[i for i in hospitales if x[i, z].X > 0.5] for z in zonas]
    _zonas_calculadas[clave] = resultado
    return [list(zona) for zona in resultado]


def ampliar_con_fronteras(zonas, insumos, n_extra=2):
    """B2: cada zona incorpora los `n_extra` hospitales de otras zonas más
    cercanos a ella (distancia al hospital más cercano de la zona); esos
    quedan compartidos entre dos vehículos.

    Recibe: zonas (como las de `particion_compacta`), insumos, n_extra.
    Devuelve: lista de zonas del mismo largo, con solapamiento.
    """
    if n_extra < 1:
        raise ValueError("n_extra debe ser al menos 1 (con 0 las zonas no cambian).")
    delta = _distancia_simetrica(insumos)
    ampliadas = []
    for zona in zonas:
        afuera = [j for j in range(len(delta)) if j not in zona]
        cercania = sorted(afuera, key=lambda j: min(delta[i, j] for i in zona))
        ampliadas.append(list(zona) + cercania[:n_extra])
    return ampliadas


def _zonas_por_vehiculo(zonas, insumos):
    """{k: [hospitales i]}. Valida que haya una zona por vehículo y que todo
    hospital esté en alguna zona."""
    vehiculos = list(insumos.flota["vehiculo"])
    if len(zonas) != len(vehiculos):
        raise ValueError(f"Hay {len(zonas)} zonas y {len(vehiculos)} vehículos; deben coincidir.")
    sin_zona = set(range(len(insumos.hospitales))) - {i for zona in zonas for i in zona}
    if sin_zona:
        nombres = [insumos.hospitales["nombre"][i] for i in sorted(sin_zona)]
        raise ValueError(f"Hospitales sin zona (nunca serían visitados): {nombres}")
    return {k: list(zona) for k, zona in zip(vehiculos, zonas)}


def describir_zonas(zonas, insumos):
    """Diagnóstico por vehículo: hospitales, demanda media de la zona y uso
    medio de la capacidad θ. Un hospital compartido cuenta entero en cada
    zona donde aparece."""
    demanda = _demanda_media(insumos)
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    filas = []
    for k, zona in _zonas_por_vehiculo(zonas, insumos).items():
        carga = demanda[zona].sum()
        filas.append({
            "vehiculo": k,
            "hospitales": ", ".join(insumos.hospitales["nombre"][i] for i in zona),
            "n_hospitales": len(zona),
            "demanda_media": round(carga, 1),
            "theta": theta[k],
            "uso_medio": round(carga / theta[k], 2),
        })
    return pd.DataFrame(filas).set_index("vehiculo")


# ---------------------------------------------------------------------------
# Modelo de calendario con zonas
# ---------------------------------------------------------------------------


def resolver_calendario_zonas(escenarios, estado_inicial, insumos, zonas):
    """Modelo de calendario del modelo común en que cada vehículo
    k sólo visita los hospitales de su zona.

    Recibe: lo mismo que `modelo_comun.resolver_modelo_calendario`, más
        zonas (una lista de hospitales i por vehículo, en el orden de la flota).
    Devuelve: SolucionCalendario; en info_resolucion queda además "zonas".
    """
    zona = _zonas_por_vehiculo(zonas, insumos)
    vehiculos, dias = list(zona), range(escenarios.dias)
    alfa = insumos.parametros["costo_km"]
    delta = modelo_comun.matriz_distancias(insumos)
    # Nodos del ruteo: 0 = banco b, i + 1 = hospital i (sólo dentro de x).
    nodo = lambda i: i + 1
    nodos = {k: [0] + [nodo(i) for i in zona[k]] for k in vehiculos}
    pares = [(i, k) for k in vehiculos for i in zona[k]]

    modelo = gp.Model("heuristica_b_calendario")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)

    y = modelo.addVars([(i, k, t) for (i, k) in pares for t in dias], vtype=GRB.BINARY, name="y")
    w = modelo.addVars(vehiculos, dias, vtype=GRB.BINARY, name="w")
    x = modelo.addVars([(m, n, k, t) for k in vehiculos for m in nodos[k] for n in nodos[k] if m != n for t in dias],
                       vtype=GRB.BINARY, name="x")
    rho = modelo.addVars([(i, k, t) for (i, k) in pares for t in dias], lb=1,
                         ub={(i, k, t): len(zona[k]) for (i, k) in pares for t in dias}, name="rho")

    for k in vehiculos:
        n_k = len(zona[k])
        for t in dias:
            for i in zona[k]:
                modelo.addConstr(gp.quicksum(x[nodo(i), n, k, t] for n in nodos[k] if n != nodo(i)) == y[i, k, t], name=f"sale[{i},{k},{t}]")
                modelo.addConstr(gp.quicksum(x[m, nodo(i), k, t] for m in nodos[k] if m != nodo(i)) == y[i, k, t], name=f"entra[{i},{k},{t}]")
                modelo.addConstr(y[i, k, t] <= w[k, t], name=f"usa[{i},{k},{t}]")
            modelo.addConstr(gp.quicksum(x[0, n, k, t] for n in nodos[k] if n != 0) == w[k, t], name=f"sale_banco[{k},{t}]")
            modelo.addConstr(gp.quicksum(x[m, 0, k, t] for m in nodos[k] if m != 0) == w[k, t], name=f"vuelve_banco[{k},{t}]")
            for i in zona[k]:
                for j in zona[k]:
                    if i != j:
                        modelo.addConstr(
                            rho[i, k, t] - rho[j, k, t] + n_k * x[nodo(i), nodo(j), k, t] <= n_k - 1,
                            name=f"mtz[{i},{j},{k},{t}]",
                        )

    transporte = alfa * gp.quicksum(delta[m, n] * x[m, n, k, t] for (m, n, k, t) in x)
    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado_inicial, insumos,
        visita=lambda i, k, t: y[i, k, t] if (i, k, t) in y else 0, hoy_comun=False,
    )
    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)
    info = modelo_comun._resolver(modelo)

    hoy = estado_inicial.dia
    visitas = {(hoy + t, i, k) for (i, k, t) in y if y[i, k, t].X > 0.5}
    rutas, km = {}, 0.0
    for k in vehiculos:
        for t in dias:
            if w[k, t].X < 0.5:
                continue
            ruta, actual = [], 0
            while True:
                siguiente = next(n for n in nodos[k] if n != actual and x[actual, n, k, t].X > 0.5)
                km += delta[actual, siguiente]
                if siguiente == 0:
                    break
                ruta.append(siguiente - 1)  # de nodo a hospital i
                actual = siguiente
            rutas[hoy + t, k] = ruta
    info["zonas"] = [zona[k] for k in vehiculos]
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)


def crear_resolver_calendario(zonas):
    """Función con la firma de `modelo_comun.resolver_modelo_calendario` para
    unas zonas dadas, lista para `simular(..., resolver_calendario=...)`."""

    def resolver(escenarios, estado_inicial, insumos):
        return resolver_calendario_zonas(escenarios, estado_inicial, insumos, zonas)

    return resolver


def resolver_modelo_calendario(escenarios, estado_inicial, insumos):
    """B1 (zonas rígidas y compactas). Misma firma y resultado que el modelo común."""
    return resolver_calendario_zonas(escenarios, estado_inicial, insumos, particion_compacta(insumos))


def resolver_modelo_calendario_fronteras(escenarios, estado_inicial, insumos):
    """B2 (zonas compactas que comparten sus hospitales de borde). Misma firma
    y resultado que el modelo común."""
    zonas = ampliar_con_fronteras(particion_compacta(insumos), insumos)
    return resolver_calendario_zonas(escenarios, estado_inicial, insumos, zonas)


def resolver_modelo_calendario_angular(escenarios, estado_inicial, insumos):
    """B1 con la partición angular original (obsoleta). Sólo para reproducir
    los resultados publicados el 2026-10-05."""
    return resolver_calendario_zonas(escenarios, estado_inicial, insumos, particion_angular(insumos))
