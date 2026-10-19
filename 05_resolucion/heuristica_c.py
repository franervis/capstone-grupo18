"""Heurística C: ventana más corta en el modelo de cantidades.

La C se descartó como heurística; el código queda como registro reproducible.

Intento 2: heurística de cantidades (la propuesta original de la C).
- Según la propuesta de la C, el calendario se fija el lunes y lo que se acorta
  es la ventana del modelo de cantidades. El calendario viene de otra
  heurística; se usa el de la B (`heuristica_b.resolver_modelo_calendario`).
- Cantidades (cada día): el modelo de cantidades del modelo común, pero sólo
  sobre hoy y los días siguientes de la ventana (`cfg.VENTANA_CANTIDADES_C`,
  3 días), sin pasar del domingo.
- El costo de la heurística es que el modelo de cada día no ve lo que pasa
  después de su ventana. Por ejemplo, el modelo del lunes no sabe que un
  hospital quizás no se vuelve a visitar hasta el viernes.

Intento 1 (descartado): heurística independiente.
- Además de las cantidades con ventana, resolvía su propio calendario por
  bloques (`resolver_modelo_calendario_bloques`): por defecto lunes a jueves
  y después viernes a domingo (`cfg.BLOQUES_CALENDARIO_C`). Cada bloque es el
  modelo de calendario del modelo común (ruteo completo + inventario) sobre
  sus días. Cada bloque parte, en cada escenario s, del inventario que el
  bloque anterior dejó al cierre en ese mismo escenario. Al final los bloques
  se juntan en un solo calendario, que queda fijo toda la semana.
- Se descartó porque los bloques no convergen en tiempos razonables: el ruteo completo sigue siendo
  difícil aunque se acorten los días. Se deja en el código para poder
  reproducirlo.

En ambos intentos, el mínimo terminal se exige al cierre de cada ventana o
bloque: el modelo común ya lo exige el último día que ve, así que basta con
recortar los escenarios. Los escenarios se recortan, no se sortean de nuevo,
así que el azar es el mismo que en A y B.

- El bloque de ruteo de `modelo_comun.resolver_modelo_calendario` se copia
  aquí (como en la heurística B).
- La dinámica de inventario, los costos y la resolución se IMPORTAN del modelo
  común, para que la comparación sea justa. El único cambio al modelo común es
  el argumento opcional `inicio_por_escenario` de `_agregar_inventario`, que
  por defecto no cambia nada (lo usa sólo el intento 1).

Uso (misma firma que el modelo común):

    from codigo import heuristica_b, heuristica_c
    # Intento 2:
    simular(..., resolver_calendario=heuristica_b.resolver_modelo_calendario,
            resolver_cantidades=heuristica_c.resolver_modelo_cantidades)
    # Intento 1 (descartado):
    simular(..., resolver_calendario=heuristica_c.resolver_modelo_calendario_bloques,
            resolver_cantidades=heuristica_c.resolver_modelo_cantidades)
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from codigo import configuracion as cfg
from codigo import modelo_comun
from codigo.escenarios import Escenarios
from codigo.modelo_comun import EstadoSistema, SolucionCalendario


def recortar_escenarios(escenarios, desde, hasta):
    """Escenarios de los días desde, ..., hasta - 1 (los mismos sorteos, sin
    sortear de nuevo). Los días se cuentan desde el día 0 de `escenarios`."""
    return Escenarios(
        oferta=escenarios.oferta[:, desde:hasta],
        demanda=escenarios.demanda[:, desde:hasta],
        hospitales=escenarios.hospitales,
        mes=escenarios.mes,
    )


# ---------------------------------------------------------------------------
# Modelo de calendario por bloques
# ---------------------------------------------------------------------------


def _inventario_siguiente(modelo, escenarios, insumos):
    """Inventario con que parte el día siguiente al último del modelo, en cada
    escenario: lo que quedó al cierre, envejecido un día (lo de edad L - 1
    vence). Se lee de las variables B_cierre y V_cierre de `_agregar_inventario`.

    Devuelve: (banco (S, L), hospitales (S, n_h, L)), enteros.
    """
    L = insumos.parametros["shelf_life"]
    n_s, ultimo, n_h = escenarios.n_escenarios, escenarios.dias - 1, len(insumos.hospitales)
    banco = np.zeros((n_s, L))
    hospitales = np.zeros((n_s, n_h, L))
    for s in range(n_s):
        for a in range(1, L):
            banco[s, a] = modelo.getVarByName(f"B_cierre[{a - 1},{ultimo},{s}]").X
            for i in range(n_h):
                hospitales[s, i, a] = modelo.getVarByName(f"V_cierre[{i},{a - 1},{ultimo},{s}]").X
    # Son enteros (r, q y u lo son); se redondea sólo la tolerancia numérica de Gurobi.
    return np.round(banco).astype(int), np.round(hospitales).astype(int)


def _resolver_bloque(escenarios, estado, insumos, inicio_por_escenario):
    """Modelo de calendario del modelo común sobre los días de un bloque.

    Recibe:
        escenarios: Escenarios de los días del bloque.
        estado: EstadoSistema; su `dia` es el primer día del bloque en la
            semana. Si inicio_por_escenario es None, su inventario es el
            inicial de todos los escenarios.
        insumos: Insumos.
        inicio_por_escenario: None o el inventario inicial de cada escenario
            (ver `modelo_comun._agregar_inventario`).
    Devuelve: (SolucionCalendario del bloque, inventario con que parte el día
        siguiente al bloque en cada escenario, como en `_inventario_siguiente`).
    """
    n_h = len(insumos.hospitales)
    hospitales, vehiculos, dias = range(n_h), list(insumos.flota["vehiculo"]), range(escenarios.dias)
    alfa = insumos.parametros["costo_km"]
    delta = modelo_comun.matriz_distancias(insumos)
    # Nodos del ruteo: 0 = banco b, i + 1 = hospital i (sólo dentro de x).
    nodo = lambda i: i + 1
    nodos = range(n_h + 1)

    modelo = gp.Model("heuristica_c_calendario")
    modelo_comun._configurar(modelo, cfg.TIEMPO_LIMITE_CALENDARIO_S)

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
    costo_inventario, _ = modelo_comun._agregar_inventario(
        modelo, escenarios, estado, insumos,
        visita=lambda i, k, t: y[i, k, t], hoy_comun=False,
        inicio_por_escenario=inicio_por_escenario,
    )
    modelo.setObjective(transporte + costo_inventario, GRB.MINIMIZE)
    info = modelo_comun._resolver(modelo)

    hoy = estado.dia
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
    solucion = SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)
    return solucion, _inventario_siguiente(modelo, escenarios, insumos)


def resolver_calendario_bloques(escenarios, estado_inicial, insumos, bloques):
    """Calendario de la semana resuelto por bloques de días consecutivos.

    Recibe: lo mismo que `modelo_comun.resolver_modelo_calendario`, más
        bloques (largo en días de cada bloque, en orden; deben sumar los
        días de `escenarios`).
    Devuelve: SolucionCalendario con las visitas y rutas de todos los bloques.
        En info_resolucion: "optimo" sólo si todos los bloques lo son; "gap"
        es el mayor de los bloques; "valor_objetivo", "cota", "tiempo_s" y
        "trabajo" son la suma de los bloques (el valor objetivo incluye la
        penalización del mínimo al cierre de cada bloque); "bloques" tiene la
        información de cada uno.
    """
    if sum(bloques) != escenarios.dias or min(bloques) < 1:
        raise ValueError(f"Los bloques {bloques} deben ser positivos y sumar {escenarios.dias} días.")

    visitas, rutas, km, infos = set(), {}, 0.0, []
    estado, inicio, desde = estado_inicial, None, 0
    for largo in bloques:
        bloque = recortar_escenarios(escenarios, desde, desde + largo)
        solucion, inicio = _resolver_bloque(bloque, estado, insumos, inicio)
        visitas |= solucion.visitas
        rutas.update(solucion.rutas)
        km += solucion.distancia_km
        infos.append(solucion.info_resolucion)
        desde += largo
        # Del estado del bloque siguiente sólo se usa el día: su inventario va por escenario (inicio).
        estado = EstadoSistema(dia=estado_inicial.dia + desde, inventario_banco=None, inventario_hospitales=None)

    info = {
        "estado": [i["estado"] for i in infos],
        "optimo": all(i["optimo"] for i in infos),
        "valor_objetivo": sum(i["valor_objetivo"] for i in infos),
        "cota": sum(i["cota"] for i in infos),
        "gap": max(i["gap"] for i in infos),
        "tiempo_s": sum(i["tiempo_s"] for i in infos),
        "trabajo": sum(i["trabajo"] for i in infos),
        "bloques": infos,
    }
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)


# ---------------------------------------------------------------------------
# Modelo de cantidades con ventana
# ---------------------------------------------------------------------------


def resolver_cantidades_ventana(escenarios, estado, calendario, insumos, ventana):
    """Modelo de cantidades del modelo común sobre hoy y los días
    siguientes, hasta completar `ventana` días o llegar al domingo. El mínimo
    terminal se exige al cierre de la ventana.

    Recibe: lo mismo que `modelo_comun.resolver_modelo_cantidades`, más
        ventana (días que ve el modelo, contando hoy).
    Devuelve: DecisionesDia.
    """
    if ventana < 1:
        raise ValueError("La ventana debe ser de al menos 1 día.")
    dias = min(ventana, escenarios.dias)
    return modelo_comun.resolver_modelo_cantidades(recortar_escenarios(escenarios, 0, dias), estado, calendario, insumos)


# ---------------------------------------------------------------------------
# Funciones con la firma del modelo común
# ---------------------------------------------------------------------------


def crear_resolver_calendario(bloques):
    """Función con la firma de `modelo_comun.resolver_modelo_calendario` para
    unos bloques dados, lista para `simular(..., resolver_calendario=...)`."""

    def resolver(escenarios, estado_inicial, insumos):
        return resolver_calendario_bloques(escenarios, estado_inicial, insumos, bloques)

    return resolver


def crear_resolver_cantidades(ventana):
    """Función con la firma de `modelo_comun.resolver_modelo_cantidades` para
    una ventana dada, lista para `simular(..., resolver_cantidades=...)`."""

    def resolver(escenarios, estado, calendario, insumos):
        return resolver_cantidades_ventana(escenarios, estado, calendario, insumos, ventana)

    return resolver


def resolver_modelo_calendario_bloques(escenarios, estado_inicial, insumos):
    """Intento 1 (descartado): calendario por bloques de
    `cfg.BLOQUES_CALENDARIO_C`. Misma firma y resultado que el modelo común."""
    return resolver_calendario_bloques(escenarios, estado_inicial, insumos, cfg.BLOQUES_CALENDARIO_C)


def resolver_modelo_cantidades(escenarios, estado, calendario, insumos):
    """Cantidades con la ventana de `cfg.VENTANA_CANTIDADES_C`. Misma firma y
    resultado que el modelo común."""
    return resolver_cantidades_ventana(escenarios, estado, calendario, insumos, cfg.VENTANA_CANTIDADES_C)
