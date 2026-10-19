"""Variantes EXPLORADAS del caso base (METODOLOGIA.md Anexo 15.A), y funciones
comunes que usa el caso base oficial (`caso_base_agnostico.py`): reglas simples, sin optimización.

Es la política contra la que se compara la metodología: una forma intuitiva y
miope de operar, sin escenarios y mirando sólo el día en curso. Se enchufa
igual que una heurística (§11.7), así que corre sobre las mismas semanas, con
la misma actualización del inventario y los mismos KPIs.

Calendario (`resolver_modelo_calendario`, §15.A.1), el mismo todas las semanas:
    - 5 zonas fijas, una por vehículo: barrido angular alrededor del banco,
      cortado en grupos consecutivos con la demanda media lo más pareja posible.
    - Cada vehículo visita todos los días todos los hospitales de su zona.
    - Orden de la ruta: vecino más cercano, sin optimizar.

Cantidades (`resolver_modelo_cantidades`, §15.A.1), sólo con lo que se sabe hoy:
    - Admisión: toda la oferta que quepa en el banco (CFG "todo") o sólo lo que
      se despacha hoy más μ_b ("ajustada").
    - Pedido de cada hospital: max(0, η_i − inventario), con
      η_i = min(cuantil de su demanda diaria, C_i).
    - Si una zona pide más que θ_k, o la red más de lo que hay en el banco, se
      recorta en proporción al pedido, redondeando hacia abajo.
    - Se despacha primero lo más viejo del banco.

Uso:
    from codigo import caso_base
    simular_semanas(semanas, ins, n_escenarios=1, **caso_base.politica(0.95, "todo"))

Sin argumentos, `politica()` usa CUANTIL_CASO_BASE y ADMISION_CASO_BASE de
configuracion.py. Los escenarios que genera `ejecutar_semana` no se usan, así
que basta n_escenarios=1.

Modo con escenarios (`politica_escenarios`, §15.A.3): las mismas zonas, pero
cada día decide con los 30 escenarios que reciben las heurísticas, sin
optimizar: prueba niveles y colchones y elige el de menor costo promedio, y si
no alcanza reparte unidad por unidad donde es más probable que se use.

Para correr la sensibilidad completa (8 variantes + el modo con escenarios)
y guardar los resultados:
    python -m codigo.caso_base
"""

import math
import sys

import numpy as np
import pandas as pd

from codigo import cargador_datos, modelo_comun
from codigo import configuracion as cfg
from codigo.modelo_comun import DecisionesDia, SolucionCalendario

ARCHIVO_SEMANAS = "01_datos/semanas_simuladas_2024_5xmes.csv"
ARCHIVO_RESULTADOS = "resultados/caso_base_semanas_2024_5xmes.csv"
SEMILLA_RESULTADOS = 2024  # la misma de la heurística A (resultados/README.md)


# ---------------------------------------------------------------------------
# Parámetros del caso base
# ---------------------------------------------------------------------------


def valor_cuantil(cuantil, insumos):
    """Convierte el cuantil de configuración en un número: "critico" es el
    fractil crítico π / (π + γ) con los costos de parametros.csv."""
    if cuantil == "critico":
        par = insumos.parametros
        return par["costo_shortage"] / (par["costo_shortage"] + par["costo_holding_hospital"])
    return float(cuantil)


def _demanda_historica(insumos):
    """Demanda diaria histórica con que se construye la política: la de los
    insumos (en la prueba real, solo 2022–2023) o, si no se guardó, toda."""
    if insumos.demanda_historica is not None:
        return insumos.demanda_historica
    return cargador_datos.cargar_demanda_historica()


def niveles_objetivo(insumos, cuantil, fuente=None):
    """η_i = min(cuantil de la demanda diaria del hospital i, C_i) (§15.A.1).

    Recibe: cuantil y fuente ("historico": percentil directo de la demanda
        histórica, redondeado hacia arriba; "distribucion": percentil de la
        distribución ajustada; por defecto FUENTE_NIVEL_CASO_BASE).
    Devuelve: arreglo de enteros por hospital (orden de Insumos.hospitales).
    """
    fuente = cfg.FUENTE_NIVEL_CASO_BASE if fuente is None else fuente
    q = valor_cuantil(cuantil, insumos)
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    if fuente == "historico":
        percentiles = _demanda_historica(insumos).groupby("hospital")["demanda"].quantile(q)  # interpolación lineal, como Excel
        nivel = lambda nombre: math.ceil(percentiles[nombre] - 1e-9)
    elif fuente == "distribucion":
        nivel = lambda nombre: int(insumos.distribuciones_demanda[nombre].distribucion.ppf(q))
    else:
        raise ValueError(f"Fuente '{fuente}' no reconocida.")
    return np.array([min(nivel(nombre), int(C[i])) for i, nombre in enumerate(insumos.hospitales["nombre"])])


def demanda_media(insumos):
    """Demanda diaria media de cada hospital: el promedio histórico o, con
    FUENTE_NIVEL_CASO_BASE = "distribucion", la media de su distribución
    ajustada. Son iguales, porque el ajuste conserva la media."""
    if cfg.FUENTE_NIVEL_CASO_BASE == "historico":
        historia = _demanda_historica(insumos)
        medias = historia.groupby("hospital")["demanda"].mean()
        return np.array([medias[n] for n in insumos.hospitales["nombre"]])
    return np.array([insumos.distribuciones_demanda[n].distribucion.mean() for n in insumos.hospitales["nombre"]])


def zonas(insumos):
    """Las zonas fijas de los vehículos (§15.A.1).

    Los hospitales se ordenan por su ángulo respecto al banco y se cortan en
    tantos grupos consecutivos como vehículos. Entre todos los puntos de partida
    del barrido y todos los cortes, se elige el que deja la zona más cargada
    (en demanda media) lo más baja posible.

    Devuelve: lista, por vehículo (orden de Insumos.flota), de listas de hospitales i.
    Falla si alguna zona demanda en promedio más que su vehículo.
    """
    nodos = cargador_datos.cargar_hospitales()
    banco = nodos.loc[nodos["nombre"] == insumos.nombre_banco].iloc[0]
    hospitales = insumos.hospitales
    angulo = np.arctan2(hospitales["lat"].to_numpy() - banco["lat"], hospitales["lon"].to_numpy() - banco["lon"])
    media = demanda_media(insumos)
    orden = [int(i) for i in np.argsort(angulo, kind="stable")]
    n, K = len(orden), len(insumos.flota)

    mejor = None
    for inicio in range(n):
        secuencia = orden[inicio:] + orden[:inicio]
        acumulada = np.concatenate([[0.0], np.cumsum(media[secuencia])])
        # carga[k][j]: menor zona más cargada al cortar los primeros j hospitales en k zonas.
        carga = np.full((K + 1, n + 1), np.inf)
        carga[0, 0] = 0.0
        corte = {}
        for k in range(1, K + 1):
            for j in range(k, n + 1):
                for i in range(k - 1, j):
                    valor = max(carga[k - 1, i], acumulada[j] - acumulada[i])
                    if valor < carga[k, j]:
                        carga[k, j], corte[k, j] = valor, i
        if mejor is None or carga[K, n] < mejor[0] - 1e-9:
            grupos, j = [], n
            for k in range(K, 0, -1):
                i = corte[k, j]
                grupos.append(secuencia[i:j])
                j = i
            mejor = (carga[K, n], grupos[::-1])

    grupos = mejor[1]
    for grupo, theta in zip(grupos, insumos.flota["capacidad"]):
        if media[grupo].sum() > theta:
            raise ValueError("Una zona demanda en promedio más que la capacidad de su vehículo.")
    return grupos


def vecino_mas_cercano(hospitales, delta):
    """Orden de visita por vecino más cercano desde el banco (§15.A.1).

    Recibe: hospitales i y la matriz δ de `modelo_comun.matriz_distancias`
        (banco en la posición 0, hospital i en la i + 1).
    Devuelve: (lista de hospitales en orden de visita, km de la ruta ida y vuelta).
    """
    ruta, pendientes, actual, km = [], set(hospitales), 0, 0.0
    while pendientes:
        siguiente = min(pendientes, key=lambda i: (delta[actual, i + 1], i))
        km += delta[actual, siguiente + 1]
        ruta.append(siguiente)
        pendientes.remove(siguiente)
        actual = siguiente + 1
    return ruta, km + delta[actual, 0]


def recortar(pedidos, disponible):
    """Si los pedidos superan lo disponible, los recorta en proporción,
    redondeando hacia abajo (§15.A.1). Si alcanza, los deja iguales."""
    pedidos = np.asarray(pedidos, dtype=int)
    if pedidos.sum() <= disponible:
        return pedidos
    return np.floor(pedidos * (max(disponible, 0) / pedidos.sum())).astype(int)


# ---------------------------------------------------------------------------
# Las dos funciones del caso base
# ---------------------------------------------------------------------------


def politica(cuantil=None, admision=None):
    """Las funciones del caso base para un cuantil y una regla de admisión.

    Recibe: cuantil (número o "critico"; por defecto CUANTIL_CASO_BASE) y
        admision ("todo" o "ajustada"; por defecto ADMISION_CASO_BASE).
    Devuelve: {"resolver_calendario": ..., "resolver_cantidades": ...}, listo
        para pasar a `simular_semanas`, `simular_semana` o `ejecutar_semana`.
    """
    cuantil = cfg.CUANTIL_CASO_BASE if cuantil is None else cuantil
    admision = cfg.ADMISION_CASO_BASE if admision is None else admision
    if admision not in ("todo", "ajustada"):
        raise ValueError(f"Regla de admisión '{admision}' no reconocida.")

    def calendario(escenarios, estado_inicial, insumos):
        return resolver_modelo_calendario(escenarios, estado_inicial, insumos)

    def cantidades(escenarios, estado, calendario_fijo, insumos):
        return resolver_modelo_cantidades(escenarios, estado, calendario_fijo, insumos, cuantil, admision)

    return {"resolver_calendario": calendario, "resolver_cantidades": cantidades}


def resolver_modelo_calendario(escenarios, estado_inicial, insumos):
    """Calendario del caso base (§15.A.1): cada vehículo visita su zona todos
    los días. Misma firma que `modelo_comun.resolver_modelo_calendario`; no usa
    los escenarios ni el inventario.

    Devuelve: SolucionCalendario.
    """
    delta = modelo_comun.matriz_distancias(insumos)
    vehiculos = list(insumos.flota["vehiculo"])
    hoy = estado_inicial.dia
    visitas, rutas, km = set(), {}, 0.0
    for k, zona in zip(vehiculos, zonas(insumos)):
        ruta, largo = vecino_mas_cercano(zona, delta)
        for t in range(hoy, insumos.parametros["horizonte"]):
            rutas[t, k] = ruta
            visitas |= {(t, i, k) for i in ruta}
            km += largo
    info = {"metodo": "caso_base", "optimo": None, "gap": None}
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)


def resolver_modelo_cantidades(escenarios, estado, calendario, insumos, cuantil=None, admision=None):
    """Decisiones de hoy del caso base (§15.A.1). Misma firma que
    `modelo_comun.resolver_modelo_cantidades`, más el cuantil y la regla de
    admisión (por defecto, los de configuracion.py). De los escenarios sólo se
    usa la oferta de hoy, que es la real.

    Devuelve: DecisionesDia.
    """
    cuantil = cfg.CUANTIL_CASO_BASE if cuantil is None else cuantil
    admision = cfg.ADMISION_CASO_BASE if admision is None else admision
    L = insumos.parametros["shelf_life"]
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    mu_b, _ = modelo_comun.minimos_terminales(insumos)
    eta = niveles_objetivo(insumos, cuantil)
    inventario = estado.inventario_hospitales.sum(axis=1)
    oferta_hoy = int(escenarios.oferta[0, 0])

    # Pedidos de cada hospital visitado hoy, recortados por la capacidad de su vehículo.
    zona_hoy = {}
    for t, i, k in calendario.visitas:
        if t == estado.dia:
            zona_hoy.setdefault(k, []).append(i)
    envios = []
    for k in sorted(zona_hoy):
        hospitales = sorted(zona_hoy[k])
        pedidos = [max(0, int(eta[i] - inventario[i])) for i in hospitales]
        envios += [(i, k, int(u)) for i, u in zip(hospitales, recortar(pedidos, theta[k]))]
    pedido_total = sum(u for *_, u in envios)

    # Admisión.
    banco = estado.inventario_banco.astype(int).copy()
    espacio = insumos.capacidad_banco - int(banco.sum())
    if admision == "todo":
        admitida = min(oferta_hoy, espacio)
    else:
        admitida = min(oferta_hoy, espacio, max(0, math.ceil(pedido_total + mu_b) - int(banco.sum())))
    admitida = max(0, int(admitida))
    banco[0] += admitida

    # Recorte por lo que hay en el banco y despacho FIFO (lo más viejo primero).
    unidades = recortar([u for *_, u in envios], int(banco.sum()))
    despachos = {}
    for (i, k, _), pendiente in zip(envios, unidades):
        for a in range(L - 1, -1, -1):
            tomar = int(min(pendiente, banco[a]))
            if tomar > 0:
                despachos[i, k, a] = tomar
                banco[a] -= tomar
                pendiente -= tomar

    return DecisionesDia(oferta_admitida=admitida, despachos=despachos, info_resolucion={"metodo": "caso_base"})


# ---------------------------------------------------------------------------
# Modo con escenarios (§15.A.3)
# ---------------------------------------------------------------------------
# Las mismas zonas y visitas diarias, pero las cantidades se deciden mirando
# los escenarios que `ejecutar_semana` genera cada día (los mismos que reciben
# las heurísticas). No se optimiza: se prueban opciones, se promedia el costo
# en los escenarios y se elige la más barata, o se cuenta en cuántos
# escenarios se usaría cada unidad.


def nivel_hospital(demandas, inventario, capacidad, gamma, pi):
    """Regla 1: el nivel S (entre el inventario actual y la capacidad) con
    menor costo promedio en los escenarios de hoy. Cada unidad que sobra pasa
    la noche (γ) y cada unidad que falta es faltante (π). Si hay empate, el menor."""
    niveles = np.arange(inventario, max(inventario, capacidad) + 1)[:, None]
    d = np.asarray(demandas)[None, :]
    costo = (gamma * np.maximum(niveles - d, 0) + pi * np.maximum(d - niveles, 0)).mean(axis=1)
    return int(niveles[np.argmin(costo), 0])


def colchon_banco(faltas, capacidad, h_b, pi):
    """Regla 3: el colchón b que el banco guarda para mañana con menor costo
    promedio en los escenarios. Las b unidades pasan la noche (h_b cada una) y lo
    que mañana falte por sobre b es faltante (π). Si hay empate, el menor."""
    colchones = np.arange(0, capacidad + 1)[:, None]
    f = np.asarray(faltas)[None, :]
    costo = h_b * colchones[:, 0] + pi * np.maximum(f - colchones, 0).mean(axis=1)
    return int(colchones[np.argmin(costo), 0])


def repartir_por_probabilidad(pedidos, inventario, demandas, vehiculo_de, theta, disponible):
    """Regla 2: entrega unidad por unidad al hospital donde es más probable que
    esa unidad se use hoy, contando en cuántos escenarios la demanda la alcanza.
    Nadie recibe más que su pedido ni un vehículo lleva más que θ_k. Si alcanza
    para todos, cada hospital recibe exactamente su pedido.

    Recibe: pedidos {i: unidades}, inventario (arreglo por hospital), demandas
        (S, n_h) de hoy en los escenarios, vehiculo_de {i: k}, theta {k: θ_k} y
        las unidades disponibles en el banco.
    Devuelve: {i: unidades entregadas}.
    """
    import heapq

    entregado = {i: 0 for i in pedidos}
    libre = dict(theta)
    probabilidad = lambda i: float(np.mean(demandas[:, i] > inventario[i] + entregado[i]))
    cola = [(-probabilidad(i), i) for i in sorted(pedidos) if pedidos[i] > 0]
    heapq.heapify(cola)
    while cola and disponible > 0:
        _, i = heapq.heappop(cola)
        k = vehiculo_de[i]
        if libre[k] == 0:
            continue
        entregado[i] += 1
        libre[k] -= 1
        disponible -= 1
        if entregado[i] < pedidos[i]:
            heapq.heappush(cola, (-probabilidad(i), i))
    return entregado


def politica_escenarios():
    """Las funciones del caso base con escenarios, listas para `simular_semanas`.
    Necesita escenarios: correrlo con n_escenarios = N_ESCENARIOS_DEFAULT (30)."""
    return {"resolver_calendario": resolver_modelo_calendario, "resolver_cantidades": resolver_cantidades_escenarios}


def resolver_cantidades_escenarios(escenarios, estado, calendario, insumos):
    """Decisiones de hoy del caso base con escenarios (§15.A.3). Misma firma que
    `modelo_comun.resolver_modelo_cantidades`.

    1. Nivel de cada hospital: el de menor costo promedio en la demanda de hoy de
       los escenarios (regla 1). Pedido = nivel − inventario.
    2. Banco: admite lo que se despacha hoy más un colchón para mañana. El colchón
       es el de menor costo promedio frente a lo que faltaría mañana en cada
       escenario: lo que habrá que reponer (la demanda de hoy, hasta lo que lleva
       la flota) menos la oferta de mañana (regla 3). El domingo, el colchón es μ_b.
    3. Si no alcanza (vehículo o banco), unidad por unidad al hospital donde es
       más probable que se use (regla 2). Despacho FIFO.

    Devuelve: DecisionesDia.
    """
    par = insumos.parametros
    L = par["shelf_life"]
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    mu_b, _ = modelo_comun.minimos_terminales(insumos)
    demandas = modelo_comun._demanda_por_hospital(escenarios, insumos)  # (S, T, n_h)
    inventario = estado.inventario_hospitales.sum(axis=1)
    oferta_hoy = int(escenarios.oferta[0, 0])

    # Regla 1: pedido de cada hospital visitado hoy.
    vehiculo_de = {i: k for t, i, k in calendario.visitas if t == estado.dia}
    pedidos = {
        i: nivel_hospital(demandas[:, 0, i], int(inventario[i]), int(C[i]),
                          par["costo_holding_hospital"], par["costo_shortage"]) - int(inventario[i])
        for i in sorted(vehiculo_de)
    }
    por_vehiculo = {}
    for i, k in vehiculo_de.items():
        por_vehiculo[k] = por_vehiculo.get(k, 0) + pedidos[i]
    despacho_hoy = sum(min(total, theta[k]) for k, total in por_vehiculo.items())

    # Regla 3: admisión = lo de hoy + colchón para mañana.
    banco = estado.inventario_banco.astype(int).copy()
    if escenarios.dias >= 2:
        reponer = np.minimum(demandas[:, 0, :].sum(axis=1), sum(theta.values()))
        faltas = np.maximum(0, reponer - escenarios.oferta[:, 1])
        colchon = colchon_banco(faltas, insumos.capacidad_banco, par["costo_holding_banco"], par["costo_shortage"])
    else:
        colchon = math.ceil(mu_b)
    espacio = insumos.capacidad_banco - int(banco.sum())
    admitida = max(0, int(min(oferta_hoy, espacio, despacho_hoy + colchon - int(banco.sum()))))
    banco[0] += admitida

    # Regla 2: reparto por probabilidad de uso y despacho FIFO.
    entregado = repartir_por_probabilidad(pedidos, inventario, demandas[:, 0, :], vehiculo_de, theta, int(banco.sum()))
    despachos = {}
    for i in sorted(entregado):
        pendiente, k = entregado[i], vehiculo_de[i]
        for a in range(L - 1, -1, -1):
            tomar = int(min(pendiente, banco[a]))
            if tomar > 0:
                despachos[i, k, a] = tomar
                banco[a] -= tomar
                pendiente -= tomar

    return DecisionesDia(oferta_admitida=admitida, despachos=despachos,
                         info_resolucion={"metodo": "caso_base_escenarios", "colchon": colchon})


# ---------------------------------------------------------------------------
# Sensibilidad sobre las 60 semanas (§15.A.1)
# ---------------------------------------------------------------------------


def nombre_variante(cuantil, admision):
    etiqueta = "critico" if cuantil == "critico" else f"P{round(100 * float(cuantil))}"
    return f"{etiqueta}_{admision}"


def correr_variantes(insumos, semanas, variantes=None, semilla=SEMILLA_RESULTADOS):
    """Corre cada variante (cuantil, admisión) sobre las semanas.

    Devuelve: DataFrame con una fila por variante y semana (columnas de
        `simulacion.simular_semanas` más variante, cuantil y admision).
    """
    from codigo.simulacion import simular_semanas

    tablas = []
    for cuantil, admision in variantes or cfg.VARIANTES_CASO_BASE:
        tabla = simular_semanas(semanas, insumos, n_escenarios=1, semilla=semilla, **politica(cuantil, admision))
        tabla.insert(0, "admision", admision)
        tabla.insert(0, "cuantil", round(valor_cuantil(cuantil, insumos), 4))
        tabla.insert(0, "variante", nombre_variante(cuantil, admision))
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def correr_escenarios(insumos, semanas, semilla=SEMILLA_RESULTADOS, n_escenarios=None):
    """Corre el caso base con escenarios (§15.A.3) sobre las semanas, con los
    mismos escenarios que recibirían las heurísticas (misma semilla y número).

    Devuelve: DataFrame con una fila por semana y variante "escenarios".
    """
    from codigo.simulacion import simular_semanas

    n_escenarios = cfg.N_ESCENARIOS_DEFAULT if n_escenarios is None else n_escenarios
    tabla = simular_semanas(semanas, insumos, n_escenarios=n_escenarios, semilla=semilla, **politica_escenarios())
    tabla.insert(0, "admision", "escenarios")
    tabla.insert(0, "cuantil", np.nan)
    tabla.insert(0, "variante", "escenarios")
    return tabla


def resumen_anual(tabla):
    """Promedio por semana del año completo, ponderando los 12 meses por igual
    (como en resultados/README.md), para cada variante."""
    columnas = ["costo_operacional", "costo_transporte", "costo_holding_banco", "costo_holding_hospitales",
                "costo_shortage", "costo_wastage", "penalizacion_terminal", "costo_total", "km",
                "unidades_faltantes", "unidades_vencidas", "deficit_terminal", "fill_rate_red"]
    tabla = tabla.assign(mes=pd.to_datetime(tabla["fecha_lunes"]).dt.month)
    por_mes = tabla.groupby(["variante", "mes"], sort=False)[columnas].mean()
    return por_mes.groupby(level="variante", sort=False).mean()


def main():
    from codigo.horizonte_rodante import preparar_insumos
    from codigo.simulacion import cargar_semanas

    insumos = preparar_insumos()
    semanas = cargar_semanas(ARCHIVO_SEMANAS, insumos)
    tabla = pd.concat([correr_variantes(insumos, semanas), correr_escenarios(insumos, semanas)], ignore_index=True)
    tabla.to_csv(ARCHIVO_RESULTADOS, index=False, encoding="utf-8")
    pd.set_option("display.width", 250)
    resumen = resumen_anual(tabla).T
    print(resumen.map(lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:,.0f}").to_string())
    print(f"\nGuardado en {ARCHIVO_RESULTADOS} ({len(tabla)} filas).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
