"""Caso base OFICIAL del SIRP: política agnóstica a la demanda y a la oferta.

Sus reglas usan solo la estructura del problema: distancias, capacidades de
hospitales, camiones y banco, y vida útil. Ninguna regla usa demanda ni oferta
histórica. La demanda y la oferta aparecen recién cuando se corre sobre un
escenario; de la oferta solo se usa la de hoy, que se observa al inicio del día.

Algoritmo:
    Una vez, antes de operar:
    1. Zonas: una por camión. Se parte por el hospital sin zona más lejano del
       banco y se le suma el hospital sin zona más cercano a la zona, hasta
       juntar ~1/K de la capacidad de almacenamiento total de los hospitales.
    2. Rutas: cada camión visita su zona todos los días, en orden de vecino
       más cercano desde el banco.
    3. Nivel objetivo de cada hospital: ⌈β · C_i⌉, con β = capacidad diaria
       de la flota / capacidad total de los hospitales.
    Cada día t, viendo solo la oferta de hoy O_t y el inventario actual:
    4. Pedido de cada hospital: max(0, nivel − inventario del hospital).
    5. Si una zona pide más que su camión, se recorta en proporción (hacia abajo).
    6. El banco admite min(O_t, espacio libre, pedidos + μ_b − inventario del
       banco), con μ_b = 10% de su capacidad.
    7. Si el banco no alcanza, se recorta en proporción; se despacha FIFO.

Uso:
    from codigo import caso_base_agnostico as cba
    simular_semanas_encadenadas(semanas, ins, 1, 2024, **cba.politica())

Para correr todo (60 semanas simuladas y escenarios construidos sobre 2024)
y guardar los resultados (segundos):
    python -m codigo.caso_base_agnostico
"""

import math
import sys

import numpy as np
import pandas as pd

from codigo import caso_base, kpis, modelo_comun
from codigo import configuracion as cfg
from codigo.modelo_comun import DecisionesDia, SolucionCalendario

ARCHIVO_SIMULADAS = "resultados/caso_base_agnostico_semanas_2024_5xmes.csv"
ARCHIVO_ESCENARIOS = "resultados/caso_base_agnostico_escenarios_2024.csv"
ARCHIVO_INTERVALOS = "resultados/caso_base_agnostico_intervalos.csv"
ARCHIVO_HOSPITALES = "resultados/caso_base_agnostico_hospitales.csv"
ESCENARIOS = ("real", "favorable", "neutro", "desfavorable")
SEMILLA = 2024  # la de las heurísticas: mismos escenarios del modelo por semana
KPIS = ["costo_operacional", "costo_total", "costo_transporte", "costo_holding_banco", "costo_holding_hospitales",
        "costo_shortage", "costo_wastage", "penalizacion_terminal", "deficit_terminal", "fill_rate_red",
        "unidades_faltantes", "unidades_vencidas", "oferta_rechazada", "km", "utilizacion_flota", "unidades_despachadas"]


# ---------------------------------------------------------------------------
# Reglas fijas (solo estructura)
# ---------------------------------------------------------------------------


def valor_beta(insumos, beta=None):
    """β como número. None = capacidad diaria de la flota / capacidad total de los hospitales."""
    beta = cfg.BETA_CASO_BASE if beta is None else beta
    if beta is None:
        return float(insumos.flota["capacidad"].sum() / insumos.hospitales["capacidad_maxima"].sum())
    return float(beta)


def niveles(insumos, beta=None):
    """Nivel objetivo de cada hospital: ⌈β · C_i⌉ (orden de Insumos.hospitales)."""
    b = valor_beta(insumos, beta)
    return np.array([math.ceil(b * c - 1e-9) for c in insumos.hospitales["capacidad_maxima"]], dtype=int)


def zonas_por_distancia(insumos):
    """Una zona por camión, armada por cercanía y equilibrada por capacidad de
    almacenamiento (no por demanda).

    Se parte por el hospital sin zona más lejano del banco y se le suma el
    hospital sin zona más cercano a cualquiera de la zona, mientras la
    capacidad de la zona no pase el objetivo (capacidad total / K) en más de la
    mitad de la capacidad del que se agrega. El último camión queda con el resto.

    Devuelve: lista, por vehículo (orden de Insumos.flota), de listas de hospitales i.
    """
    delta = modelo_comun.matriz_distancias(insumos)
    d = (delta[1:, 1:] + delta[1:, 1:].T) / 2
    C = insumos.hospitales["capacidad_maxima"].to_numpy(dtype=float)
    K = len(insumos.flota)
    objetivo = C.sum() / K
    libres, zonas = set(range(len(C))), []
    for z in range(K):
        if z == K - 1:
            zonas.append(sorted(libres))
            break
        semilla = max(libres, key=lambda i: (delta[0, i + 1], -i))
        zona = [semilla]
        libres.remove(semilla)
        while libres:
            cercano = min(libres, key=lambda j: (min(d[j, i] for i in zona), j))
            if C[zona].sum() + C[cercano] > objetivo + 0.5 * C[cercano]:
                break
            zona.append(cercano)
            libres.remove(cercano)
        zonas.append(zona)
    return zonas


# ---------------------------------------------------------------------------
# Las dos funciones de la política
# ---------------------------------------------------------------------------


def politica(beta=None):
    """{"resolver_calendario": ..., "resolver_cantidades": ...} para un β (por defecto BETA_CASO_BASE)."""
    def cantidades(escenarios, estado, calendario, insumos):
        return resolver_modelo_cantidades(escenarios, estado, calendario, insumos, beta)

    return {"resolver_calendario": resolver_modelo_calendario, "resolver_cantidades": cantidades}


def resolver_modelo_calendario(escenarios, estado_inicial, insumos):
    """Calendario fijo: cada camión visita su zona todos los días, por vecino
    más cercano. Misma firma que `modelo_comun.resolver_modelo_calendario`;
    no usa los escenarios ni el inventario."""
    delta = modelo_comun.matriz_distancias(insumos)
    visitas, rutas, km = set(), {}, 0.0
    for k, zona in zip(insumos.flota["vehiculo"], zonas_por_distancia(insumos)):
        ruta, largo = caso_base.vecino_mas_cercano(zona, delta)
        for t in range(estado_inicial.dia, insumos.parametros["horizonte"]):
            rutas[t, k] = ruta
            visitas |= {(t, i, k) for i in ruta}
            km += largo
    info = {"metodo": "caso_base_agnostico", "optimo": None, "gap": None}
    return SolucionCalendario(visitas=visitas, rutas=rutas, distancia_km=km, info_resolucion=info)


def resolver_modelo_cantidades(escenarios, estado, calendario, insumos, beta=None):
    """Decisiones de hoy (pasos 4 a 7 del algoritmo). Misma firma que
    `modelo_comun.resolver_modelo_cantidades`, más β. De los escenarios solo se
    usa la oferta de hoy, que es la real."""
    L = insumos.parametros["shelf_life"]
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    mu_b, _ = modelo_comun.minimos_terminales(insumos)
    nivel = niveles(insumos, beta)
    inventario = estado.inventario_hospitales.sum(axis=1)
    oferta_hoy = int(escenarios.oferta[0, 0])

    zona_hoy = {}
    for t, i, k in calendario.visitas:
        if t == estado.dia:
            zona_hoy.setdefault(k, []).append(i)
    envios = []
    for k in sorted(zona_hoy):
        hospitales = sorted(zona_hoy[k])
        pedidos = [max(0, int(nivel[i] - inventario[i])) for i in hospitales]
        envios += [(i, k, int(u)) for i, u in zip(hospitales, caso_base.recortar(pedidos, theta[k]))]

    banco = estado.inventario_banco.astype(int).copy()
    espacio = insumos.capacidad_banco - int(banco.sum())
    pedido_total = sum(u for *_, u in envios)
    admitida = max(0, int(min(oferta_hoy, espacio, math.ceil(pedido_total + mu_b) - int(banco.sum()))))
    banco[0] += admitida

    unidades = caso_base.recortar([u for *_, u in envios], int(banco.sum()))
    despachos = {}
    for (i, k, _), pendiente in zip(envios, unidades):
        for a in range(L - 1, -1, -1):
            tomar = int(min(pendiente, banco[a]))
            if tomar > 0:
                despachos[i, k, a] = tomar
                banco[a] -= tomar
                pendiente -= tomar
    return DecisionesDia(oferta_admitida=admitida, despachos=despachos, info_resolucion={"metodo": "caso_base_agnostico"})


# ---------------------------------------------------------------------------
# Corridas
# ---------------------------------------------------------------------------


def nombre_variante(insumos, beta=None):
    return f"agnostico_{round(100 * valor_beta(insumos, beta))}"


def _con_rechazo(beta):
    """Política que además anota la oferta rechazada de cada día."""
    pol, rechazos = politica(beta), []

    def cantidades(escenarios, estado, calendario, insumos):
        d = pol["resolver_cantidades"](escenarios, estado, calendario, insumos)
        rechazos.append(int(escenarios.oferta[0, 0]) - d.oferta_admitida)
        return d

    return {"resolver_calendario": pol["resolver_calendario"], "resolver_cantidades": cantidades}, rechazos


def correr(insumos, semanas, encadenadas, betas=None):
    """Corre cada β sobre las semanas (encadenadas o independientes).

    Devuelve: DataFrame con una fila por β y semana, con la oferta rechazada.
    """
    from codigo.simulacion import simular_semanas, simular_semanas_encadenadas

    tablas = []
    for beta in betas or cfg.BETAS_SENSIBILIDAD_CASO_BASE:
        pol, rechazos = _con_rechazo(beta)
        correr_ = simular_semanas_encadenadas if encadenadas else simular_semanas
        tabla = correr_(semanas, insumos, 1, SEMILLA, **pol)
        tabla["oferta_rechazada"] = np.array(rechazos).reshape(-1, insumos.parametros["horizonte"]).sum(axis=1)
        tabla["oferta_semana"] = [int(s.oferta.sum()) for s in semanas]
        tabla.insert(0, "beta", round(valor_beta(insumos, beta), 4))
        tabla.insert(0, "variante", nombre_variante(insumos, beta))
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def correr_escenarios(insumos, semanas_reales, betas=None):
    """Corre cada β en las semanas reales encadenadas y en cada escenario construido."""
    from codigo.evaluacion_historica import escenario_construido

    tablas = []
    for escenario in ESCENARIOS:
        tabla = correr(insumos, escenario_construido(semanas_reales, escenario), encadenadas=True, betas=betas)
        tabla.insert(0, "escenario", escenario)
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def tabla_intervalos(tabla, columnas=("escenario", "variante")):
    """Promedio e intervalo del 95% de cada KPI por escenario y variante."""
    from codigo.evaluacion_historica import intervalo

    filas = []
    for clave, grupo in tabla.groupby(list(columnas), sort=False):
        for kpi in KPIS:
            p, a, b, n = intervalo(grupo[kpi])
            filas.append({**dict(zip(columnas, clave)), "kpi": kpi, "promedio": p, "inferior": a, "superior": b, "semanas": n})
    return pd.DataFrame(filas)


def tabla_hospitales(tabla, insumos):
    """Fill rate de cada hospital por escenario y variante, con su nivel objetivo."""
    filas = []
    for (escenario, variante), grupo in tabla.groupby(["escenario", "variante"], sort=False):
        h = kpis.kpis_hospitales(grupo.reset_index(drop=True), insumos)
        nivel = dict(zip(insumos.hospitales["nombre"], niveles(insumos, float(grupo["beta"].iloc[0]))))
        h.insert(0, "variante", variante)
        h.insert(0, "escenario", escenario)
        h["nivel_objetivo"] = h["hospital"].map(nivel)
        filas.append(h)
    return pd.concat(filas, ignore_index=True)


def main():
    from codigo import evaluacion_historica
    from codigo.horizonte_rodante import preparar_insumos
    from codigo.simulacion import cargar_semanas

    # 60 semanas simuladas (independientes): para comparar con las heurísticas.
    insumos = preparar_insumos()
    simuladas = correr(insumos, cargar_semanas(caso_base.ARCHIVO_SEMANAS, insumos), encadenadas=False)
    simuladas.to_csv(ARCHIVO_SIMULADAS, index=False, encoding="utf-8")

    # 52 semanas reales de 2024 encadenadas, tal cual y en los escenarios construidos.
    reales = evaluacion_historica.preparar_insumos()
    escenarios = correr_escenarios(reales, evaluacion_historica.semanas_historicas(reales))
    escenarios.to_csv(ARCHIVO_ESCENARIOS, index=False, encoding="utf-8")
    intervalos = tabla_intervalos(escenarios)
    intervalos.to_csv(ARCHIVO_INTERVALOS, index=False, encoding="utf-8")
    tabla_hospitales(escenarios, reales).to_csv(ARCHIVO_HOSPITALES, index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    principal = nombre_variante(reales)
    for kpi in ("costo_operacional", "fill_rate_red", "unidades_faltantes"):
        v = intervalos[(intervalos["kpi"] == kpi)]
        formato = (lambda x: f"{100 * x:.1f}%") if kpi == "fill_rate_red" else (lambda x: f"{x / 1e6:.2f} M" if x > 1e4 else f"{x:,.0f}")
        v = v.assign(texto=[f"{formato(p)} [{formato(a)}; {formato(b)}]" for p, a, b in zip(v["promedio"], v["inferior"], v["superior"])])
        print(f"\n{kpi} (52 semanas de 2024 encadenadas, promedio [IC 95%]):")
        print(v.pivot(index="variante", columns="escenario", values="texto")[list(ESCENARIOS)].to_string())
    print(f"\nPrincipal: {principal} (β = {valor_beta(reales):.4f}). Guardado en {ARCHIVO_SIMULADAS}, {ARCHIVO_ESCENARIOS}, "
          f"{ARCHIVO_INTERVALOS} y {ARCHIVO_HOSPITALES}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
