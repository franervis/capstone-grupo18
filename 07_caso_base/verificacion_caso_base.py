"""Verificación del caso base (METODOLOGIA.md §15.A.2 y Anexo 15.A).

Las primeras 5 pruebas son del caso base OFICIAL (agnóstico); el resto, de
las variantes exploradas antes (registro).

Se corre desde la raíz del repo con:

    python -m codigo.verificacion_caso_base

Imprime OK / FALLA para cada prueba y termina con código de salida 1 si
alguna falla. No necesita licencia de Gurobi: el caso base no resuelve
ningún modelo.

Revisa que el caso base haga lo que dice la §15 y que las variantes del
Anexo 15.A (incluido el modo con escenarios del 15.A.3) hagan lo suyo, y que, cada día de cada semana y en todas sus variantes, respete las mismas restricciones que el
modelo: capacidad de hospitales, banco y vehículos, y despacho sólo a los
hospitales del calendario.
"""

import sys

import numpy as np
import pandas as pd

import dataclasses

from codigo import caso_base, caso_base_agnostico as cba, cargador_datos, evaluacion_historica, modelo_comun
from codigo import configuracion as cfg
from codigo.escenarios import Escenarios
from codigo.horizonte_rodante import ejecutar_semana, estado_inicial_semana, preparar_insumos
from codigo.modelo_comun import EstadoSistema
from codigo.simulacion import cargar_semanas, simular_semanas_encadenadas
from codigo.verificacion import P95_TABLA_A3

_BASE = None


def _base():
    """Insumos reales (se cargan una sola vez)."""
    global _BASE
    if _BASE is None:
        _BASE = preparar_insumos()
    return _BASE


def _escenario_hoy(insumos, oferta):
    """Escenarios mínimos: un escenario de un día con la oferta de hoy."""
    n_h = len(insumos.hospitales)
    return Escenarios(oferta=np.array([[oferta]]), demanda=np.zeros((1, 1, n_h), dtype=int),
                      hospitales=list(insumos.hospitales["nombre"]), mes=1)


# ---------------------------------------------------------------------------
# Variantes exploradas: reglas de la propuesta original (Anexo 15.A.1)
# ---------------------------------------------------------------------------


def prueba_zonas():
    """Las zonas cubren los 22 hospitales sin repetir, son consecutivas en el
    barrido y ninguna demanda en promedio más que su vehículo."""
    insumos = _base()
    grupos = caso_base.zonas(insumos)
    todos = sorted(i for g in grupos for i in g)
    media = caso_base.demanda_media(insumos)
    cargas = [media[g].sum() for g in grupos]
    theta = list(insumos.flota["capacidad"])
    ok = (len(grupos) == len(theta) and todos == list(range(len(insumos.hospitales)))
          and all(c <= q for c, q in zip(cargas, theta)))
    return ok, f"{len(grupos)} zonas, demanda media por zona {[round(c) for c in cargas]} (θ = {theta[0]})"


def prueba_calendario():
    """Cada hospital se visita todos los días con el mismo vehículo; el
    calendario es el mismo todas las semanas y los km calzan con las rutas."""
    insumos = _base()
    horizonte = insumos.parametros["horizonte"]
    delta = modelo_comun.matriz_distancias(insumos)
    estado = estado_inicial_semana(insumos)
    cal = caso_base.resolver_modelo_calendario(None, estado, insumos)
    otro = caso_base.resolver_modelo_calendario(None, estado, insumos)
    problemas = []
    for i in range(len(insumos.hospitales)):
        vehiculos = {k for (t, j, k) in cal.visitas if j == i}
        dias = sorted(t for (t, j, _) in cal.visitas if j == i)
        if len(vehiculos) != 1 or dias != list(range(horizonte)):
            problemas.append(i)
    km = sum(delta[0, r[0] + 1] + sum(delta[a + 1, b + 1] for a, b in zip(r, r[1:])) + delta[r[-1] + 1, 0]
             for r in cal.rutas.values())
    ok = not problemas and cal.visitas == otro.visitas and abs(km - cal.distancia_km) < 1e-6
    return ok, f"{len(cal.visitas)} visitas por semana, {cal.distancia_km:.1f} km"


def prueba_vecino_mas_cercano():
    """La ruta va siempre al hospital más cercano que falta, partiendo del banco."""
    insumos = _base()
    delta = modelo_comun.matriz_distancias(insumos)
    problemas = 0
    for zona in caso_base.zonas(insumos):
        ruta, _ = caso_base.vecino_mas_cercano(zona, delta)
        actual, pendientes = 0, set(zona)
        for i in ruta:
            if delta[actual, i + 1] > min(delta[actual, j + 1] for j in pendientes) + 1e-12:
                problemas += 1
            pendientes.remove(i)
            actual = i + 1
    return problemas == 0, "5 rutas por vecino más cercano"


def prueba_niveles_objetivo():
    """η_i por defecto = percentil 95 directo de la demanda histórica (hacia
    arriba, sin pasar C_i); con la fuente "distribucion", el P95 de la Tabla A.3.
    En la prueba real, el percentil sale solo de 2022–2023. Más percentil, más nivel."""
    insumos = _base()
    nombres = list(insumos.hospitales["nombre"])
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    historia = cargador_datos.cargar_demanda_historica()
    esperado = [min(int(np.ceil(np.quantile(historia[historia["hospital"] == n]["demanda"], 0.95) - 1e-9)), c) for n, c in zip(nombres, C)]
    ok = list(caso_base.niveles_objetivo(insumos, 0.95)) == esperado
    ok &= list(caso_base.niveles_objetivo(insumos, 0.95, "distribucion")) == [min(P95_TABLA_A3[n], c) for n, c in zip(nombres, C)]
    real = evaluacion_historica.preparar_insumos()
    corte = historia[historia["fecha"] < cfg.INICIO_PRUEBA_HISTORICA]
    esperado_real = [min(int(np.ceil(np.quantile(corte[corte["hospital"] == n]["demanda"], 0.95) - 1e-9)), c) for n, c in zip(nombres, C)]
    ok &= list(caso_base.niveles_objetivo(real, 0.95)) == esperado_real
    par = insumos.parametros
    critico = par["costo_shortage"] / (par["costo_shortage"] + par["costo_holding_hospital"])
    ok &= abs(caso_base.valor_cuantil("critico", insumos) - critico) < 1e-12
    ok &= all((caso_base.niveles_objetivo(insumos, x) <= caso_base.niveles_objetivo(insumos, y)).all()
              for x, y in [(0.90, 0.95), (0.95, 0.99), ("critico", 0.90)])
    return ok, (f"P95 histórico suma {sum(esperado)} (Tabla A.3: {sum(min(P95_TABLA_A3[n], c) for n, c in zip(nombres, C))}); "
                f"en la prueba real, solo 2022–2023 ({sum(esperado_real)})")


def prueba_recorte():
    """Recorte proporcional redondeando hacia abajo; si alcanza, no se recorta."""
    ok = list(caso_base.recortar([60, 60, 40], 143)) == [53, 53, 35]
    ok &= list(caso_base.recortar([10, 20], 50)) == [10, 20]
    ok &= list(caso_base.recortar([10, 20], 0)) == [0, 0]
    return ok, "[60, 60, 40] con 143 → [53, 53, 35]"


def prueba_pedido_admision_fifo():
    """Un día a mano: pedido = η − inventario, admisión de cada regla y FIFO."""
    insumos = _base()
    L = insumos.parametros["shelf_life"]
    n_h = len(insumos.hospitales)
    eta = caso_base.niveles_objetivo(insumos, 0.95)
    mu_b, _ = modelo_comun.minimos_terminales(insumos)
    cal = caso_base.resolver_modelo_calendario(None, estado_inicial_semana(insumos), insumos)

    # Hospitales llenos salvo el 0, al que le faltan 5; banco con 3 unidades de edad 4 y 10 de edad 1.
    inv_h = np.zeros((n_h, L), dtype=int)
    inv_h[:, 0] = eta
    inv_h[0, 0] = eta[0] - 5
    inv_b = np.zeros(L, dtype=int)
    inv_b[L - 1], inv_b[1] = 3, 10
    estado = EstadoSistema(dia=2, inventario_banco=inv_b, inventario_hospitales=inv_h)

    todo = caso_base.resolver_modelo_cantidades(_escenario_hoy(insumos, 700), estado, cal, insumos, 0.95, "todo")
    ajustada = caso_base.resolver_modelo_cantidades(_escenario_hoy(insumos, 700), estado, cal, insumos, 0.95, "ajustada")
    k0 = next(k for (t, i, k) in cal.visitas if i == 0 and t == 2)
    esperado_todo = min(700, insumos.capacidad_banco - 13)
    esperado_ajustada = int(np.ceil(5 + mu_b)) - 13
    fifo = {(0, k0, L - 1): 3, (0, k0, 1): 2}
    ok = (todo.oferta_admitida == esperado_todo and ajustada.oferta_admitida == esperado_ajustada
          and todo.despachos == fifo and ajustada.despachos == fifo)
    return ok, (f"admite {todo.oferta_admitida} (todo) y {ajustada.oferta_admitida} (ajustada); "
                f"despacha 3 de edad 4 y 2 de edad 1")


# ---------------------------------------------------------------------------
# Modo con escenarios (§15.A.3)
# ---------------------------------------------------------------------------


def prueba_regla_nivel():
    """Regla 1: el nivel de menor costo promedio es el cuantil π / (π + γ) de
    la demanda de los escenarios, y nunca baja del inventario ni pasa la capacidad."""
    par = _base().parametros
    pi, gamma = par["costo_shortage"], par["costo_holding_hospital"]
    rng = np.random.default_rng(1)
    problemas = 0
    for _ in range(200):
        d = rng.poisson(rng.uniform(5, 50), size=30)
        nivel = caso_base.nivel_hospital(d, 0, 500, gamma, pi)
        # menor S con proporción de escenarios D <= S al menos π / (π + γ)
        esperado = min(x for x in range(501) if np.mean(d <= x) >= pi / (pi + gamma) - 1e-12)
        problemas += nivel != esperado
    ok = problemas == 0
    ok &= caso_base.nivel_hospital(np.full(30, 40), 60, 144, gamma, pi) == 60
    ok &= caso_base.nivel_hospital(np.full(30, 200), 10, 48, gamma, pi) == 48
    return ok, "200 casos al azar: coincide con el cuantil 0,772 de los escenarios"


def prueba_regla_colchon():
    """Regla 3: el colchón de menor costo es el cuantil 1 − h_b / π de lo que
    faltaría mañana; si mañana no falta nada en ningún escenario, es 0."""
    par = _base().parametros
    pi, h_b = par["costo_shortage"], par["costo_holding_banco"]
    rng = np.random.default_rng(2)
    problemas = 0
    for _ in range(200):
        f = np.maximum(0, rng.normal(0, 80, size=30).round()).astype(int)
        b = caso_base.colchon_banco(f, 1200, h_b, pi)
        esperado = min(x for x in range(1201) if np.mean(f <= x) >= 1 - h_b / pi - 1e-12)
        problemas += b != esperado
    ok = problemas == 0 and caso_base.colchon_banco(np.zeros(30, int), 1200, h_b, pi) == 0
    return ok, f"200 casos al azar: coincide con el cuantil {1 - h_b / pi:.3f} de lo que faltaría"


def prueba_regla_reparto():
    """Regla 2: si alcanza, cada hospital recibe su pedido; si no, la unidad va
    al hospital donde más escenarios la usarían, sin pasar θ_k."""
    demandas = np.array([[10, 3], [10, 3], [10, 0], [10, 0]] * 5)  # hospital 0 la usaría siempre
    inventario = np.array([5, 0])
    alcanza = caso_base.repartir_por_probabilidad({0: 4, 1: 2}, inventario, demandas, {0: 0, 1: 1}, {0: 143, 1: 143}, 100)
    escaso = caso_base.repartir_por_probabilidad({0: 4, 1: 2}, inventario, demandas, {0: 0, 1: 1}, {0: 143, 1: 143}, 3)
    vehiculo = caso_base.repartir_por_probabilidad({0: 4, 1: 2}, inventario, demandas, {0: 0, 1: 0}, {0: 2}, 100)
    ok = alcanza == {0: 4, 1: 2} and escaso == {0: 3, 1: 0} and sum(vehiculo.values()) == 2
    return ok, f"alcanza {alcanza}; con 3 unidades {escaso}; con un vehículo de 2 {vehiculo}"


# ---------------------------------------------------------------------------
# Restricciones en semanas completas
# ---------------------------------------------------------------------------


def prueba_restricciones_semanas():
    """Todas las variantes, en una semana de cada mes: cada día respeta la
    capacidad de hospitales, banco y vehículos, la oferta disponible y el
    calendario. Lo despachado sale del banco por edad sin dejarlo negativo."""
    insumos = _base()
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    semanas = cargar_semanas(caso_base.ARCHIVO_SEMANAS, insumos)[::5]
    problemas, dias = [], 0

    variantes = [(c, a, caso_base.politica(c, a), 1) for c, a in cfg.VARIANTES_CASO_BASE]
    variantes.append(("escenarios", "escenarios", caso_base.politica_escenarios(), cfg.N_ESCENARIOS_DEFAULT))
    for cuantil, admision, pol, n_escenarios in variantes:

        def verificar(escenarios, estado, cal, ins, pol=pol, variante=(cuantil, admision)):
            nonlocal dias
            dec = pol["resolver_cantidades"](escenarios, estado, cal, ins)
            banco = estado.inventario_banco.copy()
            banco[0] += dec.oferta_admitida
            llegadas = np.zeros(len(C), dtype=int)
            carga = {}
            for (i, k, a), u in dec.despachos.items():
                banco[a] -= u
                llegadas[i] += u
                carga[k] = carga.get(k, 0) + u
                if (estado.dia, i, k) not in cal.visitas:
                    problemas.append((variante, estado.dia, "despacho fuera del calendario"))
            if dec.oferta_admitida > escenarios.oferta[0, 0]:
                problemas.append((variante, estado.dia, "admite más que la oferta"))
            if estado.inventario_banco.sum() + dec.oferta_admitida > ins.capacidad_banco:
                problemas.append((variante, estado.dia, "capacidad del banco"))
            if (banco < 0).any():
                problemas.append((variante, estado.dia, "banco negativo"))
            if any(c > theta[k] for k, c in carga.items()):
                problemas.append((variante, estado.dia, "capacidad del vehículo"))
            if (estado.inventario_hospitales.sum(axis=1) + llegadas > C).any():
                problemas.append((variante, estado.dia, "capacidad del hospital"))
            dias += 1
            return dec

        for semana in semanas:
            dia = lambda f, s=semana: (pd.Timestamp(f) - s.fecha_lunes).days
            ejecutar_semana(
                semana.fecha_lunes, estado_inicial_semana(insumos),
                observar_oferta=lambda f, s=semana, dia=dia: int(s.oferta[dia(f)]),
                observar_demanda=lambda f, s=semana, dia=dia: s.demanda[dia(f)],
                insumos=insumos, n_escenarios=n_escenarios,
                resolver_calendario=pol["resolver_calendario"], resolver_cantidades=verificar,
            )
    return not problemas, f"{dias} días revisados ({len(variantes)} variantes × 12 semanas × 7 días)" + (
        f"; problemas: {problemas[:3]}" if problemas else "")


def prueba_resultados_guardados():
    """El archivo de resultados tiene las 60 semanas de cada variante y
    coincide con volver a correr una semana."""
    insumos = _base()
    tabla = pd.read_csv(caso_base.ARCHIVO_RESULTADOS)
    n_variantes = len(cfg.VARIANTES_CASO_BASE)
    semanas = cargar_semanas(caso_base.ARCHIVO_SEMANAS, insumos)
    cuantil, admision = cfg.VARIANTES_CASO_BASE[0]
    nueva = caso_base.correr_variantes(insumos, semanas[:1], [(cuantil, admision)])
    guardada = tabla[(tabla["variante"] == caso_base.nombre_variante(cuantil, admision))
                     & (tabla["semana"] == semanas[0].indice)]
    escenarios = caso_base.correr_escenarios(insumos, semanas[:1])
    guardada_esc = tabla[(tabla["variante"] == "escenarios") & (tabla["semana"] == semanas[0].indice)]
    ok = len(tabla) == 60 * (n_variantes + 1)
    ok &= abs(guardada["costo_total"].item() - nueva["costo_total"].item()) < 1e-6
    ok &= abs(guardada_esc["costo_total"].item() - escenarios["costo_total"].item()) < 1e-6
    return ok, f"{len(tabla)} filas ({n_variantes} variantes + escenarios, × 60 semanas)"


# ---------------------------------------------------------------------------
# Prueba con las semanas reales de 2024 (§15.A.4)
# ---------------------------------------------------------------------------


def prueba_sin_datos_de_prueba():
    """La política de la prueba real se ajusta sólo con 2022–2023: la Normal de
    enero tiene la media de los eneros de 2022–2023, y la demanda de cada
    hospital, la media de esos dos años."""
    insumos = evaluacion_historica.preparar_insumos()
    corte = pd.Timestamp(cfg.INICIO_PRUEBA_HISTORICA)
    oferta = cargador_datos.cargar_oferta_historica()
    enero = oferta[(oferta["fecha"].dt.month == 1) & (oferta["fecha"] < corte)]["oferta"].mean()
    demanda = cargador_datos.cargar_demanda_historica()
    media = demanda[demanda["fecha"] < corte].groupby("hospital")["demanda"].mean()
    dif = max(abs(insumos.distribuciones_demanda[n].distribucion.mean() - media[n]) for n in media.index)
    ok = abs(insumos.distribuciones_oferta[1].media - enero) < 1e-9 and dif < 1e-6
    return ok, f"oferta de enero {insumos.distribuciones_oferta[1].media:.1f} = media 2022–2023; demanda por hospital igual a la de 2022–2023"


def prueba_semanas_reales():
    """52 semanas de lunes a domingo de 2024, con la oferta y la demanda tal
    como están en los CSV."""
    insumos = evaluacion_historica.preparar_insumos()
    semanas = evaluacion_historica.semanas_historicas(insumos)
    oferta = cargador_datos.cargar_oferta_historica().set_index("fecha")["oferta"]
    demanda = cargador_datos.cargar_demanda_historica().set_index(["fecha", "hospital"])["demanda"]
    nombres = list(insumos.hospitales["nombre"])
    ok = len(semanas) == 52 and all(s.fecha_lunes.dayofweek == 0 for s in semanas)
    ok &= semanas[0].fecha_lunes == pd.Timestamp("2024-01-01") and semanas[-1].fecha_lunes == pd.Timestamp("2024-12-23")
    for s in (semanas[0], semanas[30], semanas[-1]):
        for t in range(7):
            fecha = s.fecha_lunes + pd.Timedelta(days=t)
            ok &= s.oferta[t] == oferta[fecha]
            ok &= all(s.demanda[t, i] == demanda[fecha, n] for i, n in enumerate(nombres))
    return ok, f"{len(semanas)} semanas, del {semanas[0].fecha_lunes.date()} al {semanas[-1].fecha_lunes.date()}"


def prueba_clasificacion_e_intervalos():
    """Tercios por holgura (cada favorable tiene más holgura que cada neutro, y
    cada neutro más que cada desfavorable) e intervalo t calculado a mano."""
    insumos = evaluacion_historica.preparar_insumos()
    c = evaluacion_historica.clasificar_semanas(evaluacion_historica.semanas_historicas(insumos))
    g = c.groupby("escenario")["holgura"]
    ok = g.min()["favorable"] >= g.max()["neutro"] and g.min()["neutro"] >= g.max()["desfavorable"]
    ok &= sorted(g.size()) == [17, 17, 18]
    # Intervalo a mano: [1, 2, 3, 4, 5] tiene promedio 3 y desviación 1,5811; t(0,975; 4) = 2,7764.
    promedio, inferior, superior, n = evaluacion_historica.intervalo([1, 2, 3, 4, 5], 0.95)
    ok &= abs(promedio - 3) < 1e-12 and abs(superior - (3 + 2.776445 * 1.581139 / 5 ** 0.5)) < 1e-5 and n == 5
    return ok, f"tercios de {dict(g.size())} semanas; intervalo de [1..5]: [{inferior:.3f}, {superior:.3f}]"


def prueba_historico_guardado():
    """El archivo de la prueba real tiene las 9 variantes × 52 semanas y
    coincide con volver a correr una semana."""
    insumos = evaluacion_historica.preparar_insumos()
    semanas = evaluacion_historica.semanas_historicas(insumos)
    tabla = pd.read_csv(evaluacion_historica.ARCHIVO_HISTORICO)
    nueva = caso_base.correr_escenarios(insumos, semanas[20:21])
    guardada = tabla[(tabla["variante"] == "escenarios") & (tabla["semana"] == 20)]
    ok = len(tabla) == 52 * (len(cfg.VARIANTES_CASO_BASE) + 1)
    ok &= abs(guardada["costo_total"].item() - nueva["costo_total"].item()) < 1e-6
    return ok, f"{len(tabla)} filas; la semana 20 se reproduce"


def prueba_comparacion():
    """La comparación con las heurísticas usa las mismas 60 semanas para todas
    las políticas, 20 por escenario, y la diferencia pareada promedio es igual a
    la diferencia de los promedios."""
    from codigo import comparacion_politicas as cp

    insumos = _base()
    politicas = cp.cargar_politicas()
    semanas = cargar_semanas(caso_base.ARCHIVO_SEMANAS, insumos)
    clasificacion = evaluacion_historica.clasificar_semanas(semanas)
    ok = all(list(t.index) == [s.indice for s in semanas] for t in politicas.values())
    ok &= sorted(clasificacion["escenario"].value_counts()) == [20, 20, 20]
    pareada = pd.read_csv(cp.ARCHIVO_PAREADA)
    fila = pareada[(pareada["heuristica"] == "Heurística B (zonas compactas)") & (pareada["caso_base"] == "Caso base agnóstico β 28% (principal)")
                   & (pareada["escenario"] == "todas") & (pareada["kpi"] == "costo_operacional")].iloc[0]
    b, c = politicas["Heurística B (zonas compactas)"], politicas["Caso base agnóstico β 28% (principal)"]
    ok &= abs(fila["diferencia_promedio"] - (b["costo_operacional"].mean() - c["costo_operacional"].mean())) < 1e-6
    return ok, f"{len(politicas)} políticas en las mismas 60 semanas; B − CB agnóstico = {fila['diferencia_promedio']:,.0f}"


# ---------------------------------------------------------------------------
# Caso base OFICIAL: agnóstico (§15)
# ---------------------------------------------------------------------------


def prueba_agnostico_no_usa_demanda():
    """La política agnóstica toma exactamente las mismas decisiones aunque se le
    borren todos los datos de demanda y oferta (historia y distribuciones)."""
    insumos = _base()
    ciego = dataclasses.replace(insumos, distribuciones_demanda={}, distribuciones_oferta={}, demanda_historica=pd.DataFrame())
    estado = estado_inicial_semana(insumos)
    cal, cal_ciego = (cba.resolver_modelo_calendario(None, estado, x) for x in (insumos, ciego))
    ok = cal.visitas == cal_ciego.visitas and cal.rutas == cal_ciego.rutas
    rng = np.random.default_rng(0)
    for _ in range(20):
        inv_h = rng.integers(0, 40, (len(insumos.hospitales), 5)); inv_b = rng.integers(0, 150, 5)
        est = EstadoSistema(dia=int(rng.integers(0, 7)), inventario_banco=inv_b, inventario_hospitales=inv_h)
        esc = _escenario_hoy(insumos, int(rng.integers(300, 900)))
        a = cba.resolver_modelo_cantidades(esc, est, cal, insumos); b = cba.resolver_modelo_cantidades(esc, est, cal_ciego, ciego)
        ok &= a.oferta_admitida == b.oferta_admitida and a.despachos == b.despachos
    return ok, "calendario y 20 días al azar idénticos sin ningún dato de demanda ni oferta"


def prueba_agnostico_reglas():
    """β = capacidad de la flota / capacidad de los hospitales; niveles ⌈β·C⌉; las
    zonas cubren los 22 hospitales una vez, equilibradas por capacidad."""
    insumos = _base()
    beta = insumos.flota["capacidad"].sum() / insumos.hospitales["capacidad_maxima"].sum()
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    zonas = cba.zonas_por_distancia(insumos)
    todos = sorted(i for z in zonas for i in z)
    capacidad = [int(C[z].sum()) for z in zonas]
    ok = abs(cba.valor_beta(insumos) - beta) < 1e-12 and list(cba.niveles(insumos)) == [int(np.ceil(beta * c - 1e-9)) for c in C]
    ok &= len(zonas) == len(insumos.flota) and todos == list(range(len(C)))
    ok &= max(capacidad) - min(capacidad) <= C.max()
    return ok, f"β = {beta:.4f}; niveles grande/mediano/pequeño = {sorted(set(cba.niveles(insumos)), reverse=True)}; capacidad por zona {capacidad}"


def prueba_escenarios_construidos():
    """Favorable y desfavorable = datos reales × factor, redondeados; neutro = reales
    con ruido de a lo más ±1% (más el redondeo); real sin cambios; mismas fechas."""
    insumos = evaluacion_historica.preparar_insumos()
    reales = evaluacion_historica.semanas_historicas(insumos)
    ok = all(a is b for a, b in zip(evaluacion_historica.escenario_construido(reales, "real"), reales))
    for nombre, f in cfg.ESCENARIOS_CONSTRUIDOS.items():
        esc = evaluacion_historica.escenario_construido(reales, nombre)
        ok &= len(esc) == len(reales) and all(e.fecha_lunes == r.fecha_lunes and e.indice == r.indice for e, r in zip(esc, reales))
        for e, r in zip(esc, reales):
            if f["ruido"] == 0:
                ok &= np.array_equal(e.demanda, np.rint(r.demanda * f["demanda"])) and np.array_equal(e.oferta, np.rint(r.oferta * f["oferta"]))
            else:
                ok &= (np.abs(e.demanda - r.demanda) <= f["ruido"] * r.demanda + 0.5).all()
                ok &= (np.abs(e.oferta - r.oferta) <= f["ruido"] * r.oferta + 0.5).all()
    neutro = evaluacion_historica.escenario_construido(reales, "neutro")
    dif = abs(sum(s.demanda.sum() for s in neutro) / sum(s.demanda.sum() for s in reales) - 1)
    return ok, f"3 escenarios × 52 semanas; el neutro difiere de lo real en {100 * dif:.3f}% de la demanda total"


def prueba_agnostico_restricciones():
    """En los 4 escenarios (real, favorable, neutro, desfavorable), cada día: capacidad
    de hospitales, banco y vehículos, oferta disponible y calendario."""
    insumos = evaluacion_historica.preparar_insumos()
    theta = dict(zip(insumos.flota["vehiculo"], insumos.flota["capacidad"]))
    C = insumos.hospitales["capacidad_maxima"].to_numpy()
    reales = evaluacion_historica.semanas_historicas(insumos)[32:44]  # 12 semanas seguidas (agosto a octubre)
    problemas, dias = [], 0
    for nombre in cba.ESCENARIOS:
        semanas = evaluacion_historica.escenario_construido(reales, nombre)
        for beta in cfg.BETAS_SENSIBILIDAD_CASO_BASE:
            pol = cba.politica(beta)

            def verificar(esc, estado, cal, ins, pol=pol, clave=(nombre, beta)):
                nonlocal dias
                d = pol["resolver_cantidades"](esc, estado, cal, ins)
                banco = estado.inventario_banco.copy(); banco[0] += d.oferta_admitida
                llegadas = np.zeros(len(C), dtype=int); carga = {}
                for (i, k, a), u in d.despachos.items():
                    banco[a] -= u; llegadas[i] += u; carga[k] = carga.get(k, 0) + u
                    if (estado.dia, i, k) not in cal.visitas:
                        problemas.append((clave, "fuera del calendario"))
                if d.oferta_admitida > esc.oferta[0, 0] or estado.inventario_banco.sum() + d.oferta_admitida > ins.capacidad_banco:
                    problemas.append((clave, "oferta o capacidad del banco"))
                if (banco < 0).any() or any(c > theta[k] for k, c in carga.items()):
                    problemas.append((clave, "banco negativo o vehículo"))
                if (estado.inventario_hospitales.sum(axis=1) + llegadas > C).any():
                    problemas.append((clave, "capacidad del hospital"))
                dias += 1
                return d

            simular_semanas_encadenadas(semanas, insumos, 1, cba.SEMILLA, resolver_calendario=pol["resolver_calendario"],
                                        resolver_cantidades=verificar)
    return not problemas, f"{dias} días revisados (4 escenarios × 3 β × {len(reales)} semanas)" + (f"; {problemas[:3]}" if problemas else "")


def prueba_agnostico_resultados():
    """Los archivos del caso base agnóstico tienen todas las filas y se reproducen."""
    reales = evaluacion_historica.preparar_insumos()
    tabla = pd.read_csv(cba.ARCHIVO_ESCENARIOS)
    simuladas = pd.read_csv(cba.ARCHIVO_SIMULADAS)
    n_beta = len(cfg.BETAS_SENSIBILIDAD_CASO_BASE)
    ok = len(tabla) == len(cba.ESCENARIOS) * n_beta * 52 and len(simuladas) == n_beta * 60
    semanas = evaluacion_historica.escenario_construido(evaluacion_historica.semanas_historicas(reales), "desfavorable")
    nueva = cba.correr(reales, semanas, encadenadas=True, betas=[None])
    guardada = tabla[(tabla["escenario"] == "desfavorable") & (tabla["variante"] == cba.nombre_variante(reales))]
    ok &= np.allclose(nueva["costo_total"].to_numpy(), guardada["costo_total"].to_numpy())
    return ok, f"{len(tabla)} filas de escenarios y {len(simuladas)} simuladas; el desfavorable se reproduce"


def prueba_comparacion_anio_simulado():
    """En el año simulado, las tres políticas vivieron las mismas 52 semanas con la
    misma demanda, y la diferencia pareada guardada es la diferencia de promedios."""
    from codigo import comparacion_anio_simulado as cas

    tablas = cas.cargar()  # falla si no son las mismas semanas o la misma demanda
    pareada = pd.read_csv(cas.ARCHIVO_PAREADA)
    fila = pareada[(pareada["politica"] == "Heurística B (zonas compactas)") & (pareada["contra"] == "Caso base agnóstico")
                   & (pareada["kpi"] == "costo_operacional")].iloc[0]
    esperado = tablas["Heurística B (zonas compactas)"]["costo_operacional"].mean() - tablas["Caso base agnóstico"]["costo_operacional"].mean()
    ok = len(tablas) == 3 and abs(fila["diferencia_promedio"] - esperado) < 1e-6
    return ok, f"3 políticas, 52 semanas iguales; B − caso base = {fila['diferencia_promedio']:,.0f}"


PRUEBAS = [
    ("OFICIAL agnóstico: no usa demanda ni oferta conocidas", prueba_agnostico_no_usa_demanda),
    ("OFICIAL agnóstico: β, niveles y zonas por capacidad", prueba_agnostico_reglas),
    ("OFICIAL agnóstico: escenarios construidos", prueba_escenarios_construidos),
    ("OFICIAL agnóstico: restricciones en los 4 escenarios", prueba_agnostico_restricciones),
    ("OFICIAL agnóstico: resultados guardados reproducibles", prueba_agnostico_resultados),
    ("Zonas: cubren todo, sin repetir, bajo θ", prueba_zonas),
    ("Calendario diario, fijo y km de las rutas", prueba_calendario),
    ("Rutas por vecino más cercano", prueba_vecino_mas_cercano),
    ("Niveles objetivo: percentil histórico, Tabla A.3 y corte 2022–2023", prueba_niveles_objetivo),
    ("Recorte proporcional hacia abajo", prueba_recorte),
    ("Pedido, admisión y FIFO en un día a mano", prueba_pedido_admision_fifo),
    ("Escenarios, regla 1: nivel de menor costo promedio", prueba_regla_nivel),
    ("Escenarios, regla 3: colchón del banco", prueba_regla_colchon),
    ("Escenarios, regla 2: reparto por probabilidad de uso", prueba_regla_reparto),
    ("Restricciones en semanas completas", prueba_restricciones_semanas),
    ("Resultados guardados reproducibles", prueba_resultados_guardados),
    ("Prueba real: ajuste sólo con 2022–2023", prueba_sin_datos_de_prueba),
    ("Prueba real: 52 semanas de 2024 tal como en los datos", prueba_semanas_reales),
    ("Prueba real: escenarios por tercios e intervalos", prueba_clasificacion_e_intervalos),
    ("Prueba real: resultados guardados reproducibles", prueba_historico_guardado),
    ("Comparación con las heurísticas (pareada)", prueba_comparacion),
    ("Comparación en el año simulado (caso base, A y B)", prueba_comparacion_anio_simulado),
]


def main():
    fallas = 0
    for nombre, prueba in PRUEBAS:
        try:
            ok, detalle = prueba()
        except Exception as error:  # una prueba que se cae cuenta como falla
            ok, detalle = False, f"error: {error!r}"
        fallas += not ok
        print(f"[{'OK   ' if ok else 'FALLA'}] {nombre}: {detalle}")
    print(f"\n{len(PRUEBAS) - fallas} de {len(PRUEBAS)} pruebas OK.")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
