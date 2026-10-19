"""Verificación del modelo común (METODOLOGIA.md §12).

Se corre desde la raíz del repo con:

    python -m codigo.verificacion

Imprime OK / FALLA para cada prueba y termina con código de salida 1 si
alguna falla. Hay que correrlo cada vez que alguien cambie el modelo, la
dinámica de inventario o la carga de datos.

Tres niveles:
1. Datos y distribuciones: lo que entra al modelo es lo que dicen los datos.
2. Instancias chicas con resultado calculado a mano: el modelo de optimización
   hace exactamente lo que acordamos (capacidad, vencimiento, holding,
   terminal, visitas, edades, admisión).
3. Plan versus realidad: con un solo escenario igual a la realidad, el costo
   que planifica el modelo es el mismo que cobra la simulación; además se
   revisan rutas, calendario, capacidades y conservación de unidades.

Las pruebas usan gap 0 (óptimo exacto), así que se pueden comparar costos.
"""

import itertools
import math
import sys

import numpy as np
import pandas as pd

from codigo import cargador_datos, modelo_comun
from codigo import configuracion as cfg
from codigo.escenarios import Escenarios, generar_escenarios
from codigo.horizonte_rodante import Insumos, actualizar_estado, estado_inicial_semana, preparar_insumos
from codigo.modelo_comun import DecisionesDia, EstadoSistema, SolucionCalendario
from codigo.simulacion import generadores, generar_semanas_reales, simular_semana

# P95 del modelo ajustado por hospital, Tabla A.3 del Informe 1.
P95_TABLA_A3 = {
    "Instituto Traumatológico": 42, "Hospital Calvo Mackenna": 83, "Hospital San José": 51,
    "Hospital Roberto del Río": 63, "Hospital del Salvador": 48, "Hospital San Juan de Dios": 57,
    "Hospital Barros Luco": 42, "Ex Posta Central": 47, "Hospital Félix Bulnes": 44,
    "Hospital San Borja Arriarán": 53, "Hospital Luis Tisné": 22, "Hospital Padre Hurtado": 55,
    "Hospital Parroquial San Bernardo": 31, "Hospital Lucio Córdova": 13,
    "Hospital San José de Maipo": 15, "Hospital Exequiel González Cortés": 62,
    "Hospital Sótero del Río": 64, "Hospital El Pino": 42, "Instituto Geriatría": 10,
    "Instituto Nacional del Cáncer": 20, "Instituto Neurocirugía": 17, "Instituto Psiquiátrico": 36,
}

_BASE = None


def _base():
    """Insumos reales (se cargan una sola vez)."""
    global _BASE
    if _BASE is None:
        _BASE = preparar_insumos()
    return _BASE


def _insumos_chicos(nombres, n_vehiculos=1):
    """Insumos reales restringidos a algunos hospitales y vehículos."""
    base = _base()
    hospitales = base.hospitales[base.hospitales["nombre"].isin(nombres)].reset_index(drop=True)
    return Insumos(
        parametros=base.parametros,
        hospitales=hospitales,
        nombre_banco=base.nombre_banco,
        capacidad_banco=base.capacidad_banco,
        flota=base.flota.head(n_vehiculos).reset_index(drop=True),
        distancias=base.distancias,
        distribuciones_demanda={n: base.distribuciones_demanda[n] for n in hospitales["nombre"]},
        distribuciones_oferta=base.distribuciones_oferta,
    )


def _estado(insumos, dia=0, banco=None, hospitales=None):
    """EstadoSistema con inventario dado como {edad: unidades} y {(i, edad): unidades}."""
    L = insumos.parametros["shelf_life"]
    inv_b = np.zeros(L, dtype=int)
    inv_h = np.zeros((len(insumos.hospitales), L), dtype=int)
    for a, u in (banco or {}).items():
        inv_b[a] = u
    for (i, a), u in (hospitales or {}).items():
        inv_h[i, a] = u
    return EstadoSistema(dia=dia, inventario_banco=inv_b, inventario_hospitales=inv_h)


def _cantidades(insumos, estado, oferta, demanda, visitas):
    """Resuelve el modelo de cantidades con UN escenario conocido.
    oferta: lista por día; demanda: lista por día de listas por hospital."""
    escenarios = Escenarios(
        oferta=np.array([oferta], dtype=int),
        demanda=np.array([demanda], dtype=int),
        hospitales=list(insumos.hospitales["nombre"]),
        mes=1,
    )
    return modelo_comun.resolver_modelo_cantidades(escenarios, estado, SolucionCalendario(visitas=set(visitas)), insumos)


def _iguales(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b)) + 1e-6


def _minimos(insumos):
    return modelo_comun.minimos_terminales(insumos)


# ---------------------------------------------------------------------------
# Nivel 1: datos y distribuciones
# ---------------------------------------------------------------------------


def prueba_parametros():
    """Los parámetros, capacidades, flota y distancias cargados coinciden con los CSV."""
    par = cargador_datos.cargar_parametros()
    crudo = pd.read_csv(cfg.RUTA_DATOS / "parametros.csv", encoding=cfg.CODIFICACION_ARCHIVOS)
    errores = [n for n, v in zip(crudo["parametro"], crudo["valor"]) if par[n] != float(v)]
    capacidades = pd.read_csv(cfg.RUTA_DATOS / "capacidades_hospitales.csv", encoding=cfg.CODIFICACION_ARCHIVOS, index_col=0)
    hosp = cargador_datos.tabla_hospitales()
    if list(hosp["capacidad_maxima"]) != list(capacidades["capacidad_maxima"].iloc[1:]):
        errores.append("capacidades de hospitales")
    if cargador_datos.capacidad_banco() != int(capacidades["capacidad_maxima"].iloc[0]) or cargador_datos.capacidad_banco() != 1200:
        errores.append("capacidad del banco (debe ser la fila 0 de capacidades_hospitales.csv, 1200)")
    if len(hosp) != 22:
        errores.append("número de hospitales")
    flota = cargador_datos.cargar_flota()
    if len(flota) != 5 or set(flota["capacidad"]) != {143}:
        errores.append("flota")
    delta = cargador_datos.cargar_distancias()
    if delta.shape != (23, 23) or not (np.diag(delta.to_numpy()) == 0).all():
        errores.append("matriz de distancias")
    if not _iguales(delta.loc["Centro Metropolitano de Sangre", "Hospital del Salvador"], 0.34):
        errores.append("distancia Centro–Salvador (Tabla A.4: 0.34)")
    if list(delta.index[1:]) != list(hosp["nombre"]):
        errores.append("orden de la matriz distinto al de hospitales")
    return not errores, "todo coincide" if not errores else f"diferencias en: {errores}"


def prueba_inventario_inicial():
    """Inventario inicial = 10% de la capacidad redondeado hacia arriba, edad 0."""
    ins = _base()
    banco, hosp = modelo_comun.inventario_inicial(ins)
    cap = ins.hospitales["capacidad_maxima"].to_numpy()
    esperado_h = np.ceil(0.1 * cap).astype(int)
    ok = banco[0] == math.ceil(0.1 * ins.capacidad_banco) == 120 and banco[1:].sum() == 0 and (hosp[:, 0] == esperado_h).all() and hosp[:, 1:].sum() == 0
    return ok, f"banco {banco[0]}; hospitales por capacidad: {dict(zip(cap, hosp[:, 0]))}"


def prueba_distribuciones_demanda():
    """Familias 15/6/1 y P95 del ajuste iguales a la Tabla A.3 para los 22 hospitales."""
    dd = _base().distribuciones_demanda
    conteo = pd.Series([d.tipo for d in dd.values()]).value_counts().to_dict()
    distintos = {h: (int(d.distribucion.ppf(0.95)), P95_TABLA_A3[h]) for h, d in dd.items()
                 if int(d.distribucion.ppf(0.95)) != P95_TABLA_A3[h]}
    ok = conteo == {"binomial_negativa": 15, "binomial": 6, "poisson": 1} and not distintos
    return ok, f"familias {conteo}; P95 distintos a la Tabla A.3: {distintos or 'ninguno'}"


def prueba_distribuciones_oferta():
    """La Normal de cada mes tiene la media histórica del mes; los escenarios la reproducen."""
    ins = _base()
    oferta = cargador_datos.cargar_oferta_historica()
    medias = oferta.groupby(oferta["fecha"].dt.month)["oferta"].mean()
    ok = all(_iguales(ins.distribuciones_oferta[m].media, medias[m]) for m in range(1, 13))
    esc = generar_escenarios(7, 4, ins.distribuciones_demanda, 5000,
                             distribuciones_oferta=ins.distribuciones_oferta, oferta_hoy=700,
                             rng=np.random.default_rng(0))
    d = ins.distribuciones_oferta[4]
    resto = esc.oferta[:, 1:]
    ok &= (esc.oferta[:, 0] == 700).all()
    ok &= abs(resto.mean() - d.media) < 3 and abs(resto.std() - d.desviacion) < 3
    ok &= abs(np.corrcoef(resto[:, 0], resto[:, 1])[0, 1]) < 0.05
    return ok, (f"abril: media simulada {resto.mean():.1f} vs {d.media:.1f}, "
                f"desviación {resto.std():.1f} vs {d.desviacion:.1f}; oferta de hoy fija; días independientes")


def prueba_demanda_escenarios():
    """La demanda simulada por hospital tiene la media histórica de cada hospital."""
    ins = _base()
    historica = cargador_datos.cargar_demanda_historica().groupby("hospital")["demanda"].mean()
    esc = generar_escenarios(7, 1, ins.distribuciones_demanda, 3000,
                             distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(1))
    medias = esc.demanda.mean(axis=(0, 1))
    peor = max(abs(medias[j] - historica[h]) / historica[h] for j, h in enumerate(esc.hospitales))
    return peor < 0.02, f"máxima diferencia relativa de media: {100 * peor:.2f}%"


# ---------------------------------------------------------------------------
# Nivel 2: instancias chicas con resultado calculado a mano
# ---------------------------------------------------------------------------


def prueba_sin_oferta():
    """Sin oferta ni inventario: toda la demanda es faltante y se paga el mínimo terminal completo."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    dec = _cantidades(ins, _estado(ins, dia=5), [0, 0], [[10], [20]], {(5, 0, 0), (6, 0, 0)})
    esperado = par["costo_shortage"] * (30 + mu_b + mu_i[0])
    ok = _iguales(dec.info_resolucion["valor_objetivo"], esperado) and dec.oferta_admitida == 0 and not dec.despachos
    return ok, f"costo {dec.info_resolucion['valor_objetivo']:,.0f} vs a mano {esperado:,.0f}"


def prueba_vencimiento():
    """Unidades de edad 4: se pueden usar hoy; las que sobran se botan y no pagan holding."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    dec = _cantidades(ins, _estado(ins, dia=6, hospitales={(0, 4): 8}), [0], [[3]], set())
    esperado = 5 * par["costo_wastage"] + par["costo_shortage"] * (mu_b + mu_i[0])
    ok = _iguales(dec.info_resolucion["valor_objetivo"], esperado)
    return ok, f"costo {dec.info_resolucion['valor_objetivo']:,.0f} vs a mano {esperado:,.0f} (3 usadas, 5 vencidas, 0 faltantes)"


def prueba_holding_y_terminal():
    """Holding sobre lo que pasa la noche; las unidades que pasan cuentan para el mínimo terminal."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    resultados = []
    for edad in (0, 3):  # edad 3 todavía pasa a la semana siguiente
        dec = _cantidades(ins, _estado(ins, dia=6, hospitales={(0, edad): 20}), [0], [[5]], set())
        esperado = 15 * par["costo_holding_hospital"] + par["costo_shortage"] * (mu_b + max(0.0, mu_i[0] - 15))
        resultados.append((dec.info_resolucion["valor_objetivo"], esperado))
    ok = all(_iguales(a, b) for a, b in resultados)
    return ok, f"edad 0 y 3: costos {[f'{a:,.0f}' for a, _ in resultados]} vs a mano {resultados[0][1]:,.0f}"


def prueba_capacidad_hospital():
    """Capacidad = inventario con que parte el día + lo que llega, aunque se consuma ese día."""
    ins = _insumos_chicos(["Instituto Geriatría"])  # capacidad 48
    lleno = _cantidades(ins, _estado(ins, dia=6, banco={0: 50}, hospitales={(0, 1): 48}), [0], [[100]], {(6, 0, 0)})
    casi = _cantidades(ins, _estado(ins, dia=6, banco={0: 50}, hospitales={(0, 1): 40}), [0], [[100]], {(6, 0, 0)})
    llega = sum(casi.despachos.values())
    ok = not lleno.despachos and llega == 8
    return ok, f"lleno (48/48): despacha {sum(lleno.despachos.values())}; con 40/48: despacha {llega} (máximo 8)"


def prueba_capacidad_banco():
    """Capacidad del banco = inventario con que parte el día + oferta admitida (aunque venza ese día)."""
    ins = _insumos_chicos(["Hospital San José"])
    lleno = ins.capacidad_banco - 50
    dec = _cantidades(ins, _estado(ins, dia=6, banco={4: lleno}), [200], [[0]], set())
    return dec.oferta_admitida == 50, (f"banco de {ins.capacidad_banco} con {lleno} (edad 4) y oferta 200: "
                                       f"admite {dec.oferta_admitida} (máximo 50)")


def prueba_sin_visita():
    """Si el calendario no visita al hospital ese día, no recibe nada."""
    ins = _insumos_chicos(["Hospital San José"])
    sin = _cantidades(ins, _estado(ins, dia=6, banco={0: 50}), [0], [[30]], set())
    con = _cantidades(ins, _estado(ins, dia=6, banco={0: 50}), [0], [[30]], {(6, 0, 0)})
    ok = not sin.despachos and sum(con.despachos.values()) >= 30
    return ok, f"sin visita despacha {sum(sin.despachos.values())}; con visita despacha {sum(con.despachos.values())}"


def prueba_despacho_misma_edad():
    """El despacho llega el mismo día, con la misma edad que tenía en el banco."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    dec = _cantidades(ins, _estado(ins, dia=6, banco={2: 10}), [0], [[10]], {(6, 0, 0)})
    esperado = par["costo_shortage"] * (mu_b + mu_i[0])
    ok = dec.despachos == {(0, 0, 2): 10} and _iguales(dec.info_resolucion["valor_objetivo"], esperado)
    return ok, f"despachos {dec.despachos} (10 unidades de edad 2 para la demanda de hoy, sin faltantes)"


def prueba_capacidad_vehiculo():
    """Un vehículo no lleva más de su capacidad, aunque haya demanda y stock."""
    ins = _insumos_chicos(["Hospital San José", "Hospital Roberto del Río"])
    dec = _cantidades(ins, _estado(ins, dia=6, banco={0: 400}), [0], [[100, 100]], {(6, 0, 0), (6, 1, 0)})
    carga = sum(dec.despachos.values())
    return carga == 143, f"demanda 200, stock 400, 1 vehículo: carga {carga} (capacidad 143)"


def prueba_fifo_real():
    """La actualización real consume con FIFO, envejece y bota lo de edad 4 (ejemplo de METODOLOGIA §11.4)."""
    ins = _insumos_chicos(["Hospital San José"])
    est = _estado(ins, dia=1, banco={1: 30}, hospitales={(0, 3): 10, (0, 0): 20})
    dec = DecisionesDia(oferta_admitida=0, despachos={(0, 0, 1): 30})
    manana, r = actualizar_estado(est, dec, 0, [25], ins)
    esperado = [0, 20, 15, 0, 0]  # 20 de edad 0 → 1; 15 de edad 1 → 2
    ok = list(manana.inventario_hospitales[0]) == esperado and r["faltante"].sum() == 0 and r["vencidas_hospitales"].sum() == 0
    return ok, f"inventario de mañana por edad {list(manana.inventario_hospitales[0])} (esperado {esperado})"


# ---------------------------------------------------------------------------
# Nivel 3: plan versus realidad, rutas y conservación
# ---------------------------------------------------------------------------


def _ruta_optima(delta, nodos):
    """Largo de la ruta más corta banco → nodos → banco (fuerza bruta)."""
    largo = lambda orden: sum(delta[m, n] for m, n in zip((0,) + orden, orden + (0,)))
    return min(largo(orden) for orden in itertools.permutations(nodos))


def prueba_plan_vs_realidad():
    """Semana completa en una instancia chica (3 hospitales, 2 vehículos) donde el único escenario
    es la realidad. Revisa: costo planificado = costo real, oferta del lunes conocida, mes del
    lunes, rutas óptimas y km, despachos sólo según calendario, capacidades y conservación."""
    ins = _insumos_chicos(["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], n_vehiculos=2)
    fecha = pd.Timestamp("2024-09-30")  # lunes de septiembre; la semana termina en octubre
    rng_real, rng_modelo = generadores(5)
    semana = generar_semanas_reales(fecha, 1, ins, rng_real)[0]
    nombres = list(ins.hospitales["nombre"])
    C_i = ins.hospitales["capacidad_maxima"].to_numpy()
    theta = dict(zip(ins.flota["vehiculo"], ins.flota["capacidad"]))
    problemas = []

    def realidad_desde(dia):
        return Escenarios(oferta=semana.oferta[None, dia:].copy(), demanda=semana.demanda[None, dia:].copy(),
                          hospitales=nombres, mes=fecha.month)

    def calendario_determinista(escenarios, estado, insumos):
        if escenarios.oferta[0, 0] != semana.oferta[0]:
            problemas.append("el calendario no recibió la oferta real del lunes")
        if escenarios.mes != 9:
            problemas.append(f"mes {escenarios.mes} en vez del mes del lunes (9)")
        return modelo_comun.resolver_modelo_calendario(realidad_desde(0), estado, insumos)

    def cantidades_determinista(escenarios, estado, calendario, insumos):
        if escenarios.oferta[0, 0] != semana.oferta[estado.dia]:
            problemas.append(f"día {estado.dia}: la oferta de hoy no es la real")
        dec = modelo_comun.resolver_modelo_cantidades(realidad_desde(estado.dia), estado, calendario, insumos)
        llegada = np.zeros(len(nombres), dtype=int)
        carga = {}
        salida_por_edad = np.zeros_like(estado.inventario_banco)
        for (i, k, a), u in dec.despachos.items():
            llegada[i] += u
            carga[k] = carga.get(k, 0) + u
            salida_por_edad[a] += u
            if (estado.dia, i, k) not in calendario.visitas:
                problemas.append(f"día {estado.dia}: despacho a {i} con vehículo {k} sin visita")
        disponible = estado.inventario_banco.copy()
        disponible[0] += dec.oferta_admitida  # lo admitido hoy entra con edad 0
        if (salida_por_edad > disponible).any():
            problemas.append(f"día {estado.dia}: se despachan más unidades de alguna edad de las que hay en el banco")
        if any(c > theta[k] for k, c in carga.items()):
            problemas.append(f"día {estado.dia}: vehículo sobre capacidad")
        if ((estado.inventario_hospitales.sum(axis=1) + llegada) > C_i).any():
            problemas.append(f"día {estado.dia}: hospital sobre capacidad")
        if estado.inventario_banco.sum() + dec.oferta_admitida > insumos.capacidad_banco:
            problemas.append(f"día {estado.dia}: banco sobre capacidad")
        return dec

    resumen, cal, regs = simular_semana(semana, ins, n_escenarios=1, rng=rng_modelo,
                                        resolver_calendario=calendario_determinista,
                                        resolver_cantidades=cantidades_determinista)

    # Rutas: una por vehículo y día, consistentes con las visitas, óptimas y con los km correctos.
    delta = modelo_comun.matriz_distancias(ins)  # posición 0 = banco, i + 1 = hospital i
    km = 0.0
    for (t, k), ruta in cal.rutas.items():
        if len(set(ruta)) != len(ruta):
            problemas.append(f"ruta con hospitales repetidos {ruta}")
        if set(ruta) != {i for (dia, i, kk) in cal.visitas if dia == t and kk == k}:
            problemas.append(f"ruta {ruta} no calza con las visitas del día {t}")
        nodos = [0] + [i + 1 for i in ruta] + [0]
        largo = sum(delta[m, n] for m, n in zip(nodos[:-1], nodos[1:]))
        km += largo
        if not _iguales(largo, _ruta_optima(delta, tuple(i + 1 for i in ruta))):
            problemas.append(f"ruta {ruta} del día {t} no es la más corta")
    if not _iguales(km, cal.distancia_km):
        problemas.append("km del calendario no calzan con las rutas")

    # Conservación de unidades.
    banco0, hosp0 = modelo_comun.inventario_inicial(ins)
    entra = banco0.sum() + hosp0.sum() + sum(r["admitida"] for r in regs)
    consumido = semana.demanda.sum() - sum(r["faltante"].sum() for r in regs)
    vencidas = sum(r["vencidas_banco"] + r["vencidas_hospitales"].sum() for r in regs)
    final = regs[-1]["estado_final"]
    sale = consumido + vencidas + final.inventario_banco.sum() + final.inventario_hospitales.sum()
    if entra != sale:
        problemas.append(f"conservación: entran {entra}, salen {sale}")

    plan, real = cal.info_resolucion["valor_objetivo"], resumen["costo_total"]
    if not _iguales(plan, real, tol=1e-5):
        problemas.append(f"costo planificado {plan:,.2f} ≠ costo real {real:,.2f}")
    detalle = (f"costo planificado {plan:,.0f} = real {real:,.0f}; {len(cal.rutas)} rutas óptimas; "
               f"{entra} unidades conservadas" if not problemas else "; ".join(problemas))
    return not problemas, detalle


PRUEBAS = [
    ("1. Parámetros, capacidades, flota y distancias", prueba_parametros),
    ("1. Inventario inicial (10%, hacia arriba, edad 0)", prueba_inventario_inicial),
    ("1. Distribuciones de demanda (Tabla A.3)", prueba_distribuciones_demanda),
    ("1. Distribuciones de oferta y escenarios", prueba_distribuciones_oferta),
    ("1. Demanda simulada por hospital", prueba_demanda_escenarios),
    ("2. Sin oferta: todo faltante + terminal", prueba_sin_oferta),
    ("2. Vencimiento a los 5 días, sin holding", prueba_vencimiento),
    ("2. Holding y mínimo terminal", prueba_holding_y_terminal),
    ("2. Capacidad del hospital (parte + llega)", prueba_capacidad_hospital),
    ("2. Capacidad del banco (parte + admitida)", prueba_capacidad_banco),
    ("2. Sin visita no hay despacho", prueba_sin_visita),
    ("2. Despacho mismo día y misma edad", prueba_despacho_misma_edad),
    ("2. Capacidad del vehículo", prueba_capacidad_vehiculo),
    ("2. FIFO en la actualización real", prueba_fifo_real),
    ("3. Plan vs realidad, rutas y conservación", prueba_plan_vs_realidad),
]


def main():
    cfg.GAP_MIP = 0.0
    cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
    cfg.TIEMPO_LIMITE_CANTIDADES_S = 120
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
