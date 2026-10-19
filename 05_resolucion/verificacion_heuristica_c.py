"""Verificación de la heurística C (codigo/heuristica_c.py).

Se corre desde la raíz del repo con:

    python -m codigo.verificacion_heuristica_c

Imprime OK / FALLA para cada prueba y termina con código de salida 1 si
alguna falla. Las pruebas usan gap 0 (óptimo exacto), así que se pueden
comparar costos.

Revisa que:
1. Sin recortar (un bloque de 7 días, ventana de 7), la C es el modelo común.
2. El inventario inicial por escenario y el paso de un bloque al siguiente
   dan lo que se calcula a mano (intento 1).
3. El mínimo terminal se exige al cierre de la ventana.
4. Con la próxima visita fuera de la ventana, la C despacha menos, como se
   calcula a mano (es el costo de la heurística).
5. En una semana chica completa, con el intento 1 (calendario por bloques y
   ventana) y con el intento 2 (calendario de la B y ventana), se respetan
   rutas, calendario, capacidades y conservación de unidades.

Las pruebas 1, 5, 6 y 8 cubren el intento 2 y las pruebas 2, 3, 4 y 7 cubren
el intento 1.
"""

import sys

import numpy as np
import pandas as pd

from codigo import heuristica_b, heuristica_c, modelo_comun
from codigo import configuracion as cfg
from codigo.escenarios import Escenarios, generar_escenarios
from codigo.horizonte_rodante import actualizar_estado
from codigo.modelo_comun import DecisionesDia, EstadoSistema, SolucionCalendario
from codigo.simulacion import generadores, generar_semanas_reales, simular_semana
from codigo.verificacion import _estado, _iguales, _insumos_chicos, _minimos, _ruta_optima


def _escenarios(insumos, oferta, demanda):
    """Escenarios dados a mano. oferta: lista por escenario de listas por día;
    demanda: lista por escenario de listas por día de listas por hospital."""
    return Escenarios(
        oferta=np.array(oferta, dtype=int),
        demanda=np.array(demanda, dtype=int),
        hospitales=list(insumos.hospitales["nombre"]),
        mes=1,
    )


def _escenarios_al_azar(insumos, dias, n_escenarios, semilla):
    """Escenarios sorteados de las distribuciones reales (enero)."""
    esc = generar_escenarios(dias, 1, insumos.distribuciones_demanda, n_escenarios,
                             distribuciones_oferta=insumos.distribuciones_oferta,
                             rng=np.random.default_rng(semilla))
    columnas = [esc.hospitales.index(n) for n in insumos.hospitales["nombre"]]
    return Escenarios(oferta=esc.oferta, demanda=esc.demanda[:, :, columnas],
                      hospitales=list(insumos.hospitales["nombre"]), mes=1)


# ---------------------------------------------------------------------------
# Sin recortar, la C es el modelo común
# ---------------------------------------------------------------------------


def prueba_cantidades_sin_recortar():
    """Con una ventana igual o mayor que los días que quedan, las decisiones son las del modelo común."""
    ins = _insumos_chicos(["Hospital San José", "Hospital Calvo Mackenna"])
    esc = _escenarios_al_azar(ins, dias=4, n_escenarios=3, semilla=11)
    estado = _estado(ins, dia=3, banco={0: 150, 2: 40}, hospitales={(0, 1): 10, (1, 3): 20})
    calendario = SolucionCalendario(visitas={(3, 0, 0), (4, 1, 0), (5, 0, 0), (5, 1, 0)})
    comun = modelo_comun.resolver_modelo_cantidades(esc, estado, calendario, ins)
    resultados = [heuristica_c.resolver_cantidades_ventana(esc, estado, calendario, ins, ventana) for ventana in (4, 7)]
    ok = all(
        d.oferta_admitida == comun.oferta_admitida and d.despachos == comun.despachos
        and _iguales(d.info_resolucion["valor_objetivo"], comun.info_resolucion["valor_objetivo"])
        for d in resultados
    )
    return ok, (f"modelo común: admite {comun.oferta_admitida}, despacha {comun.despachos}, "
                f"costo {comun.info_resolucion['valor_objetivo']:,.0f}; ventanas 4 y 7: "
                f"{[(d.oferta_admitida, round(d.info_resolucion['valor_objetivo'])) for d in resultados]}")


def prueba_calendario_un_bloque():
    """Con un solo bloque de 7 días, el calendario es el del modelo común (mismo costo y visitas)."""
    ins = _insumos_chicos(["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], n_vehiculos=2)
    esc = _escenarios_al_azar(ins, dias=7, n_escenarios=2, semilla=12)
    estado = _estado(ins, banco={0: 120}, hospitales={(0, 0): 15, (1, 0): 15, (2, 0): 5})
    comun = modelo_comun.resolver_modelo_calendario(esc, estado, ins)
    c = heuristica_c.resolver_calendario_bloques(esc, estado, ins, (7,))
    ok = (_iguales(c.info_resolucion["valor_objetivo"], comun.info_resolucion["valor_objetivo"])
          and c.visitas == comun.visitas and _iguales(c.distancia_km, comun.distancia_km))
    return ok, (f"costo C {c.info_resolucion['valor_objetivo']:,.0f} vs común {comun.info_resolucion['valor_objetivo']:,.0f}; "
                f"{len(c.visitas)} visitas {'iguales' if c.visitas == comun.visitas else 'DISTINTAS'}; "
                f"km {c.distancia_km:.2f} vs {comun.distancia_km:.2f}")


# ---------------------------------------------------------------------------
# Inventario por escenario y paso entre bloques (a mano)
# ---------------------------------------------------------------------------


def prueba_inicio_por_escenario():
    """Cada escenario parte con su propio inventario: costo esperado igual al calculado a mano."""
    ins = _insumos_chicos(["Hospital San José"])  # capacidad 144: μ_i = 14,4
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    L = par["shelf_life"]
    esc = _escenarios(ins, oferta=[[0], [0]], demanda=[[[0]], [[0]]])
    banco = np.zeros((2, L), dtype=int)
    hospitales = np.zeros((2, 1, L), dtype=int)
    hospitales[0, 0, 0] = 20  # escenario 0: 20 unidades en el hospital; escenario 1: nada
    estado = EstadoSistema(dia=6, inventario_banco=None, inventario_hospitales=None)
    solucion, _ = heuristica_c._resolver_bloque(esc, estado, ins, (banco, hospitales))
    s0 = 20 * par["costo_holding_hospital"] + par["costo_shortage"] * mu_b
    s1 = par["costo_shortage"] * (mu_b + mu_i[0])
    esperado = (s0 + s1) / 2
    ok = _iguales(solucion.info_resolucion["valor_objetivo"], esperado) and not solucion.visitas
    return ok, f"costo {solucion.info_resolucion['valor_objetivo']:,.0f} vs a mano {esperado:,.0f} (escenarios {s0:,.0f} y {s1:,.0f})"


def prueba_paso_entre_bloques():
    """El bloque siguiente parte de lo que quedó al cierre en cada escenario, envejecido un día."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    esc = _escenarios(ins, oferta=[[50], [0]], demanda=[[[0]], [[0]]])
    estado = _estado(ins, dia=0, banco={3: 10, 4: 7})
    solucion, (banco, hospitales) = heuristica_c._resolver_bloque(esc, estado, ins, None)
    esperado_banco = [[0, 50, 0, 0, 10], [0, 0, 0, 0, 10]]  # las 7 de edad 4 vencen; las de edad 3 pasan a 4
    # A mano: escenario 0 admite las 50 (holding 450 < déficit 2300); no conviene despachar.
    beta, omega, pi = par["costo_holding_banco"], par["costo_wastage"], par["costo_shortage"]
    s0 = 60 * beta + 7 * omega + pi * (mu_b - 60 + mu_i[0])
    s1 = 10 * beta + 7 * omega + pi * (mu_b - 10 + mu_i[0])
    esperado = (s0 + s1) / 2
    ok = (banco.tolist() == esperado_banco and hospitales.sum() == 0
          and _iguales(solucion.info_resolucion["valor_objetivo"], esperado))
    return ok, (f"inventario siguiente del banco {banco.tolist()} (esperado {esperado_banco}); "
                f"costo {solucion.info_resolucion['valor_objetivo']:,.0f} vs a mano {esperado:,.0f}")


# ---------------------------------------------------------------------------
# Mínimo al cierre de la ventana y próxima visita fuera de ella (a mano)
# ---------------------------------------------------------------------------


def prueba_minimo_al_cierre_de_la_ventana():
    """Con ventana de 3, el mínimo se cobra al cierre del tercer día (o del domingo si queda antes)."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    mu_b, mu_i = _minimos(ins)
    deficit = par["costo_shortage"] * (mu_b - 100 + mu_i[0])
    resultados = []
    for dia, dias_restantes, noches in ((0, 7, 3), (5, 2, 2)):
        esc = _escenarios(ins, oferta=[[0] * dias_restantes], demanda=[[[0]] * dias_restantes])
        dec = heuristica_c.resolver_cantidades_ventana(esc, _estado(ins, dia=dia, banco={0: 100}), SolucionCalendario(), ins, 3)
        esperado = 100 * par["costo_holding_banco"] * noches + deficit
        resultados.append((dia, dec.info_resolucion["valor_objetivo"], esperado))
    ok = all(_iguales(costo, esperado) for _, costo, esperado in resultados)
    return ok, "; ".join(f"parte el día {d}: costo {c:,.0f} vs a mano {e:,.0f}" for d, c, e in resultados)


def prueba_visita_fuera_de_la_ventana():
    """Visitas lunes y viernes, demanda 20 diaria: con ventana 7 se despachan 80 el lunes; con
    ventana 3, 75 (60 + 15 del mínimo del miércoles), y el jueves faltan 5."""
    ins = _insumos_chicos(["Hospital San José"])
    par = ins.parametros
    esc = _escenarios(ins, oferta=[[0] * 7], demanda=[[[20]] * 7])
    estado = _estado(ins, dia=0, banco={0: 400})
    calendario = SolucionCalendario(visitas={(0, 0, 0), (4, 0, 0)})
    despachado, faltante = {}, {}
    for ventana in (7, 3):
        dec = heuristica_c.resolver_cantidades_ventana(esc, estado, calendario, ins, ventana)
        despachado[ventana] = sum(dec.despachos.values())
        # Lunes a jueves en la realidad (sin visitas martes a jueves).
        real, falta = estado, 0
        for dia in range(4):
            decision = dec if dia == 0 else DecisionesDia()
            real, r = actualizar_estado(real, decision, 0, [20], ins)
            falta += r["faltante"].sum()
        faltante[ventana] = falta
    ok = despachado == {7: 80, 3: 75} and faltante == {7: 0, 3: 5}
    return ok, (f"despacho del lunes: ventana 7 da {despachado[7]} (a mano 80), ventana 3 da {despachado[3]} (a mano 75); "
                f"faltantes de lunes a jueves: {faltante[7]} y {faltante[3]} "
                f"(= {faltante[3] * par['costo_shortage']:,.0f} de shortage)")


# ---------------------------------------------------------------------------
# Semana chica completa
# ---------------------------------------------------------------------------


def _revisar_semana_chica(resolver_calendario):
    """Semana completa (3 hospitales, 2 vehículos, 3 escenarios) con las cantidades de la C y el
    calendario dado. Revisa rutas óptimas y km, despachos sólo según calendario, capacidades y
    conservación de unidades.

    Devuelve: (lista de problemas, insumos, calendario, resumen de la semana, unidades conservadas).
    """
    ins = _insumos_chicos(["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], n_vehiculos=2)
    rng_real, rng_modelo = generadores(5)
    semana = generar_semanas_reales(pd.Timestamp("2024-09-02"), 1, ins, rng_real)[0]
    C_i = ins.hospitales["capacidad_maxima"].to_numpy()
    theta = dict(zip(ins.flota["vehiculo"], ins.flota["capacidad"]))
    problemas = []

    def cantidades_revisadas(escenarios, estado, calendario, insumos):
        if escenarios.oferta[0, 0] != semana.oferta[estado.dia]:
            problemas.append(f"día {estado.dia}: la oferta de hoy no es la real")
        dec = heuristica_c.resolver_modelo_cantidades(escenarios, estado, calendario, insumos)
        llegada = np.zeros(len(C_i), dtype=int)
        carga = {}
        for (i, k, a), u in dec.despachos.items():
            llegada[i] += u
            carga[k] = carga.get(k, 0) + u
            if (estado.dia, i, k) not in calendario.visitas:
                problemas.append(f"día {estado.dia}: despacho a {i} con vehículo {k} sin visita")
        if any(c > theta[k] for k, c in carga.items()):
            problemas.append(f"día {estado.dia}: vehículo sobre capacidad")
        if ((estado.inventario_hospitales.sum(axis=1) + llegada) > C_i).any():
            problemas.append(f"día {estado.dia}: hospital sobre capacidad")
        if estado.inventario_banco.sum() + dec.oferta_admitida > insumos.capacidad_banco:
            problemas.append(f"día {estado.dia}: banco sobre capacidad")
        return dec

    resumen, cal, regs = simular_semana(semana, ins, n_escenarios=3, rng=rng_modelo,
                                        resolver_calendario=resolver_calendario,
                                        resolver_cantidades=cantidades_revisadas)

    if {t for (t, _, _) in cal.visitas} - set(range(7)):
        problemas.append("visitas fuera de la semana")

    delta = modelo_comun.matriz_distancias(ins)  # posición 0 = banco, i + 1 = hospital i
    km = 0.0
    for (t, k), ruta in cal.rutas.items():
        if set(ruta) != {i for (dia, i, kk) in cal.visitas if dia == t and kk == k}:
            problemas.append(f"ruta {ruta} no calza con las visitas del día {t}")
        nodos = [0] + [i + 1 for i in ruta] + [0]
        largo = sum(delta[m, n] for m, n in zip(nodos[:-1], nodos[1:]))
        km += largo
        if not _iguales(largo, _ruta_optima(delta, tuple(i + 1 for i in ruta))):
            problemas.append(f"ruta {ruta} del día {t} no es la más corta")
    if not _iguales(km, cal.distancia_km):
        problemas.append("km del calendario no calzan con las rutas")

    banco0, hosp0 = modelo_comun.inventario_inicial(ins)
    entra = banco0.sum() + hosp0.sum() + sum(r["admitida"] for r in regs)
    consumido = semana.demanda.sum() - sum(r["faltante"].sum() for r in regs)
    vencidas = sum(r["vencidas_banco"] + r["vencidas_hospitales"].sum() for r in regs)
    final = regs[-1]["estado_final"]
    sale = consumido + vencidas + final.inventario_banco.sum() + final.inventario_hospitales.sum()
    if entra != sale:
        problemas.append(f"conservación: entran {entra}, salen {sale}")
    return problemas, ins, cal, resumen, entra


def prueba_semana_chica():
    """Intento 1: semana chica completa con el calendario por bloques; además, dos bloques óptimos."""
    problemas, ins, cal, resumen, entra = _revisar_semana_chica(heuristica_c.resolver_modelo_calendario_bloques)
    bloques = cal.info_resolucion["bloques"]
    if len(bloques) != len(cfg.BLOQUES_CALENDARIO_C):
        problemas.append(f"{len(bloques)} bloques en vez de {len(cfg.BLOQUES_CALENDARIO_C)}")
    if not cal.info_resolucion["optimo"]:
        problemas.append("algún bloque no llegó al óptimo")

    dias_visitas = sorted({t for (t, _, _) in cal.visitas})
    detalle = (f"{len(bloques)} bloques óptimos; días con visitas {dias_visitas}; {len(cal.rutas)} rutas óptimas; "
               f"{entra} unidades conservadas; costo real {resumen['costo_total']:,.0f}"
               if not problemas else "; ".join(problemas))
    return not problemas, detalle


def prueba_semana_chica_intento_2():
    """Intento 2: semana chica completa con el calendario de la B; además, calendario óptimo y
    cada vehículo visita sólo hospitales de su zona."""
    problemas, ins, cal, resumen, entra = _revisar_semana_chica(heuristica_b.resolver_modelo_calendario)
    if not cal.info_resolucion["optimo"]:
        problemas.append("el calendario de la B no llegó al óptimo")
    zonas = heuristica_b._zonas_por_vehiculo(heuristica_b.particion_compacta(ins), ins)
    fuera = sorted((t, i, k) for (t, i, k) in cal.visitas if i not in zonas[k])
    if fuera:
        problemas.append(f"visitas fuera de la zona del vehículo: {fuera}")

    dias_visitas = sorted({t for (t, _, _) in cal.visitas})
    detalle = (f"calendario óptimo; zonas {list(zonas.values())} respetadas; días con visitas {dias_visitas}; "
               f"{len(cal.rutas)} rutas óptimas; {entra} unidades conservadas; "
               f"costo real {resumen['costo_total']:,.0f}"
               if not problemas else "; ".join(problemas))
    return not problemas, detalle


PRUEBAS = [
    ("Cantidades sin recortar = modelo común", prueba_cantidades_sin_recortar),
    ("Calendario de un bloque = modelo común", prueba_calendario_un_bloque),
    ("Inventario inicial por escenario (a mano)", prueba_inicio_por_escenario),
    ("Paso del inventario entre bloques (a mano)", prueba_paso_entre_bloques),
    ("Mínimo al cierre de la ventana (a mano)", prueba_minimo_al_cierre_de_la_ventana),
    ("Próxima visita fuera de la ventana (a mano)", prueba_visita_fuera_de_la_ventana),
    ("Semana chica completa, intento 1 (calendario por bloques)", prueba_semana_chica),
    ("Semana chica completa, intento 2 (calendario de la B)", prueba_semana_chica_intento_2),
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
