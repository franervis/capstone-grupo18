"""Orquestador del ciclo semanal con horizonte rodante.

Dos modelos operan en momentos distintos:

1. Modelo de calendario: se resuelve UNA vez, el lunes, y fija qué hospitales
   visita cada vehículo cada día. Queda fijo los 7 días.
2. Modelo de cantidades: se resuelve TODOS los días (lunes incluido) sobre los
   días restantes, con el calendario fijo y el inventario real como estado
   inicial. Sólo se implementan las decisiones de hoy.

Cada día se revela la "realidad" en el mismo orden en que llegaría: la oferta
al inicio del día (antes de decidir) y la demanda después de despachar. Con
eso, `actualizar_estado` calcula el inventario real del día siguiente.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from codigo import cargador_datos, distribuciones, modelo_comun
from codigo import configuracion as cfg
from codigo.escenarios import generar_escenarios
from codigo.modelo_comun import DecisionesDia, EstadoSistema, SolucionCalendario  # noqa: F401 (se reexportan)


@dataclass
class Insumos:
    """Todo lo que los modelos necesitan y no cambia durante la semana.

    Atributos:
        parametros: diccionario de `datos/parametros.csv`.
        hospitales: DataFrame de `cargador_datos.tabla_hospitales()` (define
            el orden de los hospitales en todo el código).
        nombre_banco: nombre del Centro Metropolitano en `hospitales.csv`.
        capacidad_banco: capacidad de almacenamiento del banco (C_b): fila 0
            de `capacidades_hospitales.csv` (1200). No se usa
            `capacidad_banco` de `parametros.csv` (1000), por decisión del grupo.
        flota: DataFrame de `cargador_datos.cargar_flota()`.
        distancias: matriz de `cargador_datos.cargar_distancias()`.
        distribuciones_demanda: {hospital: DistribucionDemanda}.
        distribuciones_oferta: {mes: DistribucionOfertaMensual}.
        demanda_historica: la demanda diaria histórica con que se ajustaron las
            distribuciones (de `cargador_datos.cargar_demanda_historica`). La usa
            el caso base para sus percentiles; None = no se guardó.
    """

    parametros: dict
    hospitales: pd.DataFrame
    nombre_banco: str
    capacidad_banco: int
    flota: pd.DataFrame
    distancias: pd.DataFrame
    distribuciones_demanda: dict
    distribuciones_oferta: dict
    demanda_historica: pd.DataFrame = None


def preparar_insumos():
    """Carga los datos y ajusta las distribuciones de oferta y demanda.

    Devuelve: un objeto Insumos.
    """
    nodos = cargador_datos.cargar_hospitales()
    demanda = cargador_datos.cargar_demanda_historica()
    return Insumos(
        parametros=cargador_datos.cargar_parametros(),
        hospitales=cargador_datos.tabla_hospitales(),
        nombre_banco=nodos.loc[nodos["tipo"] == "Banco", "nombre"].item(),
        capacidad_banco=cargador_datos.capacidad_banco(),
        flota=cargador_datos.cargar_flota(),
        distancias=cargador_datos.cargar_distancias(),
        distribuciones_demanda=distribuciones.ajustar_distribuciones_demanda(demanda),
        distribuciones_oferta=distribuciones.ajustar_distribuciones_oferta(
            cargador_datos.cargar_oferta_historica()
        ),
        demanda_historica=demanda,
    )


def estado_inicial_semana(insumos):
    """EstadoSistema al inicio del lunes con el inventario inicial supuesto
    (10% de la capacidad, redondeado hacia arriba, edad 0)."""
    banco, hospitales = modelo_comun.inventario_inicial(insumos)
    return EstadoSistema(
        dia=0,
        inventario_banco=banco,
        inventario_hospitales=hospitales,
    )


def actualizar_estado(estado, decisiones, oferta_real, demanda_real, insumos):
    """Aplica lo que realmente pasó en el día y calcula el estado de mañana.

    No es un modelo de optimización: son cuentas, en este orden:
    1. La oferta admitida entra al banco con edad 0.
    2. Los despachos salen del banco y llegan a cada hospital con su misma edad.
    3. Cada hospital consume la demanda real con FIFO (primero lo más viejo);
       lo que no alcanza es demanda insatisfecha.
    4. Las unidades de edad L-1 que quedan se botan (vencidas). Las demás
       pasan la noche (pagan holding) y envejecen un día.

    Recibe:
        estado: EstadoSistema de hoy.
        decisiones: DecisionesDia implementadas hoy.
        oferta_real: oferta real de hoy (int).
        demanda_real: arreglo con la demanda real de hoy por hospital, en el
            orden de Insumos.hospitales.
        insumos: Insumos.
    Devuelve: (EstadoSistema de mañana, diccionario con lo que pasó hoy:
        oferta, admitida, despachada, faltante por hospital, vencidas en
        banco y hospitales, unidades que pasan la noche en banco y hospitales).
    """
    L = insumos.parametros["shelf_life"]
    if decisiones.oferta_admitida > oferta_real:
        raise ValueError("Se admitió más oferta de la disponible.")

    banco = estado.inventario_banco.astype(int).copy()
    hospitales = estado.inventario_hospitales.astype(int).copy()

    banco[0] += decisiones.oferta_admitida
    for (i, _k, a), unidades in decisiones.despachos.items():
        banco[a] -= unidades
        hospitales[i, a] += unidades
    if (banco < 0).any():
        raise ValueError("Se despachó más de lo que había en el banco.")

    faltante = np.zeros(len(hospitales), dtype=int)
    for i, pendiente in enumerate(np.asarray(demanda_real, dtype=int)):
        for a in range(L - 1, -1, -1):  # FIFO: de la edad mayor a la menor
            usado = min(pendiente, hospitales[i, a])
            hospitales[i, a] -= usado
            pendiente -= usado
        faltante[i] = pendiente

    resultado = {
        "oferta": int(oferta_real),
        "admitida": int(decisiones.oferta_admitida),
        "despachada": int(sum(decisiones.despachos.values())),
        "faltante": faltante,
        "vencidas_banco": int(banco[L - 1]),
        "vencidas_hospitales": hospitales[:, L - 1].copy(),
        "noche_banco": int(banco[: L - 1].sum()),
        "noche_hospitales": hospitales[:, : L - 1].sum(axis=1),
    }

    banco_manana = np.zeros(L, dtype=int)
    banco_manana[1:] = banco[: L - 1]
    hospitales_manana = np.zeros_like(hospitales)
    hospitales_manana[:, 1:] = hospitales[:, : L - 1]

    estado_manana = EstadoSistema(
        dia=estado.dia + 1,
        inventario_banco=banco_manana,
        inventario_hospitales=hospitales_manana,
    )
    return estado_manana, resultado


def ejecutar_semana(
    fecha_lunes,
    estado_inicial,
    observar_oferta,
    observar_demanda,
    insumos,
    n_escenarios=cfg.N_ESCENARIOS_DEFAULT,
    rng=None,
    resolver_calendario=modelo_comun.resolver_modelo_calendario,
    resolver_cantidades=modelo_comun.resolver_modelo_cantidades,
):
    """Ejecuta una semana completa con calendario fijo y horizonte rodante.

    Recibe:
        fecha_lunes: fecha (pandas Timestamp o compatible) del lunes.
        estado_inicial: EstadoSistema al inicio del lunes.
        observar_oferta: función fecha -> oferta real ese día (int). Se llama
            al inicio de cada jornada, ANTES de decidir.
        observar_demanda: función fecha -> arreglo con la demanda real por
            hospital (orden de Insumos.hospitales). Se llama DESPUÉS de
            implementar las decisiones del día.
        insumos: Insumos (de `preparar_insumos`).
        n_escenarios: escenarios por resolución.
        rng: numpy Generator para los escenarios del modelo (opcional).
        resolver_calendario, resolver_cantidades: funciones que resuelven cada
            modelo. Por defecto, las del modelo común. Una heurística entrega
            aquí sus propias funciones, con la misma firma y el mismo tipo de
            resultado.

    Devuelve: (SolucionCalendario, lista de 7 registros diarios). Cada
        registro tiene la fecha, las decisiones y lo que pasó ese día (ver
        `actualizar_estado`), además del estado final de la semana en
        el último registro (clave "estado_final").
    """
    fecha_lunes = pd.Timestamp(fecha_lunes)
    if fecha_lunes.dayofweek != cfg.DIA_INICIO_SEMANA:
        raise ValueError(f"{fecha_lunes.date()} no es el día de inicio de semana.")
    rng = rng if rng is not None else np.random.default_rng(cfg.SEMILLA)
    horizonte = insumos.parametros["horizonte"]
    mes_semana = fecha_lunes.month  # toda la semana usa el mes del lunes

    def escenarios_para(dias_restantes, oferta_hoy):
        return generar_escenarios(
            dias_restantes,
            mes_semana,
            insumos.distribuciones_demanda,
            n_escenarios,
            distribuciones_oferta=insumos.distribuciones_oferta,
            oferta_hoy=oferta_hoy,
            rng=rng,
        )

    # 1. Calendario: una sola vez. Supuesto vigente: la oferta del lunes ya se conoce.
    oferta_lunes = observar_oferta(fecha_lunes)
    escenarios_semana = escenarios_para(horizonte, oferta_lunes)
    calendario = resolver_calendario(escenarios_semana, estado_inicial, insumos)

    # 2. Loop diario del horizonte rodante (el lunes también resuelve cantidades).
    estado = estado_inicial
    registros = []
    for dia in range(horizonte):
        fecha = fecha_lunes + pd.Timedelta(days=dia)
        oferta_hoy = oferta_lunes if dia == 0 else observar_oferta(fecha)

        escenarios = escenarios_para(horizonte - dia, oferta_hoy)
        decisiones = resolver_cantidades(escenarios, estado, calendario, insumos)

        demanda_hoy = observar_demanda(fecha)
        estado, resultado = actualizar_estado(estado, decisiones, oferta_hoy, demanda_hoy, insumos)
        registros.append({"fecha": fecha, "decisiones": decisiones, **resultado})

    registros[-1]["estado_final"] = estado
    return calendario, registros
