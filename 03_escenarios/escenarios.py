"""Generación de escenarios de oferta y demanda para el horizonte rodante.

Oferta: cada día se sortea, de forma independiente, de la Normal del mes
(redondeada a entero y truncada en cfg.OFERTA_MINIMA). Toda la ventana usa el
mes del lunes de la semana (`mes_actual`), aunque la ventana cruce de mes.
No hay autocorrelación entre días: quitada la estacionalidad mensual, la
oferta es independiente día a día (análisis de
`analisis_de_la_oferta_y_demanda/Oferta_autocorrelacion.ipynb`).

Demanda: independiente entre hospitales y entre días, muestreada de la
distribución marginal de cada hospital.
"""

from dataclasses import dataclass

import numpy as np

from codigo import configuracion as cfg


@dataclass
class Escenarios:
    """Conjunto de escenarios para los días restantes.

    Atributos:
        oferta: arreglo int de forma (n_escenarios, dias_restantes).
        demanda: arreglo int de forma (n_escenarios, dias_restantes, n_hospitales).
        hospitales: nombres de los hospitales, en el orden del último eje de `demanda`.
        mes: mes usado para la oferta.
    """

    oferta: np.ndarray
    demanda: np.ndarray
    hospitales: list
    mes: int

    @property
    def n_escenarios(self):
        return self.oferta.shape[0]

    @property
    def dias(self):
        return self.oferta.shape[1]


def generar_escenarios(
    dias_restantes,
    mes_actual,
    hospitales,
    n_escenarios=cfg.N_ESCENARIOS_DEFAULT,
    *,
    distribuciones_oferta,
    oferta_hoy=None,
    rng=None,
):
    """Genera escenarios de oferta y demanda para los días restantes.

    Recibe:
        dias_restantes: número de días a generar (incluye el día en curso).
        mes_actual: mes (1–12) del lunes de la semana; se usa para todos los días.
        hospitales: diccionario {nombre: DistribucionDemanda}
            (de `distribuciones.ajustar_distribuciones_demanda`).
        n_escenarios: número de escenarios (por defecto cfg.N_ESCENARIOS_DEFAULT).
        distribuciones_oferta: diccionario {mes: DistribucionOfertaMensual}
            (de `distribuciones.ajustar_distribuciones_oferta`).
        oferta_hoy: oferta del día en curso si ya se conoce (según el
            Informe 1 §3.1, la oferta del día se conoce al inicio de la
            jornada). Si se entrega, el día 0 vale esto en todos los escenarios.
        rng: numpy Generator (opcional); si es None se usa cfg.SEMILLA.

    Devuelve: un objeto Escenarios (ver la clase).

    Supone: oferta independiente día a día dentro del mes; oferta y demanda
        independientes entre sí (correlación ≈ −0.02, Informe 1 §4.4);
        demandas independientes entre hospitales y días (Informe 1 §4.2).
    """
    if dias_restantes < 1:
        raise ValueError("dias_restantes debe ser al menos 1.")
    if mes_actual not in distribuciones_oferta:
        raise KeyError(f"No hay distribución de oferta para el mes {mes_actual}.")
    rng = rng if rng is not None else np.random.default_rng(cfg.SEMILLA)

    distribucion_mes = distribuciones_oferta[mes_actual]
    oferta = distribucion_mes.sample(n_escenarios * dias_restantes, rng=rng)
    oferta = oferta.reshape(n_escenarios, dias_restantes)
    if oferta_hoy is not None:
        oferta[:, 0] = oferta_hoy  # el valor observado se respeta tal cual

    nombres = list(hospitales)
    demanda = np.empty((n_escenarios, dias_restantes, len(nombres)), dtype=int)
    for indice, nombre in enumerate(nombres):
        muestras = hospitales[nombre].sample(n_escenarios * dias_restantes, rng=rng)
        demanda[:, :, indice] = muestras.reshape(n_escenarios, dias_restantes)

    return Escenarios(oferta=oferta, demanda=demanda, hospitales=nombres, mes=mes_actual)
