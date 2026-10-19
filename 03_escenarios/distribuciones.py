"""Distribuciones marginales de demanda (por hospital) y oferta (por mes).

Demanda: binomial negativa, binomial o Poisson según la Tabla A.3 del
Informe 1. Los parámetros se estiman por máxima verosimilitud con el mismo
procedimiento del notebook `Demanda_Capstone.ipynb`, que es el que produjo
esa tabla.

Oferta: una Normal por mes del año, ajustada por máxima verosimilitud
(media y desviación muestral sin corregir, igual que `scipy.stats.norm.fit`
en el notebook de oferta).
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

from codigo import configuracion as cfg

TIPOS_DEMANDA = ("binomial_negativa", "binomial", "poisson")


def _generador(rng):
    """Devuelve `rng` si es un Generator; si es None, crea uno con cfg.SEMILLA."""
    return rng if rng is not None else np.random.default_rng(cfg.SEMILLA)


class DistribucionDemanda:
    """Distribución discreta de la demanda diaria de un hospital.

    Atributos:
        tipo: "binomial_negativa", "binomial" o "poisson".
        parametros: diccionario con los parámetros ajustados
            (binomial_negativa: n, p; binomial: n, p; poisson: mu).
        distribucion: objeto congelado de scipy.stats.
    """

    def __init__(self, tipo, parametros, distribucion):
        self.tipo = tipo
        self.parametros = parametros
        self.distribucion = distribucion

    def sample(self, n, rng=None):
        """Genera `n` realizaciones independientes de la demanda diaria.

        Recibe: n (int); rng (numpy Generator, opcional).
        Devuelve: arreglo de enteros de largo n.
        """
        return self.distribucion.rvs(size=n, random_state=_generador(rng)).astype(int)

    def __repr__(self):
        return f"DistribucionDemanda({self.tipo}, {self.parametros})"


def _ajustar_binomial_negativa(x, media):
    def menos_log_verosimilitud(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(x, n=r, p=r / (r + media)).sum()

    ajuste = minimize_scalar(
        menos_log_verosimilitud,
        bounds=cfg.COTAS_LOG_R_BINOMIAL_NEGATIVA,
        method="bounded",
    )
    r = np.exp(ajuste.x)
    prob_exito = r / (r + media)
    return {"n": r, "p": prob_exito}, stats.nbinom(n=r, p=prob_exito)


def _ajustar_binomial(x, media):
    n_min = max(int(x.max()), int(np.ceil(media)))
    n_max = max(n_min + cfg.HOLGURA_N_BINOMIAL, cfg.FACTOR_N_BINOMIAL * n_min)
    mejor = None
    for n in range(n_min, n_max + 1):
        prob_exito = media / n
        if 0 < prob_exito < 1:
            log_verosimilitud = stats.binom.logpmf(x, n=n, p=prob_exito).sum()
            if mejor is None or log_verosimilitud > mejor[0]:
                mejor = (log_verosimilitud, n, prob_exito)
    if mejor is None:
        raise ValueError("No se encontró un ajuste binomial válido.")
    _, n, prob_exito = mejor
    return {"n": n, "p": prob_exito}, stats.binom(n=n, p=prob_exito)


def ajustar_distribucion_demanda(serie_demanda, tipo):
    """Ajusta la distribución de demanda de un hospital.

    Recibe:
        serie_demanda: secuencia (Series, lista o arreglo) con la demanda
            diaria histórica del hospital. Deben ser enteros no negativos.
        tipo: "binomial_negativa", "binomial" o "poisson" (Tabla A.3).
    Devuelve: un DistribucionDemanda con método .sample(n).
    Supone: días independientes e idénticamente distribuidos (el Informe 1
        encontró autocorrelación y estacionalidad despreciables en la demanda).
        El tipo viene dado y no se valida aquí si es el mejor ajuste.
    """
    if tipo not in TIPOS_DEMANDA:
        raise ValueError(f"Tipo '{tipo}' no reconocido; debe ser uno de {TIPOS_DEMANDA}.")
    x = np.asarray(serie_demanda)
    if not np.allclose(x, np.round(x)) or (x < 0).any():
        raise ValueError("La demanda debe ser entera y no negativa.")
    x = np.round(x).astype(int)
    media = x.mean()
    varianza = x.var(ddof=1)

    if tipo == "poisson":
        parametros, distribucion = {"mu": media}, stats.poisson(mu=media)
    elif tipo == "binomial_negativa":
        if varianza <= media:
            raise ValueError("La binomial negativa requiere varianza > media.")
        parametros, distribucion = _ajustar_binomial_negativa(x, media)
    else:
        if varianza >= media:
            raise ValueError("La binomial requiere varianza < media.")
        parametros, distribucion = _ajustar_binomial(x, media)
    return DistribucionDemanda(tipo, parametros, distribucion)


def ajustar_distribuciones_demanda(demanda_historica, tipos=None):
    """Ajusta la distribución de demanda de todos los hospitales.

    Recibe:
        demanda_historica: DataFrame de `cargador_datos.cargar_demanda_historica()`.
        tipos: diccionario {hospital: tipo}; por defecto
            cfg.TIPO_DISTRIBUCION_DEMANDA (Tabla A.3).
    Devuelve: diccionario {hospital: DistribucionDemanda}, en el orden de
        aparición de los hospitales en los datos.
    """
    tipos = cfg.TIPO_DISTRIBUCION_DEMANDA if tipos is None else tipos
    distribuciones = {}
    for hospital, grupo in demanda_historica.groupby("hospital", sort=False):
        if hospital not in tipos:
            raise KeyError(f"No hay tipo de distribución definido para '{hospital}'.")
        distribuciones[hospital] = ajustar_distribucion_demanda(grupo["demanda"], tipos[hospital])
    return distribuciones


class DistribucionOfertaMensual:
    """Distribución Normal de la oferta diaria para un mes del año.

    Atributos: mes (1–12), media, desviacion.
    """

    def __init__(self, mes, media, desviacion):
        self.mes = mes
        self.media = media
        self.desviacion = desviacion

    def a_unidades(self, valores_continuos):
        """Convierte valores continuos de oferta a unidades: redondea a entero
        y trunca en cfg.OFERTA_MINIMA."""
        redondeados = np.rint(valores_continuos).astype(int)
        return np.maximum(redondeados, cfg.OFERTA_MINIMA)

    def sample(self, n, rng=None):
        """Genera `n` realizaciones independientes de la oferta diaria del mes.

        Recibe: n (int); rng (numpy Generator, opcional).
        Devuelve: arreglo de enteros de largo n (redondeados y truncados).
        """
        valores = _generador(rng).normal(self.media, self.desviacion, size=n)
        return self.a_unidades(valores)

    def __repr__(self):
        return f"DistribucionOfertaMensual(mes={self.mes}, media={self.media:.1f}, desviacion={self.desviacion:.1f})"


def ajustar_distribucion_oferta(oferta_historica, mes):
    """Ajusta una Normal a la oferta histórica de un mes del año.

    Recibe:
        oferta_historica: DataFrame de `cargador_datos.cargar_oferta_historica()`.
        mes: entero 1–12.
    Devuelve: un DistribucionOfertaMensual con método .sample(n).
    Supone: dentro de un mes la oferta es Normal (Informe 1 §4.3 y §5.8.2;
        validación formal pendiente).
    """
    valores = oferta_historica.loc[oferta_historica["fecha"].dt.month == mes, "oferta"]
    if valores.empty:
        raise ValueError(f"No hay datos de oferta para el mes {mes}.")
    media, desviacion = stats.norm.fit(valores.to_numpy(dtype=float))
    return DistribucionOfertaMensual(mes, media, desviacion)


def ajustar_distribuciones_oferta(oferta_historica):
    """Ajusta la Normal de oferta para los 12 meses.

    Devuelve: diccionario {mes: DistribucionOfertaMensual}.
    """
    meses = sorted(oferta_historica["fecha"].dt.month.unique())
    return {int(mes): ajustar_distribucion_oferta(oferta_historica, mes) for mes in meses}
