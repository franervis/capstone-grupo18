"""Validación A.3 y comparación retrospectiva. Uso: python -m analisis_demanda.analisis --help."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from codigo.cargador_datos import cargar_demanda_historica, cargar_parametros, tabla_hospitales

SEMILLA = None

TIPO_DISTRIBUCION_DEMANDA = {
    "Instituto Traumatológico": "binomial_negativa",
    "Hospital Calvo Mackenna": "binomial_negativa",
    "Hospital San José": "binomial_negativa",
    "Hospital Roberto del Río": "binomial_negativa",
    "Hospital del Salvador": "binomial_negativa",
    "Hospital San Juan de Dios": "binomial_negativa",
    "Hospital Barros Luco": "binomial_negativa",
    "Ex Posta Central": "binomial_negativa",
    "Hospital Félix Bulnes": "binomial_negativa",
    "Hospital San Borja Arriarán": "binomial_negativa",
    "Hospital Luis Tisné": "binomial",
    "Hospital Padre Hurtado": "binomial_negativa",
    "Hospital Parroquial San Bernardo": "poisson",
    "Hospital Lucio Córdova": "binomial",
    "Hospital San José de Maipo": "binomial",
    "Hospital Exequiel González Cortés": "binomial_negativa",
    "Hospital Sótero del Río": "binomial_negativa",
    "Hospital El Pino": "binomial_negativa",
    "Instituto Geriatría": "binomial",
    "Instituto Nacional del Cáncer": "binomial",
    "Instituto Neurocirugía": "binomial",
    "Instituto Psiquiátrico": "binomial_negativa",
}

# Rango de búsqueda de log(r) en la máxima verosimilitud de la binomial
# negativa. Es el mismo del notebook Demanda_Capstone.ipynb, con el que se
# obtuvo la Tabla A.3; se mantiene para reproducir esos ajustes.
# Si se cambia: si el óptimo queda en el borde, ampliar el rango.
COTAS_LOG_R_BINOMIAL_NEGATIVA = (-8.0, 12.0)

# Rango de búsqueda del parámetro n (número de ensayos) de la binomial,
# igual que en el notebook de demanda:
#   n_min = max(máximo observado, ceil(media))
#   n_max = max(n_min + HOLGURA_N_BINOMIAL, FACTOR_N_BINOMIAL * n_min)
# Si se cambia: un rango más amplio es más lento pero explora más valores.
HOLGURA_N_BINOMIAL = 100
FACTOR_N_BINOMIAL = 4

# Validación de demanda: nivel nominal y bootstrap paramétrico con reajuste.
# 49.999 réplicas dan resolución de p = 0.00002. Aumentar para mayor precisión.
ALFA_VALIDACION_DEMANDA = 0.05
REPLICAS_VALIDACION_DEMANDA = 49999
# Procesos independientes por hospital; la semilla por hospital preserva
# resultados idénticos al cambiar este valor. Reducir a 1 para uso secuencial.
TRABAJADORES_VALIDACION_DEMANDA = 4
# Semilla exclusiva del análisis; no modifica la aleatoriedad de escenarios.
SEMILLA_VALIDACION_DEMANDA = 20261002
# Agrupar categorías contiguas de Pearson hasta reunir al menos 5 esperados.
MIN_ESPERADO_CHI2 = 5.0
# Masa máxima omitida en la suma AD de soporte infinito (no en Pearson).
TOL_COLA_AD = 1e-12

# Comparación retrospectiva: parámetros ajustados sólo antes del corte.
CORTE_COMPARACION = "2024-01-01"
FIN_COMPARACION = "2024-12-31"
HOSPITALES_COMPARACION = (
    "Instituto Traumatológico", "Hospital San José", "Hospital San Juan de Dios",
    "Hospital Barros Luco", "Instituto Geriatría",
)
CUANTILES_COMPARACION = (0.95, 0.99)
# Bootstrap pareado en bloques circulares de una semana sobre días de prueba.
REPLICAS_COMPARACION = 9999
BLOQUE_COMPARACION = 7
SEMILLA_COMPARACION = 20261003
ALFA_COMPARACION = 0.05
# Cotas numéricas de log(sigma) para Normal censurada y redondeada.
COTAS_LOG_SIGMA_NORMAL = (-6.0, 8.0)


TIPOS_DEMANDA = ("binomial_negativa", "binomial", "poisson")
PRUEBAS = ("ks", "ad", "chi2")


# Ajustes de las familias A.3


def _generador(rng):
    """Devuelve `rng` si es un Generator; si es None, crea uno con SEMILLA."""
    return rng if rng is not None else np.random.default_rng(SEMILLA)


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
    valores, frecuencias = np.unique(x, return_counts=True)

    def menos_log_verosimilitud(log_r):
        r = np.exp(log_r)
        return -np.dot(frecuencias, stats.nbinom.logpmf(valores, n=r, p=r / (r + media)))

    ajuste = minimize_scalar(
        menos_log_verosimilitud,
        bounds=COTAS_LOG_R_BINOMIAL_NEGATIVA,
        method="bounded",
    )
    if not ajuste.success or not np.isfinite(ajuste.fun):
        raise RuntimeError("Falló el ajuste de la binomial negativa.")
    r = np.exp(ajuste.x)
    p = r / (r + media)
    return {"n": r, "p": p}, stats.nbinom(n=r, p=p)


def _ajustar_binomial(x, media):
    n_min = max(int(x.max()), int(np.ceil(media)))
    n_max = max(n_min + HOLGURA_N_BINOMIAL, FACTOR_N_BINOMIAL * n_min)
    valores, frecuencias = np.unique(x, return_counts=True)
    candidatos = np.arange(max(1, n_min), n_max + 1)
    probabilidades = media / candidatos
    # Misma búsqueda exhaustiva del notebook, con frecuencias suficientes
    # para no repetir el cálculo de logpmf por cada observación idéntica.
    log_verosimilitudes = np.sum(
        stats.binom.logpmf(valores[None, :], candidatos[:, None],
                          probabilidades[:, None]) * frecuencias[None, :], axis=1
    )
    indice = int(np.argmax(log_verosimilitudes))
    if not np.isfinite(log_verosimilitudes[indice]):
        raise ValueError("No se encontró un ajuste binomial válido.")
    n, p = int(candidatos[indice]), float(probabilidades[indice])
    return {"n": n, "p": p}, stats.binom(n=n, p=p)


def ajustar_distribucion_demanda(serie_demanda, tipo, *, validar_dispersion=True):
    """Ajusta la distribución de demanda de un hospital.

    Recibe:
        serie_demanda: secuencia (Series, lista o arreglo) con la demanda
            diaria histórica del hospital. Deben ser enteros no negativos.
        tipo: "binomial_negativa", "binomial" o "poisson" (Tabla A.3).
        validar_dispersion: False sólo para reajustar la familia fijada en
            el bootstrap, incluso cuando la dispersión muestral cambia.
    Devuelve: un DistribucionDemanda con método .sample(n).
    Supone días independientes e idénticamente distribuidos; este ajuste
        no verifica esa hipótesis.
        El tipo viene dado y no se valida aquí si es el mejor ajuste.
    """
    if tipo not in TIPOS_DEMANDA:
        raise ValueError(f"Tipo '{tipo}' no reconocido; debe ser uno de {TIPOS_DEMANDA}.")
    x = np.asarray(serie_demanda, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("Se requieren al menos dos demandas finitas en un vector.")
    if not np.equal(x, np.round(x)).all() or (x < 0).any():
        raise ValueError("La demanda debe ser entera y no negativa.")
    x = np.round(x).astype(int)
    media = x.mean()
    varianza = x.var(ddof=1)

    if media == 0 and not validar_dispersion and tipo != "poisson":
        # MLE degenerada en cero: no descartar ni volver a simular réplicas.
        p = 1.0 if tipo == "binomial_negativa" else 0.0
        familia = stats.nbinom if tipo == "binomial_negativa" else stats.binom
        parametros, distribucion = {"n": 1, "p": p}, familia(n=1, p=p)
    elif tipo == "poisson":
        parametros, distribucion = {"mu": media}, stats.poisson(mu=media)
    elif tipo == "binomial_negativa":
        if validar_dispersion and varianza <= media:
            raise ValueError("La binomial negativa requiere varianza > media.")
        parametros, distribucion = _ajustar_binomial_negativa(x, media)
    else:
        if validar_dispersion and varianza >= media:
            raise ValueError("La binomial requiere varianza < media.")
        parametros, distribucion = _ajustar_binomial(x, media)
    return DistribucionDemanda(tipo, parametros, distribucion)


# Validación formal


def _validar_muestra(x):
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or x.size < 2 or not np.isfinite(x).all()
            or (x < 0).any() or not np.equal(x, np.round(x)).all()):
        raise ValueError("Se requieren al menos dos enteros finitos no negativos.")
    return x.astype(int)


def estadisticos_discretos(x, distribucion, min_esperado=MIN_ESPERADO_CHI2):
    """Devuelve [KS, AD, Pearson] y clases de Pearson (incluida toda la cola).

    AD = N sum_k (Fn(k)-F(k))² w(k)/(F(k) sf(k)), con 0<F(k)<1,
    w(k) = (pmf(k)+pmf(k+1))/2 (Choulakian; implementación dgof).
    Para soporte infinito se corta en max(máximo observado, PPF(1-tol)).
    Las categorías de Pearson se reconstruyen con la misma regla en cada
    réplica: unir enteros contiguos hasta obtener al menos min_esperado.
    """
    x = _validar_muestra(x)
    if not np.isfinite(min_esperado) or min_esperado <= 0:
        raise ValueError("min_esperado debe ser positivo y finito.")
    n = x.size
    limite = max(int(x.max()), int(distribucion.ppf(1 - TOL_COLA_AD)))
    soporte = np.arange(limite + 1)
    conteos = np.bincount(x, minlength=limite + 1)
    acumulados = conteos.cumsum()
    f = distribucion.cdf(soporte)
    sf = distribucion.sf(soporte)
    pmf = distribucion.pmf(soporte)
    # La diferencia de colas evita la cancelación de 1-F en valores extremos.
    diferencia = np.where(f <= 0.5, acumulados / n - f,
                          sf - (n - acumulados) / n)
    ks = float(np.max(np.abs(diferencia)))
    interior = (f > 0) & (sf > 0)
    # Ponderación discreta de Choulakian et al., usada por dgof::cvm.test(A2).
    # Se promedian las masas adyacentes; no es la integral unilateral con pmf(k).
    pesos_ad = (pmf + distribucion.pmf(soporte + 1)) / 2
    ad = float(n * np.sum(diferencia[interior] ** 2 * pesos_ad[interior]
                          / f[interior] / sf[interior]))
    if np.any((conteos > 0) & ~np.isfinite(distribucion.logpmf(soporte))):
        # Un dato imposible bajo H0 no puede desaparecer al agrupar clases.
        ad = np.inf
        imposible = True
    else:
        imposible = False

    # La última categoría llega a infinito, incluso si no hubo observaciones
    # en esa cola. No se renormaliza la probabilidad al rango observado.
    cortes = []
    esperado_acumulado = 0.0
    for k, prob in enumerate(pmf):
        esperado_acumulado += n * prob
        if esperado_acumulado >= min_esperado:
            cortes.append(k)
            esperado_acumulado = 0.0
    if cortes and n * distribucion.sf(cortes[-1]) < min_esperado:
        cortes.pop()  # unir remanente pequeño con la clase precedente
    limites = np.array([-1, *cortes, np.inf], dtype=float)
    f_lim = distribucion.cdf(limites)
    sf_lim = distribucion.sf(limites)
    probabilidades = np.where(f_lim[1:] <= 0.5, np.diff(f_lim),
                              sf_lim[:-1] - sf_lim[1:])
    esperados = n * probabilidades
    indices = np.searchsorted(np.asarray(cortes), x, side="left")
    observados = np.bincount(indices, minlength=len(esperados))
    if len(esperados) < 2 or esperados.min() < min_esperado * (1 - 1e-10):
        raise ValueError("No hay al menos dos clases con suficientes esperados para χ².")
    chi2 = float(np.sum((observados - esperados) ** 2 / esperados))
    if imposible:
        chi2 = np.inf
    clases = pd.DataFrame({
        "desde": (limites[:-1] + 1).astype(int), "hasta": limites[1:],
        "observado": observados, "esperado": esperados,
    })
    return np.array([ks, ad, chi2]), clases


def _en_cota(ajuste, x):
    if ajuste.tipo == "binomial_negativa":
        return bool(np.min(np.abs(np.log(ajuste.parametros["n"])
                                   - np.array(COTAS_LOG_R_BINOMIAL_NEGATIVA))) < 1e-3)
    if ajuste.tipo == "binomial":
        n_min = max(int(np.max(x)), int(np.ceil(np.mean(x))))
        n_max = max(n_min + HOLGURA_N_BINOMIAL, FACTOR_N_BINOMIAL * n_min)
        return ajuste.parametros["n"] == n_max
    return False


def validar_hospital(x, tipo, *, replicas=REPLICAS_VALIDACION_DEMANDA,
                     rng=None, min_esperado=MIN_ESPERADO_CHI2):
    """Prueba la familia fijada con parámetros desconocidos, bajo supuesto iid.

    No se descartan réplicas por su varianza: se reajusta siempre la misma
    familia, incluyendo soluciones en la frontera de la búsqueda original.
    Un error de ajuste detiene la ejecución en vez de sesgar la simulación.
    """
    x = _validar_muestra(x)
    if isinstance(replicas, bool) or not isinstance(replicas, (int, np.integer)) or replicas < 1:
        raise ValueError("replicas debe ser un entero positivo.")
    rng = np.random.default_rng(rng)
    ajuste = ajustar_distribucion_demanda(x, tipo, validar_dispersion=False)
    observados, clases = estadisticos_discretos(x, ajuste.distribucion, min_esperado)
    excedencias = np.zeros(3, dtype=int)
    cotas = 0
    for _ in range(replicas):
        muestra = ajuste.sample(len(x), rng)
        simulado = ajustar_distribucion_demanda(muestra, tipo, validar_dispersion=False)
        valores, _ = estadisticos_discretos(muestra, simulado.distribucion, min_esperado)
        # Tolerancia relativa sólo para empates numéricos.
        tolerancia = np.where(np.isfinite(observados), 1e-12 * np.abs(observados), 0)
        excedencias += valores >= observados - tolerancia
        cotas += _en_cota(simulado, muestra)
    resultado = {
        "modelo": tipo, "n_dias": len(x), "media": float(x.mean()),
        "var_media": float(x.var(ddof=1) / x.mean()),
        "parametros": json.dumps({k: float(v) for k, v in ajuste.parametros.items()},
                                  ensure_ascii=False, sort_keys=True),
        "p95_observado": float(np.quantile(x, .95)),
        "p95_modelo": float(ajuste.distribucion.ppf(.95)),
        "clases_chi2": len(clases), "min_esperado_chi2": float(clases.esperado.min()),
        "ajuste_en_cota": _en_cota(ajuste, x), "replicas_en_cota": cotas,
        "replicas": replicas,
    }
    for j, prueba in enumerate(PRUEBAS):
        b = int(excedencias[j])
        p = (b + 1) / (replicas + 1)
        resultado.update({
            f"{prueba}_estadistico": float(observados[j]),
            f"{prueba}_excedencias": b, f"{prueba}_p": p,
            f"{prueba}_error_mc": float(np.sqrt(p * (1 - p) / replicas)),
            # Intervalo binomial exacto para la probabilidad de excedencia;
            # cuantifica error Monte Carlo, no incertidumbre del modelo.
            f"{prueba}_mc95_inf": float(stats.beta.ppf(.025, b, replicas - b + 1)) if b else 0.,
            f"{prueba}_mc95_sup": float(stats.beta.ppf(.975, b + 1, replicas - b)) if b < replicas else 1.,
        })
    return resultado, clases


def ajustar_holm(valores_p):
    """Holm step-down; admite dependencia entre los contrastes."""
    p = np.asarray(valores_p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Se requiere un vector de valores p en [0,1].")
    orden = np.argsort(p)
    ajustados = np.empty_like(p)
    ajustados[orden] = np.minimum(1, np.maximum.accumulate(p[orden] * np.arange(len(p), 0, -1)))
    return ajustados


def agregar_decisiones(tabla, alfa):
    """Añade decisiones y una sensibilidad numérica a los intervalos Monte Carlo.

    Propagar intervalos marginales por Holm es un diagnóstico de sensibilidad,
    NO un intervalo simultáneo ni una nueva regla de rechazo estadístico.
    """
    tabla = tabla.copy()
    holm = ajustar_holm(tabla[[f"{p}_p" for p in PRUEBAS]].to_numpy().ravel()).reshape(-1, 3)
    for j, prueba in enumerate(PRUEBAS):
        tabla[f"{prueba}_p_holm"] = holm[:, j]
        tabla[f"{prueba}_rechaza"] = tabla[f"{prueba}_p"] <= alfa
        tabla[f"{prueba}_rechaza_holm"] = holm[:, j] <= alfa
    tabla["rechaza_alguna_sin_ajuste"] = tabla[[f"{p}_rechaza" for p in PRUEBAS]].any(axis=1)
    tabla["rechaza_alguna_holm"] = tabla[[f"{p}_rechaza_holm" for p in PRUEBAS]].any(axis=1)
    decisiones_extremos = []
    for extremo in ("inf", "sup"):
        p = tabla[[f"{p}_mc95_{extremo}" for p in PRUEBAS]].to_numpy().ravel()
        decisiones_extremos.append((ajustar_holm(p).reshape(-1, 3) <= alfa).any(axis=1))
    tabla["holm_sensible_mc"] = decisiones_extremos[0] != decisiones_extremos[1]
    tabla["alfa"] = alfa
    return tabla


def _validar_grupo(hospital, tipo, grupo, replicas, hijo):
    """Trabajo independiente; la semilla depende del hospital, no del proceso."""
    fila, clases = validar_hospital(grupo.demanda.to_numpy(), tipo, replicas=replicas,
                                   rng=np.random.default_rng(hijo))
    fila.update(hospital=hospital, fecha_inicio=str(grupo.fecha.min().date()),
                fecha_fin=str(grupo.fecha.max().date()))
    return fila, clases.assign(hospital=hospital)


def validar_demanda(datos, *, tipos=None, replicas=REPLICAS_VALIDACION_DEMANDA,
                    semilla=SEMILLA_VALIDACION_DEMANDA,
                    alfa=ALFA_VALIDACION_DEMANDA, progreso=None, trabajadores=1):
    """Devuelve resultados y clases; Holm abarca los 3 × H contrastes."""
    if not 0 < alfa < 1:
        raise ValueError("alfa debe estar entre 0 y 1.")
    if isinstance(trabajadores, bool) or not isinstance(trabajadores, int) or trabajadores < 1:
        raise ValueError("trabajadores debe ser un entero positivo.")
    tipos = TIPO_DISTRIBUCION_DEMANDA if tipos is None else tipos
    if datos.empty or datos[["hospital", "fecha", "demanda"]].isna().any().any():
        raise ValueError("No se admiten datos vacíos ni valores faltantes.")
    if datos.duplicated(["hospital", "fecha"]).any():
        raise ValueError("Hay observaciones duplicadas por hospital y fecha.")
    if set(datos.hospital) != set(tipos):
        raise ValueError("Los hospitales de los datos deben coincidir con las familias especificadas.")
    hijos = np.random.SeedSequence(semilla).spawn(len(tipos))
    trabajos = []
    for (hospital, tipo), hijo in zip(tipos.items(), hijos):
        grupo = datos.loc[datos.hospital == hospital].sort_values("fecha")
        trabajos.append((hospital, tipo, grupo, replicas, hijo))
    resultados = {}
    def recibir(resultado):
        fila, clases = resultado
        resultados[fila["hospital"]] = (fila, clases)
        if progreso is not None:
            progreso(fila["hospital"], fila)
    if trabajadores == 1:
        for trabajo in trabajos:
            recibir(_validar_grupo(*trabajo))
    else:
        with ProcessPoolExecutor(max_workers=min(trabajadores, len(trabajos))) as executor:
            pendientes = [executor.submit(_validar_grupo, *trabajo) for trabajo in trabajos]
            for futuro in as_completed(pendientes):
                recibir(futuro.result())
    filas, detalle = zip(*(resultados[h] for h in tipos))
    tabla = pd.DataFrame(filas)
    tabla = agregar_decisiones(tabla, alfa)
    tabla["semilla"] = semilla
    return tabla, pd.concat(detalle, ignore_index=True)


# Comparación de los cinco ajustes rechazados


class NormalDiscreta:
    """X=max(0, floor(Z+0.5)), Z~Normal(mu,sigma); no es Normal truncada."""
    def __init__(self, mu, sigma):
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("mu finita y sigma positiva son obligatorias.")
        self.mu, self.sigma = float(mu), float(sigma)

    def cdf(self, k):
        k = np.asarray(k)
        return np.where(k < 0, 0., stats.norm.cdf((np.floor(k)+.5-self.mu)/self.sigma))

    def sf(self, k):
        k = np.asarray(k)
        return np.where(k < 0, 1., stats.norm.sf((np.floor(k)+.5-self.mu)/self.sigma))

    def logpmf(self, k):
        k = np.asarray(k, dtype=float)
        a, b = (k-.5-self.mu)/self.sigma, (k+.5-self.mu)/self.sigma
        hi = np.where(a > 0, stats.norm.logsf(a), stats.norm.logcdf(b))
        lo = np.where(a > 0, stats.norm.logsf(b), stats.norm.logcdf(a))
        with np.errstate(divide="ignore", invalid="ignore"):
            valor = hi + np.log(-np.expm1(lo-hi))
        valor = np.where(k == 0, stats.norm.logcdf(b), valor)
        return np.where((k >= 0) & (k == np.floor(k)), valor, -np.inf)

    def pmf(self, k):
        return np.exp(self.logpmf(k))

    def ppf(self, q):
        return np.maximum(0, np.ceil(self.mu+self.sigma*stats.norm.ppf(q)-.5))

    def rvs(self, size, random_state=None):
        rng = np.random.default_rng(random_state)
        return np.maximum(0, np.floor(rng.normal(self.mu, self.sigma, size)+.5)).astype(int)


def ajustar_alternativas(entrenamiento, tipo):
    x = _validar_muestra(entrenamiento)
    a3 = ajustar_distribucion_demanda(x, tipo, validar_dispersion=False)
    valores, conteos = np.unique(x, return_counts=True)
    empirica = stats.rv_discrete(values=(valores, conteos/len(x)))
    def objetivo(theta):
        d = NormalDiscreta(theta[0], np.exp(theta[1]))
        return -np.dot(conteos, d.logpmf(valores))
    opt = minimize(objetivo, [x.mean(), np.log(max(x.std(), .1))],
                   method="L-BFGS-B", bounds=[(None, None), COTAS_LOG_SIGMA_NORMAL])
    if not opt.success or not np.isfinite(opt.fun):
        raise RuntimeError(f"No convergió la Normal discretizada: {opt.message}")
    if min(abs(opt.x[1]-np.array(COTAS_LOG_SIGMA_NORMAL))) < 1e-3:
        raise RuntimeError("Ajuste Normal en cota; revisar antes de comparar.")
    normal = NormalDiscreta(opt.x[0], np.exp(opt.x[1]))
    return {
        "a3": (a3.distribucion, {"familia": tipo, **a3.parametros}),
        "empirica": (empirica, {"valores": valores.tolist(), "conteos": conteos.tolist()}),
        "normal_discreta": (normal, {"mu_latente": normal.mu, "sigma_latente": normal.sigma}),
    }


def crps_discreto(distribucion, observaciones):
    """RPS sobre enteros = CRPS; menor es mejor. No normaliza por soporte."""
    y = _validar_muestra(observaciones)
    limite = max(int(y.max()), int(distribucion.ppf(1-TOL_COLA_AD)))
    k = np.arange(limite+1)
    f = distribucion.cdf(k)
    return np.sum((f[None, :] - (y[:, None] <= k[None, :]))**2, axis=1)


def pinball(y, q, tau):
    error = np.asarray(y)-q
    return np.maximum(tau*error, (tau-1)*error)


def costo_un_dia(y, q, costo_faltante, costo_exceso):
    """Prueba aislada: exceso al cierre y faltante. No simula el SIRP."""
    return costo_faltante*np.maximum(np.asarray(y)-q, 0) + costo_exceso*np.maximum(q-np.asarray(y), 0)


def evaluar_modelo(d, y, capacidad, costos):
    tau = costos["costo_shortage"]/(costos["costo_shortage"]+costos["costo_holding_hospital"])
    q_libre = int(d.ppf(tau))
    q = min(q_libre, int(capacidad))
    diarios = pd.DataFrame({"demanda": y, "crps": crps_discreto(d, y)})
    diarios["faltante_proxy"] = np.maximum(y-q, 0)
    diarios["exceso_proxy"] = np.maximum(q-y, 0)
    diarios["costo_proxy"] = costo_un_dia(y, q, costos["costo_shortage"], costos["costo_holding_hospital"])
    resumen = {"q_proxy": q, "q_proxy_sin_capacidad": q_libre, "capacidad": int(capacidad),
               "tau_proxy": tau, "crps": diarios.crps.mean(),
               "costo_proxy": diarios.costo_proxy.mean(),
               "faltante_proxy": diarios.faltante_proxy.mean(),
               "exceso_proxy": diarios.exceso_proxy.mean(),
               "dias_con_faltante_proxy": int((y > q).sum()),
               "dias_con_probabilidad_cero": int(np.isneginf(d.logpmf(y)).sum())}
    for tau_cola in CUANTILES_COMPARACION:
        nombre = f"p{int(100*tau_cola)}"
        q_cola = int(d.ppf(tau_cola))
        diarios[f"pinball_{nombre}"] = pinball(y, q_cola, tau_cola)
        resumen.update({nombre: q_cola, f"{nombre}_observado": float(np.quantile(y, tau_cola)),
                        f"cobertura_{nombre}": float(np.mean(y <= q_cola)),
                        f"cobertura_modelo_{nombre}": float(d.cdf(q_cola)),
                        f"pinball_{nombre}": diarios[f"pinball_{nombre}"].mean(),
                        f"faltante_{nombre}": float(np.maximum(y-q_cola, 0).mean())})
    return resumen, diarios


def intervalos_pareados(diferencias, *, rng, replicas, bloque, n_contrastes):
    """Percentiles por bloques circulares; conserva pareamiento día/modelo.

    El ajuste Bonferroni abarca las dos métricas × dos alternativas × H.
    Son intervalos exploratorios de evaluación, condicionados a los ajustes.
    """
    x = np.asarray(diferencias, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= bloque <= len(x):
        raise ValueError("Diferencias finitas y bloque entre 1 y N son obligatorios.")
    if replicas < 1 or n_contrastes < 1:
        raise ValueError("Se requieren réplicas y contrastes positivos.")
    n = len(x)
    resultados = np.empty((replicas, x.shape[1]))
    # Lotes acotan memoria; la secuencia RNG no depende del tamaño del lote.
    for inicio in range(0, replicas, 250):
        cantidad = min(250, replicas-inicio)
        comienzos = rng.integers(n, size=(cantidad, int(np.ceil(n/bloque))))
        indices = ((comienzos[:, :, None]+np.arange(bloque)) % n).reshape(cantidad, -1)[:, :n]
        resultados[inicio:inicio+cantidad] = x[indices].mean(axis=1)
    a = ALFA_COMPARACION/2
    return np.quantile(resultados, [a, 1-a, a/n_contrastes, 1-a/n_contrastes], axis=0)


def comparar(datos, costos, capacidades):
    if datos.duplicated(["hospital", "fecha"]).any() or datos[["hospital", "fecha", "demanda"]].isna().any().any():
        raise ValueError("Datos duplicados o faltantes.")
    resumenes, diarios_todos, pares, ajustes = [], [], [], {}
    hijos = np.random.SeedSequence(SEMILLA_COMPARACION).spawn(len(HOSPITALES_COMPARACION))
    for hospital, hijo in zip(HOSPITALES_COMPARACION, hijos):
        g = datos.loc[datos.hospital.eq(hospital)].sort_values("fecha")
        train = g.loc[g.fecha < CORTE_COMPARACION]
        test = g.loc[(g.fecha >= CORTE_COMPARACION) & (g.fecha <= FIN_COMPARACION)]
        if train.empty or test.empty or train.fecha.max() >= test.fecha.min():
            raise ValueError(f"Separación temporal inválida: {hospital}")
        modelos = ajustar_alternativas(train.demanda.to_numpy(), TIPO_DISTRIBUCION_DEMANDA[hospital])
        ajustes[hospital] = {m: p for m, (_, p) in modelos.items()}
        dias_modelos = {}
        for modelo, (dist, _) in modelos.items():
            r, d = evaluar_modelo(dist, test.demanda.to_numpy(), capacidades[hospital], costos)
            r.update(hospital=hospital, modelo=modelo, n_entrenamiento=len(train), n_prueba=len(test),
                     inicio_entrenamiento=str(train.fecha.min().date()), fin_entrenamiento=str(train.fecha.max().date()),
                     inicio_prueba=str(test.fecha.min().date()), fin_prueba=str(test.fecha.max().date()))
            resumenes.append(r)
            d["fecha"] = test.fecha.to_numpy()
            diarios_todos.append(d.assign(hospital=hospital, modelo=modelo))
            dias_modelos[modelo] = d
        combinaciones = [(m, metrica) for m in ("empirica", "normal_discreta") for metrica in ("crps", "costo_proxy")]
        dif = np.column_stack([dias_modelos[m][v]-dias_modelos["a3"][v] for m, v in combinaciones])
        intervalos = intervalos_pareados(dif, rng=np.random.default_rng(hijo),
                                       replicas=REPLICAS_COMPARACION, bloque=BLOQUE_COMPARACION,
                                       n_contrastes=len(HOSPITALES_COMPARACION)*len(combinaciones))
        for j, (modelo, metrica) in enumerate(combinaciones):
            base = dias_modelos["a3"][metrica].mean()
            delta = dif[:, j].mean()
            pares.append({"hospital": hospital, "alternativa": modelo, "metrica": metrica,
                          "delta_alternativa_menos_a3": delta, "delta_pct_sobre_a3": 100*delta/base,
                          "ic95_inf": intervalos[0, j], "ic95_sup": intervalos[1, j],
                          "ic_familia_inf": intervalos[2, j], "ic_familia_sup": intervalos[3, j],
                          "evidencia_mejora": bool(intervalos[3, j] < 0),
                          "evidencia_empeora": bool(intervalos[2, j] > 0)})
        print(f"Comparado: {hospital} ({len(train)} días de ajuste; {len(test)} de evaluación)", flush=True)
    return pd.DataFrame(resumenes), pd.concat(diarios_todos, ignore_index=True), pd.DataFrame(pares), ajustes


def unir_comparacion(resumen, pares):
    """Una fila por hospital/modelo, con diferencias e intervalos frente a A.3."""
    tabla = resumen.copy()
    for metrica in ("crps", "costo_proxy"):
        detalle = pares.loc[pares.metrica.eq(metrica)].drop(columns="metrica")
        detalle = detalle.rename(columns={"alternativa": "modelo"})
        detalle = detalle.rename(columns={c: f"{c}_{metrica}" for c in detalle
                                          if c not in ("hospital", "modelo")})
        tabla = tabla.merge(detalle, on=["hospital", "modelo"], how="left", validate="one_to_one")
    return tabla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("etapa", choices=("comparar", "validar", "todo"),
                        help="comparar es rápido; validar/todo requieren el bootstrap completo")
    parser.add_argument("--replicas", type=int, default=REPLICAS_VALIDACION_DEMANDA,
                        help="réplicas de validación por hospital (predeterminado: 49999)")
    parser.add_argument("--trabajadores", type=int, default=TRABAJADORES_VALIDACION_DEMANDA)
    parser.add_argument("--salida", type=Path, default=Path(__file__).resolve().parent,
                        help="carpeta de los CSV; usar otra para pruebas cortas")
    args = parser.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    datos = cargar_demanda_historica()
    if args.etapa in ("validar", "todo"):
        print(f"Validación: {args.replicas:,} réplicas por hospital; puede tardar varias horas.", flush=True)
        tabla, _ = validar_demanda(datos, replicas=args.replicas, trabajadores=args.trabajadores,
                                  progreso=lambda h, _: print(f"Validado: {h}", flush=True))
        tabla.to_csv(args.salida / "validacion.csv", index=False)
    if args.etapa in ("comparar", "todo"):
        capacidades = tabla_hospitales().set_index("nombre").capacidad_maxima.to_dict()
        resumen, _, pares, _ = comparar(datos, cargar_parametros(), capacidades)
        unir_comparacion(resumen, pares).to_csv(args.salida / "comparacion.csv", index=False)
    print(f"Resultados guardados en {args.salida}", flush=True)


if __name__ == "__main__":
    main()
