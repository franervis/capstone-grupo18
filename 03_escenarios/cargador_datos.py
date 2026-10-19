"""Carga de los archivos de `datos/`.

Hay una función por fuente de datos. Sólo se hace tipado (fechas, enteros)
y limpieza mínima: se quita la columna de índice sin nombre y se reparan los
tildes, que vienen mal codificados en los CSV (p. ej. "San JosÃ©" → "San José").
Los archivos originales no se modifican.
"""

import pandas as pd

from codigo import configuracion as cfg


def _leer_csv(clave, **kwargs):
    """Lee el CSV asociado a `clave` en `cfg.ARCHIVOS` desde `cfg.RUTA_DATOS`."""
    ruta = cfg.RUTA_DATOS / cfg.ARCHIVOS[clave]
    return pd.read_csv(ruta, encoding=cfg.CODIFICACION_ARCHIVOS, **kwargs)


def reparar_tildes(texto):
    """Repara un texto UTF-8 que fue leído como Latin-1 ("JosÃ©" → "José").

    Recibe: un string.
    Devuelve: el string reparado; si no tenía el problema, lo devuelve igual.
    """
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def _reparar_columnas_texto(df, columnas):
    for columna in columnas:
        df[columna] = df[columna].map(reparar_tildes)
    return df


def cargar_demanda_historica():
    """Demanda diaria por hospital, 2022–2024.

    Devuelve: DataFrame con columnas
        fecha (datetime), hospital (str), tamano (str), demanda (int).
    Una fila por (fecha, hospital): 1.096 días × 22 hospitales.
    """
    df = _leer_csv("demanda_historica")
    df = _reparar_columnas_texto(df, ["hospital", "tamano"])
    df["fecha"] = pd.to_datetime(df["fecha"], format=cfg.FORMATO_FECHA)
    df["demanda"] = df["demanda"].astype(int)
    return df


def cargar_demanda_red_diaria():
    """Demanda total diaria de la red (suma de los 22 hospitales).

    Devuelve: DataFrame con columnas fecha (datetime), demanda_total_red (int).
    """
    df = _leer_csv("demanda_red_diaria")
    df["fecha"] = pd.to_datetime(df["fecha"], format=cfg.FORMATO_FECHA)
    df["demanda_total_red"] = df["demanda_total_red"].astype(int)
    return df


def cargar_oferta_historica():
    """Oferta diaria disponible en el Centro Metropolitano, 2022–2024.

    Devuelve: DataFrame con columnas fecha (datetime), oferta (int),
    ordenado por fecha.
    """
    df = _leer_csv("oferta_historica", index_col=0)
    df["fecha"] = pd.to_datetime(df["fecha"], format=cfg.FORMATO_FECHA)
    df["oferta"] = df["oferta"].astype(int)
    return df.sort_values("fecha").reset_index(drop=True)


def cargar_hospitales():
    """Información de los nodos de la red (Centro Metropolitano + 22 hospitales).

    Devuelve: DataFrame con columnas
        nombre, lat, lon, tipo ("Banco" u "Hospital"), tamano, capacidad_maxima.
    La fila 0 es el Centro Metropolitano.
    OJO: la columna `capacidad_maxima` de este archivo está desactualizada.
    Las capacidades vigentes están en `capacidades_hospitales.csv`
    (ver `cargar_capacidades_hospitales` y `tabla_hospitales`).
    """
    df = _leer_csv("hospitales")
    return _reparar_columnas_texto(df, ["nombre", "tipo", "tamano"])


def cargar_capacidades_hospitales():
    """Capacidades de almacenamiento vigentes, por posición.

    Devuelve: Series indexada por posición (0..22), en el mismo orden que
    `hospitales.csv`. La posición 0 corresponde al Centro Metropolitano
    (ver `capacidad_banco`).
    """
    df = _leer_csv("capacidades_hospitales", index_col=0)
    return df["capacidad_maxima"].astype(int)


def tabla_hospitales():
    """Los 22 hospitales con su capacidad vigente.

    Une `hospitales.csv` (nombre, ubicación, tamaño) con
    `capacidades_hospitales.csv` (capacidad) por posición, y deja sólo las
    filas de tipo "Hospital". La capacidad del Centro Metropolitano no va
    aquí: ver `capacidad_banco`.

    Devuelve: DataFrame con columnas nombre, lat, lon, tamano,
    capacidad_maxima (int), indexado de 0 a 21.
    """
    hospitales = cargar_hospitales()
    capacidades = cargar_capacidades_hospitales()
    if len(hospitales) != len(capacidades):
        raise ValueError(
            f"hospitales.csv tiene {len(hospitales)} filas y "
            f"capacidades_hospitales.csv tiene {len(capacidades)}; deben coincidir."
        )
    hospitales["capacidad_maxima"] = capacidades.to_numpy()
    solo_hospitales = hospitales[hospitales["tipo"] == "Hospital"]
    columnas = ["nombre", "lat", "lon", "tamano", "capacidad_maxima"]
    return solo_hospitales[columnas].reset_index(drop=True)


def capacidad_banco():
    """Capacidad de almacenamiento del Centro Metropolitano: la fila 0 de
    `capacidades_hospitales.csv` (1200, igual que el Informe 1, §2.1 y §3.4).

    Decisión del grupo (2026-10-04): se usa este valor y NO
    `capacidad_banco` de `parametros.csv` (1000), que está desactualizado.
    """
    return int(cargar_capacidades_hospitales().iloc[0])


def cargar_flota():
    """Vehículos de la flota y su capacidad.

    Devuelve: DataFrame con columnas vehiculo (int, 0..4), capacidad (int).
    """
    df = _leer_csv("flota", index_col=0)
    df.index.name = "vehiculo"
    df["capacidad"] = df["capacidad"].astype(int)
    return df.reset_index()


def cargar_distancias():
    """Matriz de distancias viales en km entre el Centro y los 22 hospitales.

    Fuente: transcripción de la Tabla A.4 del Informe 1 (se calculó sobre la
    red vial de chile.gpkg). No viene en los datos del enunciado.

    Devuelve: DataFrame cuadrado 23×23 indexado por nombre en filas (origen)
    y columnas (destino), en el mismo orden que `hospitales.csv`.
    """
    return _leer_csv("distancias", index_col=0)


def cargar_parametros():
    """Parámetros generales del problema (horizonte, vida útil, costos, etc.).

    Devuelve: diccionario {nombre_parametro: valor}. Los valores enteros
    se devuelven como int; el resto, como float.
    """
    df = _leer_csv("parametros")
    parametros = {}
    for nombre, valor in zip(df["parametro"], df["valor"]):
        valor = float(valor)
        parametros[nombre] = int(valor) if valor.is_integer() else valor
    return parametros
