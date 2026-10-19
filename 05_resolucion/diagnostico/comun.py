"""Herramientas comunes del análisis: arma los escenarios de la semana 1 del año
simulado y permite capturar el modelo de Gurobi que construye cada política sin
resolverlo (reemplaza temporalmente modelo_comun._resolver)."""
import numpy as np, pandas as pd
from codigo import modelo_comun, heuristica_a, heuristica_b
from codigo.horizonte_rodante import preparar_insumos, estado_inicial_semana
from codigo.simulacion import cargar_semanas, generador_modelo_semana
from codigo.escenarios import generar_escenarios

class Capturado(Exception):
    pass

def insumos_y_semana(indice=0):
    ins = preparar_insumos()
    semanas = cargar_semanas("01_datos/semanas_simuladas_consecutivas_2024.csv", ins)
    return ins, [s for s in semanas if s.indice == indice][0]

def escenarios_calendario(ins, semana, n=30, semilla=2024):
    rng = generador_modelo_semana(semilla, semana.indice)
    return generar_escenarios(7, semana.fecha_lunes.month, ins.distribuciones_demanda, n,
                              distribuciones_oferta=ins.distribuciones_oferta,
                              oferta_hoy=int(semana.oferta[0]), rng=rng)

def capturar(funcion, *args):
    """Corre `funcion` hasta que llama a _resolver y devuelve el modelo sin resolver."""
    guardado = {}
    original = modelo_comun._resolver
    def falso(modelo):
        modelo.update(); guardado["m"] = modelo; raise Capturado
    modelo_comun._resolver = falso
    try:
        funcion(*args)
    except Capturado:
        pass
    finally:
        modelo_comun._resolver = original
    return guardado["m"]

def tamano(m):
    vs = m.getVars()
    bins = m.NumBinVars; ints = m.NumIntVars - bins
    return {"variables": m.NumVars, "binarias": bins, "enteras": ints, "continuas": m.NumVars - m.NumIntVars,
            "restricciones": m.NumConstrs, "no_ceros": m.NumNZs}

def por_grupo(m):
    """Cuenta variables por nombre de grupo (y, x, w, rho, q, r, u, ...)."""
    c = {}
    for v in m.getVars():
        g = v.VarName.split("[")[0]
        c[g] = c.get(g, 0) + 1
    return c
