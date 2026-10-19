"""Verificación de la cota lagrangiana (codigo/relajacion_lagrangiana.py) en las
instancias chicas: debe quedar POR DEBAJO del óptimo (si no, hay un error) y,
ojalá, más cerca de él que la estimación lineal del modelo común."""
import sys, pathlib, json
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.0
from codigo import modelo_comun, verificacion as v, relajacion_lagrangiana as rl
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from comun import capturar

casos = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, 3),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 3, 2),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, 2),
    (["Hospital del Salvador", "Instituto Neurocirugía", "Instituto Geriatría", "Hospital Calvo Mackenna", "Ex Posta Central", "Instituto Traumatológico"], 3, 1),
]
iteraciones = int(sys.argv[1]) if len(sys.argv) > 1 else 300
filas = []
for nombres, nv, ne in casos:
    ins = v._insumos_chicos(nombres, n_vehiculos=nv)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, ne, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(len(nombres)))
    est = estado_inicial_semana(ins)
    optimo = modelo_comun.resolver_modelo_calendario(esc, est, ins).info_resolucion["valor_objetivo"]
    lp = capturar(modelo_comun.resolver_modelo_calendario, esc, est, ins).relax(); lp.Params.OutputFlag = 0; lp.optimize()
    r = rl.cota_lagrangiana(esc, est, ins, cota_superior=optimo, iteraciones=iteraciones)
    fila = {"hospitales": len(nombres), "vehiculos": nv, "escenarios": ne, "optimo": optimo,
            "cota_lineal": lp.ObjVal, "distancia_lineal_pct": 100 * (optimo - lp.ObjVal) / optimo,
            "cota_lagrangiana": r["cota"], "distancia_lagrangiana_pct": 100 * (optimo - r["cota"]) / optimo,
            "valida": r["cota"] <= optimo * (1 + 1e-6) + 1e-3, "iteraciones": r["iteraciones"], "tiempo_s": r["tiempo_s"]}
    filas.append(fila)
    print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in fila.items()}, flush=True)
json.dump(filas, open(RESULTADOS / "verificar_lagrangiana.json", "w"), indent=1)
