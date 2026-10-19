"""Pruebas de consistencia en instancias chicas (óptimo exacto, gap 0):
el modelo común resuelve un problema más general que la B (zonas fijas) y que
la A (mismo problema, pero decide las visitas sin ver el transporte), así que
su óptimo NUNCA puede ser peor que el de ellas en los mismos escenarios."""
import sys, itertools
import sys, pathlib
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
sys.path.insert(0, str(CARPETA.parents[1]))  # raíz del repo, para importar el paquete codigo
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.0
from codigo import modelo_comun, heuristica_a, heuristica_b, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
casos = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, [[0, 2], [1]]),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 2, [[0, 1], [2, 3]]),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, [[0, 4], [1, 2, 3]]),
]
problemas = 0
filas = []
for nombres, nv, zonas in casos:
    ins = v._insumos_chicos(nombres, n_vehiculos=nv)
    for semilla, mes, n_esc in [(1, 9, 1), (2, 3, 3), (3, 10, 2)]:
        esc = generar_escenarios(7, mes, ins.distribuciones_demanda, n_esc, distribuciones_oferta=ins.distribuciones_oferta,
                                 oferta_hoy=None, rng=np.random.default_rng(semilla))
        est = estado_inicial_semana(ins)
        c = modelo_comun.resolver_modelo_calendario(esc, est, ins)
        b = heuristica_b.crear_resolver_calendario(zonas)(esc, est, ins)
        a = heuristica_a.resolver_modelo_calendario(esc, est, ins)
        oc, ob, oa = c.info_resolucion["valor_objetivo"], b.info_resolucion["valor_objetivo"], a.info_resolucion["valor_objetivo"]
        ok = c.info_resolucion["optimo"] and b.info_resolucion["optimo"] and oc <= ob + 1e-6 and oc <= oa + 1e-6
        problemas += not ok
        filas.append({"hospitales": len(nombres), "vehiculos": nv, "escenarios": n_esc, "obj_comun": oc, "obj_B": ob, "obj_A": oa, "comun_menor_o_igual": bool(ok), "t_comun": c.info_resolucion['tiempo_s'], "t_B": b.info_resolucion['tiempo_s']})
        print(f"{len(nombres)} hosp, {nv} veh, {n_esc} esc: común {oc:>12,.0f} | B {ob:>12,.0f} | A {oa:>12,.0f} | común ≤ B y ≤ A: {ok} | t común {c.info_resolucion['tiempo_s']:.2f}s B {b.info_resolucion['tiempo_s']:.2f}s", flush=True)
print("PROBLEMAS:", problemas)
import json
json.dump(filas, open(RESULTADOS / "chicas.json", "w"), indent=1)
