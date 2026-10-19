"""Cómo crece el tiempo del modelo común (calendario) con el número de hospitales,
frente a la B (zonas fijas), con 3 vehículos, 3 escenarios, gap 1% y 10 min de límite."""
import sys, time, json
import sys, pathlib
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
sys.path.insert(0, str(CARPETA.parents[1]))  # raíz del repo, para importar el paquete codigo
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
from codigo import modelo_comun, heuristica_b, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
base = v._base()
# hospitales ordenados por distancia al banco, para que cada instancia sea un "barrio" coherente
d = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()
filas = []
for n in range(3, 11):
    nombres = list(d.index[:n])
    ins = v._insumos_chicos(nombres, n_vehiculos=3)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    m = modelo_comun.resolver_modelo_calendario(esc, est, ins).info_resolucion
    zonas = [list(range(n))[k::3] for k in range(3)]
    b = heuristica_b.crear_resolver_calendario(zonas)(esc, est, ins).info_resolucion
    fila = {"hospitales": n, "t_comun": round(m["tiempo_s"], 2), "gap_comun": round(m["gap"], 4), "optimo_comun": m["optimo"],
            "t_B": round(b["tiempo_s"], 2), "obj_comun": round(m["valor_objetivo"]), "obj_B": round(b["valor_objetivo"])}
    filas.append(fila); print(fila, flush=True)
json.dump(filas, open(RESULTADOS / "escalamiento.json", "w"), indent=1)
