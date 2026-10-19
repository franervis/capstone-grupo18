"""Barrido de `sin_orden_zonas` desde 11 hasta 22 hospitales con la cantidad
REAL de vehículos (5), 3 escenarios, mes 9, semilla 7, 10 min de límite y
gap 1%. Toma los n hospitales más cercanos al banco. Guarda en
`experimento_sin_orden_zonas.json` bajo la clave `barrido_11_22`.

Para cada n: usa `particion_compacta` para las zonas y resuelve
`sin_orden_zonas`. NO corre el modelo común libre (si falta la referencia para
algún n, se corre después aparte)."""
import sys, pathlib, json, time
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
from codigo import verificacion as v, modelo_comun, heuristica_b
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from codigo.sin_orden_zonas import resolver_sin_orden_zonas

SALIDA = RESULTADOS / "experimento_sin_orden_zonas.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {"chicas": [], "medianas": []}
R.setdefault("barrido_11_22", [])
hechas = {f["hospitales"] for f in R["barrido_11_22"]}
base = v._base()
dist = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()

for n in range(11, 23):
    if n in hechas:
        continue
    ins = v._insumos_chicos(list(dist.index[:n]), n_vehiculos=5)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta,
                             rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    try:
        zonas = heuristica_b.particion_compacta(ins)
    except Exception as e:
        R["barrido_11_22"].append({"hospitales": n, "error_zonas": repr(e)})
        json.dump(R, open(SALIDA, "w"), indent=1); continue
    inicio = time.time()
    sol = resolver_sin_orden_zonas(esc, est, ins, zonas)
    rz = sol.info_resolucion
    t_zonas = time.time() - inicio
    fila = {
        "hospitales": n, "zonas": zonas, "tam_zonas": [len(z) for z in zonas],
        "t_zonas": t_zonas, "optimo_zonas": rz["optimo"], "valor_zonas": rz["valor_objetivo"], "cota_zonas": rz["cota"],
        "visitas": len(sol.visitas), "km": sol.distancia_km,
    }
    R["barrido_11_22"].append(fila)
    json.dump(R, open(SALIDA, "w"), indent=1)
    print(n, fila, flush=True)
print("FIN")
