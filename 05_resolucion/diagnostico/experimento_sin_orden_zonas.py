"""Compara `sin_orden_zonas` contra el óptimo del modelo común libre, en
instancias chicas (gap 0) y medianas (gap 1%, 10 min). Mide la pérdida de
restringir a las zonas de la heurística B (`particion_compacta`).

Chicas: 3, 4 y 5 hospitales con 2 vehículos, zonas a mano (las de
`scripts/chicas.py`), 3 semillas/meses de `chicas.py`. Compara con el óptimo
del modelo común libre (ambos a gap 0).

Medianas: 8, 9 y 10 hospitales con 3 vehículos, mes 9, 3 escenarios, mismas
instancias de `experimento_sin_orden.py`. Zonas vienen de `particion_compacta`.
Como el modelo común libre no llega al óptimo en 10 min (gap 4,6% con 9-10),
compara con el modelo sin orden libre (que SÍ llega al óptimo en 7-47 s;
resultados en `experimento_sin_orden.json`).

Guarda todo en `resultados/experimento_sin_orden_zonas.json`."""
import sys, pathlib, json, time
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
from codigo import verificacion as v, modelo_comun, heuristica_b
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from codigo.sin_orden_zonas import resolver_sin_orden_zonas
from codigo.sin_orden import resolver_sin_orden

SALIDA = RESULTADOS / "experimento_sin_orden_zonas.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {"chicas": [], "medianas": []}

# ---------- Chicas (gap 0, comparando con óptimo del modelo común libre) ----------
cfg.GAP_MIP = 0.0; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
casos_chicos = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, [[0, 2], [1]]),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 2, [[0, 1], [2, 3]]),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, [[0, 4], [1, 2, 3]]),
]
hechas_chicas = {(f["hospitales"], f["semilla"], f["mes"], f["escenarios"]) for f in R["chicas"]}
for nombres, nv, zonas in casos_chicos:
    ins = v._insumos_chicos(nombres, n_vehiculos=nv)
    for semilla, mes, n_esc in [(1, 9, 1), (2, 3, 3), (3, 10, 2)]:
        clave = (len(nombres), semilla, mes, n_esc)
        if clave in hechas_chicas:
            continue
        esc = generar_escenarios(7, mes, ins.distribuciones_demanda, n_esc, distribuciones_oferta=ins.distribuciones_oferta,
                                 oferta_hoy=None, rng=np.random.default_rng(semilla))
        est = estado_inicial_semana(ins)
        c = modelo_comun.resolver_modelo_calendario(esc, est, ins)
        z = resolver_sin_orden_zonas(esc, est, ins, zonas)
        oc, oz = c.info_resolucion["valor_objetivo"], z.info_resolucion["valor_objetivo"]
        R["chicas"].append({
            "hospitales": len(nombres), "vehiculos": nv, "semilla": semilla, "mes": mes, "escenarios": n_esc,
            "obj_comun": oc, "obj_zonas": oz, "gap_zonas_vs_comun": (oz - oc) / oc if oc > 0 else 0.0,
            "optimo_comun": c.info_resolucion["optimo"], "optimo_zonas": z.info_resolucion["optimo"],
            "t_comun": c.info_resolucion["tiempo_s"], "t_zonas": z.info_resolucion["tiempo_s"],
        })
        json.dump(R, open(SALIDA, "w"), indent=1)
        print("chicas", R["chicas"][-1], flush=True)

# ---------- Medianas (gap 1%, comparando con óptimo del sin_orden libre) ----------
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
libres = json.load(open(RESULTADOS / "experimento_sin_orden.json"))  # 8, 9, 10 ya corridos
base = v._base()
dist = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()
hechas_medianas = {f["hospitales"] for f in R["medianas"]}
for n in (8, 9, 10):
    if n in hechas_medianas:
        continue
    ins = v._insumos_chicos(list(dist.index[:n]), n_vehiculos=3)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta,
                             rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    zonas = heuristica_b.particion_compacta(ins)
    inicio = time.time()
    sol = resolver_sin_orden_zonas(esc, est, ins, zonas)
    r = sol.info_resolucion
    libre = libres[str(n)]
    R["medianas"].append({
        "hospitales": n, "zonas": zonas, "tam_zonas": [len(z) for z in zonas],
        "tiempo_s": time.time() - inicio, "optimo_zonas": r["optimo"], "gap_mip": r["gap"],
        "valor_zonas": r["valor_objetivo"], "cota_zonas": r["cota"],
        "valor_sin_orden_libre": libre["valor_objetivo"], "tiempo_sin_orden_libre": libre["tiempo_s"],
        "gap_zonas_vs_libre": (r["valor_objetivo"] - libre["valor_objetivo"]) / libre["valor_objetivo"],
        "visitas": len(sol.visitas), "km": sol.distancia_km,
    })
    json.dump(R, open(SALIDA, "w"), indent=1)
    print("medianas", R["medianas"][-1], flush=True)
    print("pausa 1 min", flush=True); time.sleep(60)

print("FIN")
