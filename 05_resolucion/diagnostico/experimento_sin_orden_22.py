"""Experimento sin orden a ESCALA COMPLETA (22 hospitales, 5 vehículos,
30 escenarios, semana 1 del año simulado, gap 1%, 1 h de límite).

Mismo enfoque que `experimento_sin_orden.py` pero con los insumos reales de la
instancia completa, para ver si quitarle la decisión del orden al modelo común
alcanza con los 22 hospitales. Guarda el resultado y, si Held-Karp sobre los
2^22 grupos o el modelo con ~4 M · 5 · 7 variables z no cabe, deja el error en
el .log y en el .json para discutirlo."""
import sys, pathlib, json, time, traceback
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 3600
from comun import insumos_y_semana, escenarios_calendario
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.sin_orden import resolver_sin_orden

SALIDA = RESULTADOS / "experimento_sin_orden_22.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {}
if "22" not in R:
    try:
        ins, semana = insumos_y_semana(0)
        esc = escenarios_calendario(ins, semana)
        est = estado_inicial_semana(ins)
        inicio = time.time()
        sol = resolver_sin_orden(esc, est, ins)
        r = sol.info_resolucion
        R["22"] = {"hospitales": 22, "tiempo_s": time.time() - inicio, "optimo": r["optimo"],
                   "gap": r["gap"], "valor_objetivo": r["valor_objetivo"], "cota": r["cota"],
                   "trabajo": r.get("trabajo"), "visitas": len(sol.visitas), "km": sol.distancia_km}
    except Exception as e:
        R["22"] = {"hospitales": 22, "error": repr(e), "traceback": traceback.format_exc()}
    json.dump(R, open(SALIDA, "w"), indent=1)
    print(json.dumps(R["22"], indent=1), flush=True)
print("FIN")
