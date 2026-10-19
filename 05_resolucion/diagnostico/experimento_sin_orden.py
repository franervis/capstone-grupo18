"""Experimento clave: ¿quitar solo la decisión del orden acelera el modelo?

Instancias medianas (8, 9 y 10 hospitales, 3 vehículos, 3 escenarios, mes 9,
gap 1%, 10 min por corrida), las mismas de `medir_mejoras.py`.
Para cada tamaño, compara:
 - modelo común (original);
 - modelo SIN LA DECISIÓN DEL ORDEN (sin_orden.resolver_sin_orden): mantiene A
   (qué hospitales va cada vehículo cada día) y C (cantidades), y usa Held-Karp
   para el orden.
Si el "sin orden" se resuelve rápido, la causa es el orden. Si sigue lento, no.

Antes de eso, verifica en 4 instancias chicas (3 a 6 hospitales, gap 0) que el
modelo sin orden da EXACTAMENTE el mismo óptimo que el modelo común: quitar la
decisión del orden no cambia el problema, solo la forma de resolverlo."""
import sys, pathlib, json, time
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
from codigo import modelo_comun, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from codigo.sin_orden import resolver_sin_orden

SALIDA = RESULTADOS / "experimento_sin_orden.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {}
# ---------- Verificación en instancias chicas (gap 0): mismo óptimo que el modelo común ----------
CHICAS = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, 3),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 3, 2),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, 2),
    (["Hospital del Salvador", "Instituto Neurocirugía", "Instituto Geriatría", "Hospital Calvo Mackenna", "Ex Posta Central", "Instituto Traumatológico"], 3, 1),
]
if "chicas" not in R:
    cfg.GAP_MIP = 0.0
    R["chicas"] = []
    for nombres, nv, ne in CHICAS:
        ins = v._insumos_chicos(nombres, n_vehiculos=nv)
        esc = generar_escenarios(7, 9, ins.distribuciones_demanda, ne, distribuciones_oferta=ins.distribuciones_oferta,
                                 rng=np.random.default_rng(len(nombres)))
        est = estado_inicial_semana(ins)
        comun = modelo_comun.resolver_modelo_calendario(esc, est, ins).info_resolucion
        sin = resolver_sin_orden(esc, est, ins).info_resolucion
        mismo = bool(comun["optimo"] and sin["optimo"]
                     and abs(comun["valor_objetivo"] - sin["valor_objetivo"]) <= 1e-6 * comun["valor_objetivo"] + 1e-3)
        R["chicas"].append({"hospitales": len(nombres), "vehiculos": nv, "escenarios": ne,
                            "obj_comun": comun["valor_objetivo"], "obj_sin_orden": sin["valor_objetivo"], "mismo_optimo": mismo})
        print("chicas", R["chicas"][-1], flush=True)
    json.dump(R, open(SALIDA, "w"), indent=1)
    assert all(f["mismo_optimo"] for f in R["chicas"]), "el modelo sin orden no coincide con el modelo común"
    cfg.GAP_MIP = 0.01

base = v._base()
dist = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()
for n in (8, 9, 10):
    ins = v._insumos_chicos(list(dist.index[:n]), n_vehiculos=3)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    clave = str(n)
    if clave in R:
        continue
    inicio = time.time()
    sol = resolver_sin_orden(esc, est, ins)
    r = sol.info_resolucion
    R[clave] = {"hospitales": n, "tiempo_s": time.time() - inicio, "optimo": r["optimo"], "gap": r["gap"],
                "valor_objetivo": r["valor_objetivo"], "cota": r["cota"], "trabajo": r.get("trabajo"), "visitas": len(sol.visitas), "km": sol.distancia_km}
    json.dump(R, open(SALIDA, "w"), indent=1)
    print(clave, R[clave], flush=True)
    print("pausa 2 min", flush=True); time.sleep(120)
print("FIN")
