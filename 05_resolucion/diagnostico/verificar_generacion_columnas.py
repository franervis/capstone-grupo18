"""Verificación de la generación de columnas (codigo/generacion_columnas.py) en
instancias chicas resueltas al óptimo exacto (gap 0). Para cada instancia:
1. Maestro por rutas con TODAS las rutas: ¿da el mismo óptimo que el modelo común?
2. Cota continua: arcos (modelo común) contra rutas, y su distancia al óptimo.
3. Ciclo de generación de columnas: ¿su cota es igual a la del maestro con todas las rutas?
4. Solución entera con las rutas generadas (price-and-branch): ¿cuánto se pierde?"""
import sys, pathlib, json, itertools, time
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.0
from codigo import modelo_comun, verificacion as v, generacion_columnas as gc
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from comun import capturar

casos = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, 3),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 3, 2),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, 2),
    (["Hospital del Salvador", "Instituto Neurocirugía", "Instituto Geriatría", "Hospital Calvo Mackenna", "Ex Posta Central", "Instituto Traumatológico"], 3, 1),
]
filas = []
for nombres, nv, ne in casos:
    ins = v._insumos_chicos(nombres, n_vehiculos=nv)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, ne, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(len(nombres)))
    est = estado_inicial_semana(ins)
    comun = modelo_comun.resolver_modelo_calendario(esc, est, ins).info_resolucion
    optimo = comun["valor_objetivo"]
    todas = [c for n in range(1, len(nombres) + 1) for c in itertools.combinations(range(len(nombres)), n)]
    t = time.time(); rutas_todas = gc.resolver_maestro(esc, est, ins, todas)[0].info_resolucion; t_todas = time.time() - t
    lp_arcos = capturar(modelo_comun.resolver_modelo_calendario, esc, est, ins).relax(); lp_arcos.Params.OutputFlag = 0; lp_arcos.optimize()
    _, lp_rutas, _ = gc.resolver_maestro(esc, est, ins, todas, relajado=True)
    cg = gc.generar_columnas(esc, est, ins)
    entero_cg = gc.resolver_maestro(esc, est, ins, cg["rutas"])[0].info_resolucion["valor_objetivo"]
    fila = {
        "hospitales": len(nombres), "vehiculos": nv, "escenarios": ne, "rutas_posibles": len(todas),
        "optimo_comun": optimo, "t_comun": comun["tiempo_s"],
        "optimo_rutas_todas": rutas_todas["valor_objetivo"], "t_rutas_todas": t_todas,
        "mismo_optimo": abs(rutas_todas["valor_objetivo"] - optimo) <= 1e-6 * optimo + 1e-3,
        "cota_arcos": lp_arcos.ObjVal, "distancia_cota_arcos_pct": 100 * (optimo - lp_arcos.ObjVal) / optimo,
        "cota_rutas": lp_rutas.ObjVal, "distancia_cota_rutas_pct": 100 * (optimo - lp_rutas.ObjVal) / optimo,
        "cota_cg": cg["valor_continuo"], "cota_cg_igual_a_todas": abs(cg["valor_continuo"] - lp_rutas.ObjVal) <= 1e-5 * lp_rutas.ObjVal + 1e-3,
        "iteraciones_cg": cg["iteraciones"], "rutas_generadas": len(cg["rutas"]),
        "entero_cg": entero_cg, "perdida_entero_cg_pct": 100 * (entero_cg - optimo) / optimo,
    }
    filas.append(fila); print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in fila.items()}, flush=True)
json.dump(filas, open(RESULTADOS / "verificar_generacion_columnas.json", "w"), indent=1)
