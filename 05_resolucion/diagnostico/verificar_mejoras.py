"""Cada mejora exacta debe dar el MISMO óptimo que el modelo común original
(gap 0) en instancias chicas. Si no, la mejora está mal planteada."""
import sys, pathlib, json
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.0
from codigo import modelo_comun, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
import modelo_comun_mejorado as mm

VARIANTES = {
    "simetria_w": dict(simetria="w"),
    "simetria_visitas": dict(simetria="visitas"),
    "desigualdades": dict(desigualdades=True),
    "lazy": dict(subtours="lazy"),
    "todo": dict(simetria="visitas", desigualdades=True, subtours="lazy"),
    "parametros": dict(parametros_gurobi={"Symmetry": 2, "Cuts": 2, "MIPFocus": 3}),
}
casos = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, 3),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 3, 2),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, 2),
    (["Hospital del Salvador", "Instituto Neurocirugía", "Instituto Geriatría", "Hospital Calvo Mackenna", "Ex Posta Central", "Instituto Traumatológico"], 3, 1),
]
filas, problemas = [], 0
for nombres, nv, n_esc in casos:
    ins = v._insumos_chicos(nombres, n_vehiculos=nv)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, n_esc, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(len(nombres)))
    est = estado_inicial_semana(ins)
    ref = modelo_comun.resolver_modelo_calendario(esc, est, ins).info_resolucion
    fila = {"hospitales": len(nombres), "vehiculos": nv, "escenarios": n_esc, "original": ref["valor_objetivo"], "t_original": ref["tiempo_s"]}
    for nombre, op in VARIANTES.items():
        r = mm.resolver(esc, est, ins, **op).info_resolucion
        igual = r["optimo"] and abs(r["valor_objetivo"] - ref["valor_objetivo"]) <= 1e-6 * max(1, ref["valor_objetivo"]) + 1e-4
        problemas += not igual
        fila[nombre] = r["valor_objetivo"]; fila[f"t_{nombre}"] = r["tiempo_s"]; fila[f"igual_{nombre}"] = bool(igual)
    filas.append(fila)
    print({k: (round(vv, 2) if isinstance(vv, float) else vv) for k, vv in fila.items()}, flush=True)
print("PROBLEMAS:", problemas)
json.dump(filas, open(RESULTADOS / "verificar_mejoras.json", "w"), indent=1)
