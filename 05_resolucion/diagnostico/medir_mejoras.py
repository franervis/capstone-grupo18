"""Instancias medianas (8, 9 y 10 hospitales más cercanos al banco, 3 vehículos,
3 escenarios, mes 9, gap 1%, 10 min de límite): tiempo, gap, cota y trabajo de
cada mejora exacta y de cada variante del análisis por partes. Guarda cada
resultado apenas termina y, si se corta, sigue desde donde quedó."""
import sys, pathlib, json, os
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
from codigo import heuristica_b, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
import modelo_comun_mejorado as mm

P = {"Symmetry": 2, "Cuts": 2, "MIPFocus": 3}
MEJORAS = {
    "original": {},
    "simetria_w": dict(simetria="w"),
    "simetria_visitas": dict(simetria="visitas"),
    "desigualdades": dict(desigualdades=True),
    "lazy": dict(subtours="lazy"),
    "parametros": dict(parametros_gurobi=P),
    "simetria+desigualdades": dict(simetria="visitas", desigualdades=True),
    "todo": dict(simetria="visitas", desigualdades=True, subtours="lazy"),
    "todo+parametros": dict(simetria="visitas", desigualdades=True, subtours="lazy", parametros_gurobi=P),
}
SALIDA = RESULTADOS / "medir_mejoras.json"
# Uso: python medir_mejoras.py [N]  → corre como máximo N corridas nuevas y se detiene
# (para repartir el trabajo en tandas y no sobrecalentar el computador).
MAXIMO = int(sys.argv[1]) if len(sys.argv) > 1 else None
nuevas = 0
filas = json.load(open(SALIDA)) if SALIDA.exists() else []
hechas = {(f["hospitales"], f["variante"]) for f in filas}
base = v._base()
d = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()

def guardar(fila):
    filas.append(fila); json.dump(filas, open(SALIDA, "w"), indent=1)
    print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in fila.items()}, flush=True)

for n in (8, 9, 10):
    ins = v._insumos_chicos(list(d.index[:n]), n_vehiculos=3)
    esc3 = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    variantes = dict(MEJORAS)
    if n == 9:  # análisis por partes: cambian el problema, solo para medir qué pesa
        variantes["partes_1_escenario"] = "1esc"
        variantes["partes_cantidades_continuas"] = dict(cantidades_continuas=True)
        variantes["partes_zonas_fijas"] = "zonas"
    for nombre, op in variantes.items():
        if (n, nombre) in hechas:
            continue
        if MAXIMO is not None and nuevas >= MAXIMO:
            print(f"Tanda terminada ({MAXIMO} corridas).")
            sys.exit(0)
        nuevas += 1
        if op == "1esc":
            esc1 = generar_escenarios(7, 9, ins.distribuciones_demanda, 1, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
            r = mm.resolver(esc1, est, ins).info_resolucion
        elif op == "zonas":
            r = heuristica_b.crear_resolver_calendario(heuristica_b.particion_compacta(ins))(esc3, est, ins).info_resolucion
        else:
            r = mm.resolver(esc3, est, ins, **op).info_resolucion
        guardar({"hospitales": n, "variante": nombre, "tiempo_s": r["tiempo_s"], "optimo": r["optimo"], "gap": r["gap"],
                 "valor_objetivo": r["valor_objetivo"], "cota": r["cota"], "trabajo": r.get("trabajo")})
print("FIN")
