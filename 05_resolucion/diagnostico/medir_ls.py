"""(l,S) en las instancias medianas: las mismas de `medir_mejoras.py` (8, 9 y 10
hospitales más cercanos al banco, 3 vehículos, 3 escenarios, mes 9, gap 1%).
Para cada una: (1) rondas de cortes (l,S) sobre la estimación, (2) modelo entero
con esos cortes. Límite de 10 min EN TOTAL (cortes + entero). Variantes: (l,S)
solas y (l,S) + desigualdades. El original y las desigualdades solas ya están
en resultados/medir_mejoras.json. Guarda cada resultado al terminar y sigue
donde quedó si se corta; 5 min de pausa entre corridas."""
import sys, pathlib, json, time, functools
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
from codigo import modelo_comun, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from comun import capturar
import modelo_comun_mejorado as mm, cortes_ls

LIMITE = 600
SALIDA = RESULTADOS / "medir_ls.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {}
base = v._base()
dist = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()
for n in (8, 9, 10):
    ins = v._insumos_chicos(list(dist.index[:n]), n_vehiculos=3)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    L = ins.parametros["shelf_life"]; D = modelo_comun._demanda_por_hospital(esc, ins)
    for nombre, op in [("ls", {}), ("ls+desigualdades", dict(desigualdades=True))]:
        clave = f"{n}_{nombre}"
        if clave in R: continue
        inicio = time.time()
        lp = capturar(functools.partial(mm.resolver, **op), esc, est, ins).relax(); lp.Params.OutputFlag = 0
        lp.optimize(); cota_sin = lp.ObjVal
        cota_con, cortes, n_rondas = cortes_ls.rondas(lp, D, n, esc.dias, 3, L)
        t_cortes = time.time() - inicio
        ent = capturar(functools.partial(mm.resolver, **op), esc, est, ins)
        cortes_ls.agregar(ent, cortes, D, L)
        ent.Params.OutputFlag = 0; ent.Params.MIPGap = 0.01; ent.Params.TimeLimit = max(1, LIMITE - t_cortes)
        ent.optimize()
        R[clave] = {"cortes": len(cortes), "rondas": n_rondas, "t_cortes": t_cortes, "estimacion_sin_ls": cota_sin, "estimacion_con_ls": cota_con,
                    "t_total": t_cortes + ent.Runtime, "optimo": ent.Status == 2, "gap": ent.MIPGap, "valor_objetivo": ent.ObjVal,
                    "cota": ent.ObjBound, "trabajo": ent.Work}
        json.dump(R, open(SALIDA, "w"), indent=1); print(clave, R[clave], flush=True)
        print("pausa 5 min", flush=True); time.sleep(300)
print("FIN")
