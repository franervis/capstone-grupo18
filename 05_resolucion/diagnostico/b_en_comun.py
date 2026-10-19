"""Escala completa, semana 1 del año simulado, 30 escenarios.
1) Resuelve el calendario de la B.
2) Construye el modelo común y FIJA sus variables y, w y x a la solución de la B.
   Si los dos modelos son consistentes, el costo debe calzar (ambos con gap ≤ 1%).
3) Le da la solución de la B al modelo común como punto de partida (MIP start) y
   lo deja buscar 30 min: muestra si la acepta y cuánto la puede mejorar, y la
   cota inferior que logra (vale también para la B: mide qué tan lejos queda la B
   del óptimo del problema completo).
4) Relajación lineal (LP) de los tres modelos: qué tan débil es la cota inicial."""
import sys, time, json
import sys, pathlib
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
sys.path.insert(0, str(CARPETA.parents[1]))  # raíz del repo, para importar el paquete codigo
RESULTADOS = CARPETA.parents[1] / "resultados"
from comun import *
from codigo import configuracion as cfg
import gurobipy as gp
OUT = RESULTADOS / "b_en_comun.json"
res = {}
ins, semana = insumos_y_semana(0); esc = escenarios_calendario(ins, semana); est = estado_inicial_semana(ins)

# 1) B
t = time.time(); solB = heuristica_b.resolver_modelo_calendario(esc, est, ins)
res["B"] = {k: solB.info_resolucion[k] for k in ["valor_objetivo", "cota", "gap", "tiempo_s", "optimo"]}
res["B"]["visitas"] = len(solB.visitas); res["B"]["km"] = solB.distancia_km
print("B:", res["B"], flush=True)

# arcos de la B en numeración de nodos del modelo común (0 = banco, i+1 = hospital i)
arcos = set()
for (t_, k), ruta in solB.rutas.items():
    nodos = [0] + [i + 1 for i in ruta] + [0]
    arcos |= {(m, n, k, t_) for m, n in zip(nodos[:-1], nodos[1:])}
salidas = {(t_, k) for (t_, k) in solB.rutas}

def fijar(m, como_inicio=False):
    for v in m.getVars():
        nom = v.VarName; g = nom.split("[")[0]
        if g not in ("y", "w", "x"): continue
        idx = tuple(int(z) for z in nom[nom.index("[")+1:-1].split(","))
        if g == "y": val = 1 if (idx[2], idx[0], idx[1]) in solB.visitas else 0
        elif g == "w": val = 1 if (idx[1], idx[0]) in salidas else 0
        else: val = 1 if idx in arcos else 0
        if como_inicio: v.Start = val
        else: v.LB = v.UB = val

# 2) modelo común con el calendario de la B fijo
m = capturar(modelo_comun.resolver_modelo_calendario, esc, est, ins)
fijar(m); m.Params.OutputFlag = 0; m.Params.MIPGap = 0.01; m.optimize()
res["comun_con_calendario_B"] = {"estado": m.Status, "valor_objetivo": m.ObjVal, "cota": m.ObjBound, "tiempo_s": m.Runtime}
print("común con calendario de B fijo:", res["comun_con_calendario_B"], flush=True)

# 4) relajaciones lineales
for nombre, f in [("comun", modelo_comun.resolver_modelo_calendario), ("B", heuristica_b.resolver_modelo_calendario), ("A_inventario", heuristica_a.resolver_modelo_calendario)]:
    mm = capturar(f, esc, est, ins).relax(); mm.Params.OutputFlag = 0; mm.optimize()
    res[f"LP_{nombre}"] = {"valor": mm.ObjVal, "tiempo_s": mm.Runtime}
    print(f"relajación LP {nombre}:", res[f"LP_{nombre}"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=str)

# 3) modelo común partiendo de la B, 30 min
m = capturar(modelo_comun.resolver_modelo_calendario, esc, est, ins)
fijar(m, como_inicio=True); m.Params.OutputFlag = 0; m.Params.MIPGap = 0.01; m.Params.TimeLimit = 1800; m.optimize()
res["comun_desde_B_30min"] = {"estado": m.Status, "valor_objetivo": m.ObjVal, "cota": m.ObjBound, "gap": m.MIPGap, "tiempo_s": m.Runtime, "trabajo": m.Work}
print("común partiendo de B (30 min):", res["comun_desde_B_30min"], flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("FIN")
