"""Desigualdades (l,S) de planificación de producción (`resultados_generacion_columnas.md` §6.3),
probadas en las 4 instancias chicas.

Para cada hospital i, escenario s, día l y conjunto de días S ⊆ {0..l}:
    Σ_{t∈S} llega[i,t,s] ≤ Σ_{t∈S} D[i,t..l,s] · Y[i,t] + inventario al cierre de l + vencidas antes de l
con llega[i,t,s] = Σ_{k,a} q[i,k,a,t,s] e Y[i,t] = Σ_k y[i,k,t].
Es exacta: lo que llega en S se consume (a lo más la demanda de t..l, y solo si se
visitó), queda guardado al cierre de l, o se botó antes. Se agregan solo las violadas
por la estimación actual (rondas de cortes). Mide: (1) que el óptimo no cambie y
(2) cuánto se acerca la estimación al óptimo."""
import sys, pathlib, json
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np, gurobipy as gp
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.0
from codigo import modelo_comun, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from comun import capturar
import functools, modelo_comun_mejorado as mm

def indices(nombre):
    return tuple(int(z) for z in nombre[nombre.index("[") + 1:-1].split(","))

def mapa(m):
    q, y, V = {}, {}, {}
    for var in m.getVars():
        g = var.VarName.split("[")[0]
        if g == "q": i, k, a, t, s = indices(var.VarName); q.setdefault((i, t, s), []).append(var)
        elif g == "y": i, k, t = indices(var.VarName); y.setdefault((i, t), []).append(var)
        elif g == "V_cierre": i, a, t, s = indices(var.VarName); V[i, a, t, s] = var
    return q, y, V

def expr_corte(q, y, V, D, i, s, l, S, L):
    """lado izquierdo − lado derecho (≤ 0 es la desigualdad)."""
    lhs = gp.quicksum(q[i, t, s]) if False else gp.quicksum(v for t in S for v in q.get((i, t, s), []))
    dem = lambda t: float(D[s, t:l + 1, i].sum())
    rhs = (gp.quicksum(dem(t) * yy for t in S for yy in y.get((i, t), []))
           + gp.quicksum(V[i, a, l, s] for a in range(L))
           + gp.quicksum(V[i, L - 1, t, s] for t in range(l)))
    return lhs - rhs

def rondas_de_cortes(m, D, n_h, T, S_esc, L, max_rondas=50):
    """Sobre el modelo relajado m: agrega las (l,S) violadas hasta que no haya más."""
    q, y, V = mapa(m); cortes = []
    for ronda in range(max_rondas):
        m.optimize()
        nuevos = 0
        for i in range(n_h):
            for s in range(S_esc):
                llega = {t: sum(v.X for v in q.get((i, t, s), [])) for t in range(T)}
                Y = {t: sum(v.X for v in y.get((i, t), [])) for t in range(T)}
                for l in range(T):
                    S = [t for t in range(l + 1) if llega[t] > float(D[s, t:l + 1, i].sum()) * Y[t] + 1e-6]
                    if not S: continue
                    viol = sum(llega[t] - float(D[s, t:l + 1, i].sum()) * Y[t] for t in S) \
                           - sum(V[i, a, l, s].X for a in range(L)) - sum(V[i, L - 1, t, s].X for t in range(l))
                    if viol > 1e-4:
                        m.addConstr(expr_corte(q, y, V, D, i, s, l, S, L) <= 0); cortes.append((i, s, l, tuple(S))); nuevos += 1
        if nuevos == 0: break
    m.optimize()
    return m.ObjVal, cortes, ronda + 1

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
    L = ins.parametros["shelf_life"]; D = modelo_comun._demanda_por_hospital(esc, ins)
    optimo = mm.resolver(esc, est, ins).info_resolucion["valor_objetivo"]
    fila = {"hospitales": len(nombres), "vehiculos": nv, "escenarios": ne, "optimo": optimo}
    for nombre, op in [("original", {}), ("desigualdades", dict(desigualdades=True))]:
        base = capturar(functools.partial(mm.resolver, **op), esc, est, ins)
        lp0 = base.relax(); lp0.Params.OutputFlag = 0; lp0.optimize()
        lp = base.relax(); lp.Params.OutputFlag = 0
        cota_ls, cortes, rondas = rondas_de_cortes(lp, D, len(nombres), esc.dias, ne, L)
        # exactitud: el modelo entero con esos cortes debe dar el mismo óptimo
        ent = capturar(functools.partial(mm.resolver, **op), esc, est, ins); ent.Params.OutputFlag = 0; ent.Params.MIPGap = 0
        q, y, V = mapa(ent)
        for (i, s, l, S) in cortes: ent.addConstr(expr_corte(q, y, V, D, i, s, l, S, L) <= 0)
        ent.optimize()
        d = lambda c: 100 * (optimo - c) / optimo
        fila[nombre] = {"distancia_sin_ls_pct": d(lp0.ObjVal), "distancia_con_ls_pct": d(cota_ls), "cortes": len(cortes), "rondas": rondas,
                        "optimo_con_cortes": ent.ObjVal, "mismo_optimo": abs(ent.ObjVal - optimo) <= 1e-6 * optimo + 1e-3,
                        "t_entero_con_cortes": ent.Runtime}
    filas.append(fila)
    print(json.dumps({k: (round(x, 2) if isinstance(x, float) else x) for k, x in fila.items()}, default=lambda o: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in o.items()}), flush=True)
json.dump(filas, open(RESULTADOS / "verificar_ls.json", "w"), indent=1)
