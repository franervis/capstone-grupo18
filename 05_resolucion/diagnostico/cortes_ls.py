"""Desigualdades (l,S) como cortes.

Para cada hospital i, escenario s, día l y conjunto de días S ⊆ {0..l}:
    Σ_{t∈S} llega[i,t,s] ≤ Σ_{t∈S} D[i,t..l,s] · Y[i,t] + inventario al cierre de l + vencidas antes de l
con llega[i,t,s] = Σ_{k,a} q[i,k,a,t,s] e Y[i,t] = Σ_k y[i,k,t]."""
import gurobipy as gp


def _indices(nombre):
    return tuple(int(z) for z in nombre[nombre.index("[") + 1:-1].split(","))


def mapa(m):
    """Variables q, y y V_cierre del modelo, agrupadas para armar los cortes."""
    q, y, V = {}, {}, {}
    for var in m.getVars():
        g = var.VarName.split("[")[0]
        if g == "q":
            i, k, a, t, s = _indices(var.VarName); q.setdefault((i, t, s), []).append(var)
        elif g == "y":
            i, k, t = _indices(var.VarName); y.setdefault((i, t), []).append(var)
        elif g == "V_cierre":
            i, a, t, s = _indices(var.VarName); V[i, a, t, s] = var
    return q, y, V


def expresion(q, y, V, D, i, s, l, S, L):
    """Lado izquierdo menos lado derecho del corte (≤ 0)."""
    dem = lambda t: float(D[s, t:l + 1, i].sum())
    return (gp.quicksum(v for t in S for v in q.get((i, t, s), []))
            - gp.quicksum(dem(t) * yy for t in S for yy in y.get((i, t), []))
            - gp.quicksum(V[i, a, l, s] for a in range(L))
            - gp.quicksum(V[i, L - 1, t, s] for t in range(l)))


def rondas(m, D, n_h, T, S_esc, L, max_rondas=50, tolerancia=1e-4):
    """Sobre el modelo relajado m: agrega las (l,S) violadas, ronda tras ronda,
    hasta que no queden. Devuelve (estimación final, lista de cortes, rondas)."""
    q, y, V = mapa(m)
    cortes = []
    ronda = 0
    for ronda in range(1, max_rondas + 1):
        m.optimize()
        nuevos = 0
        for i in range(n_h):
            for s in range(S_esc):
                llega = {t: sum(v.X for v in q.get((i, t, s), [])) for t in range(T)}
                Y = {t: sum(v.X for v in y.get((i, t), [])) for t in range(T)}
                for l in range(T):
                    S = [t for t in range(l + 1) if llega[t] > float(D[s, t:l + 1, i].sum()) * Y[t] + 1e-6]
                    if not S:
                        continue
                    viol = (sum(llega[t] - float(D[s, t:l + 1, i].sum()) * Y[t] for t in S)
                            - sum(V[i, a, l, s].X for a in range(L)) - sum(V[i, L - 1, t, s].X for t in range(l)))
                    if viol > tolerancia:
                        m.addConstr(expresion(q, y, V, D, i, s, l, S, L) <= 0)
                        cortes.append((i, s, l, tuple(S)))
                        nuevos += 1
        if nuevos == 0:
            break
    m.optimize()
    return m.ObjVal, cortes, ronda


def agregar(m, cortes, D, L):
    """Agrega una lista de cortes (encontrados en la estimación) a otro modelo, por ejemplo el entero."""
    q, y, V = mapa(m)
    for (i, s, l, S) in cortes:
        m.addConstr(expresion(q, y, V, D, i, s, l, S, L) <= 0)
