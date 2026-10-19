"""Cota inferior del modelo de calendario por relajación lagrangiana (EN DESARROLLO).

Motivo: la estimación optimista (relajación lineal) del modelo común ignora el
costo fijo de cada viaje ("pedazo de camión"). En vez de soltar
las binarias, aquí se sueltan las restricciones que CONECTAN el problema y se
les pone un precio; el problema se separa en pedazos chicos que se resuelven
con sus binarias enteras, así que la estimación paga los viajes completos.

Reformulación equivalente al modelo de calendario del modelo común:
- A[i,a,t,s]: lo que llega al hospital i, de edad a, el día t, en el escenario s.
- O[a,t,s]:   lo que sale del banco, de edad a.
- l[i,k,t,s]: lo que el vehículo k le lleva al hospital i (sin edad).
Con las conexiones
  (E1) Σ_i A[i,a,t,s] = O[a,t,s]            (lo que sale del banco llega a los hospitales)
  (E2) Σ_a A[i,a,t,s] = Σ_k l[i,k,t,s]       (lo que llega lo llevan los vehículos)
es el mismo problema: dado A y l con totales iguales, siempre se puede armar
q[i,k,a,t,s] con esos totales (problema de transporte). Se sueltan E1 (precios
lam[a,t,s]) y E2 (precios pi[i,t,s]), y quedan pedazos independientes:
- banco, por escenario: oferta admitida, salidas O, inventario del banco;
- hospital, por hospital y escenario: llegadas A, consumo e inventario;
- vehículo, por día: ruta (binaria, se paga completa) y cargas l por escenario.
  Los vehículos son idénticos, así que el pedazo de un día se resuelve una vez
  y vale por los K vehículos.

Cota: L(lam, pi) = Σ pedazos ≤ óptimo del modelo de calendario, para cualquier
precio. Los precios se ajustan con el método del subgradiente.
"""

import math
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from codigo import modelo_comun


def _modelo(nombre):
    m = gp.Model(nombre)
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 1e-6
    return m


class Lagrangiana:
    """Arma una sola vez los pedazos (banco, hospitales y vehículo por día) y
    después, para cada vector de precios, solo cambia sus costos y los resuelve."""

    def __init__(self, escenarios, estado, insumos):
        par = insumos.parametros
        self.L = L = par["shelf_life"]
        self.S, self.T = escenarios.n_escenarios, escenarios.dias
        self.n_h = n_h = len(insumos.hospitales)
        self.K = len(insumos.flota)
        capacidades = set(insumos.flota["capacidad"])
        if len(capacidades) != 1:
            raise ValueError("Supone vehículos idénticos.")
        self.theta = theta = capacidades.pop()
        p = 1.0 / self.S
        beta, gamma = par["costo_holding_banco"], par["costo_holding_hospital"]
        pi_c, omega = par["costo_shortage"], par["costo_wastage"]
        C_i = insumos.hospitales["capacidad_maxima"].to_numpy()
        mu_b, mu_i = modelo_comun.minimos_terminales(insumos)
        O = escenarios.oferta
        D = modelo_comun._demanda_por_hospital(escenarios, insumos)
        A_, T_, S_ = range(L), range(self.T), range(self.S)

        # ---- Banco, por escenario
        self.banco = []
        for s in S_:
            m = _modelo(f"banco_{s}")
            r = m.addVars(T_, vtype=GRB.INTEGER, lb=0, ub={t: float(O[s, t]) for t in T_})
            sal = m.addVars(A_, T_, lb=0)
            Bc = m.addVars(A_, T_, lb=0)
            e = m.addVar(lb=0)
            for t in T_:
                disp = {a: (float(estado.inventario_banco[a]) if t == 0 else (Bc[a - 1, t - 1] if a >= 1 else 0)) + (r[t] if a == 0 else 0)
                        for a in A_}
                for a in A_:
                    m.addConstr(Bc[a, t] == disp[a] - sal[a, t])
                m.addConstr(gp.quicksum(disp[a] for a in A_) <= insumos.capacidad_banco)
            m.addConstr(e >= mu_b - gp.quicksum(Bc[a, self.T - 1] for a in A_ if a <= L - 2))
            costo = p * (beta * gp.quicksum(Bc[a, t] for a in A_ if a <= L - 2 for t in T_)
                         + omega * gp.quicksum(Bc[L - 1, t] for t in T_) + pi_c * e)
            self.banco.append((m, sal, costo))

        # ---- Hospital, por hospital y escenario
        self.hosp = {}
        for i in range(n_h):
            for s in S_:
                m = _modelo(f"hosp_{i}_{s}")
                Ar = m.addVars(A_, T_, vtype=GRB.INTEGER, lb=0)
                u = m.addVars(A_, T_, vtype=GRB.INTEGER, lb=0)
                Vc = m.addVars(A_, T_, lb=0)
                z = m.addVars(T_, lb=0)
                e = m.addVar(lb=0)
                for t in T_:
                    disp = {a: (float(estado.inventario_hospitales[i, a]) if t == 0 else (Vc[a - 1, t - 1] if a >= 1 else 0)) + Ar[a, t]
                            for a in A_}
                    for a in A_:
                        m.addConstr(Vc[a, t] == disp[a] - u[a, t])
                    m.addConstr(u.sum("*", t) + z[t] == float(D[s, t, i]))
                    m.addConstr(gp.quicksum(disp[a] for a in A_) <= C_i[i])
                    # Lo que llega en un día lo llevan a lo más K vehículos de θ (vale en el original).
                    m.addConstr(Ar.sum("*", t) <= self.K * theta)
                m.addConstr(e >= mu_i[i] - gp.quicksum(Vc[a, self.T - 1] for a in A_ if a <= L - 2))
                costo = p * (gamma * gp.quicksum(Vc[a, t] for a in A_ if a <= L - 2 for t in T_)
                             + pi_c * z.sum() + omega * gp.quicksum(Vc[L - 1, t] for t in T_) + pi_c * e)
                self.hosp[i, s] = (m, Ar, costo)

        # ---- Vehículo, por día (ruta binaria + cargas por escenario)
        delta = modelo_comun.matriz_distancias(insumos)
        alfa = par["costo_km"]
        nodos = range(n_h + 1)
        self.vehiculo = []
        for t in T_:
            m = _modelo(f"vehiculo_{t}")
            y = m.addVars(range(n_h), vtype=GRB.BINARY)
            w = m.addVar(vtype=GRB.BINARY)
            x = m.addVars([(a, b) for a in nodos for b in nodos if a != b], vtype=GRB.BINARY)
            rho = m.addVars(range(n_h), lb=1, ub=n_h)
            l = m.addVars(range(n_h), S_, vtype=GRB.INTEGER, lb=0)
            for i in range(n_h):
                m.addConstr(gp.quicksum(x[i + 1, b] for b in nodos if b != i + 1) == y[i])
                m.addConstr(gp.quicksum(x[a, i + 1] for a in nodos if a != i + 1) == y[i])
                m.addConstr(y[i] <= w)
                for s in S_:
                    m.addConstr(l[i, s] <= theta * y[i])
            m.addConstr(gp.quicksum(x[0, b] for b in nodos if b != 0) == w)
            m.addConstr(gp.quicksum(x[a, 0] for a in nodos if a != 0) == w)
            for i in range(n_h):
                for j in range(n_h):
                    if i != j:
                        m.addConstr(rho[i] - rho[j] + n_h * x[i + 1, j + 1] <= n_h - 1)
            for s in S_:
                m.addConstr(l.sum("*", s) <= theta)
            transporte = alfa * gp.quicksum(delta[a, b] * x[a, b] for (a, b) in x)
            self.vehiculo.append((m, l, transporte))

    def evaluar(self, lam, pi):
        """Resuelve todos los pedazos con los precios dados.

        lam: arreglo (L, T, S) para E1; pi: arreglo (n_h, T, S) para E2.
        Devuelve: (cota L(lam, pi), subgradiente de E1, subgradiente de E2).
        """
        total = 0.0
        g1 = np.zeros_like(lam)  # Σ_i A − O
        g2 = np.zeros_like(pi)   # Σ_a A − Σ_k l
        for s, (m, sal, costo) in enumerate(self.banco):
            m.setObjective(costo - gp.quicksum(lam[a, t, s] * sal[a, t] for (a, t) in sal))
            m.optimize(); total += m.ObjVal
            for (a, t), v in sal.items():
                g1[a, t, s] -= v.X
        for (i, s), (m, Ar, costo) in self.hosp.items():
            m.setObjective(costo + gp.quicksum((lam[a, t, s] + pi[i, t, s]) * Ar[a, t] for (a, t) in Ar))
            m.optimize(); total += m.ObjVal
            for (a, t), v in Ar.items():
                g1[a, t, s] += v.X
                g2[i, t, s] += v.X
        for t, (m, l, transporte) in enumerate(self.vehiculo):
            m.setObjective(transporte - gp.quicksum(pi[i, t, s] * l[i, s] for (i, s) in l))
            m.optimize()
            valor = m.ObjVal  # ≤ 0: no salir siempre es posible
            total += self.K * valor
            for (i, s), v in l.items():
                g2[i, t, s] -= self.K * v.X
        return total, g1, g2


def cota_lagrangiana(escenarios, estado, insumos, cota_superior, iteraciones=300, paso_inicial=2.0,
                     paciencia=20, registro=None):
    """Método del subgradiente: busca los precios que hacen la cota lo más alta posible.

    Recibe: la instancia y una cota superior (el costo de una solución factible
        conocida, por ejemplo la de la heurística B), que se usa para el tamaño
        del paso (regla de Polyak).
    Devuelve: diccionario con la mejor cota encontrada, su historial y el tiempo.
    """
    lag = Lagrangiana(escenarios, estado, insumos)
    lam = np.zeros((lag.L, lag.T, lag.S))
    pi = np.zeros((lag.n_h, lag.T, lag.S))
    mejor, sin_mejora, paso = -math.inf, 0, paso_inicial
    historial, inicio = [], time.time()
    for it in range(1, iteraciones + 1):
        valor, g1, g2 = lag.evaluar(lam, pi)
        if valor > mejor + 1e-6:
            mejor, sin_mejora = valor, 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                paso, sin_mejora = paso / 2, 0
        norma = float((g1 ** 2).sum() + (g2 ** 2).sum())
        historial.append({"iteracion": it, "valor": valor, "mejor": mejor, "paso": paso, "norma": norma,
                          "tiempo_s": time.time() - inicio})
        if registro:
            registro(historial[-1])
        if norma < 1e-9 or paso < 1e-4:
            break
        t = paso * (cota_superior - valor) / norma
        lam += t * g1
        pi += t * g2
    return {"cota": mejor, "iteraciones": len(historial), "tiempo_s": time.time() - inicio, "historial": historial}
