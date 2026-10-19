"""¿Gurobi ya hacía algo parecido a las (l,S) por su cuenta? Para las instancias
medianas, compara cuatro estimaciones (distancia a la mejor solución conocida):
 1. básica (relajación lineal, sin cortes);
 2. básica + nuestras (l,S);
 3. la de Gurobi después de SUS propios cortes en el nodo raíz (NodeLimit = 1), sin (l,S);
 4. la de Gurobi después de sus cortes, con nuestras (l,S) agregadas."""
import sys, pathlib, json, functools
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

M = {(f["hospitales"], f["variante"]): f for f in json.load(open(RESULTADOS / "medir_mejoras.json"))}
LS = json.load(open(RESULTADOS / "medir_ls.json"))
base = v._base()
dist = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()
filas = []
for n in (8, 9, 10):
    ins = v._insumos_chicos(list(dist.index[:n]), n_vehiculos=3)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    L = ins.parametros["shelf_life"]; D = modelo_comun._demanda_por_hospital(esc, ins)
    mejor = min([M[k]["valor_objetivo"] for k in M if k[0] == n and k[1] in ("original", "desigualdades")] +
                [LS[f"{n}_ls"]["valor_objetivo"], LS[f"{n}_ls+desigualdades"]["valor_objetivo"]])
    lp = capturar(mm.resolver, esc, est, ins).relax(); lp.Params.OutputFlag = 0
    lp.optimize(); basica = lp.ObjVal
    con_ls, cortes, _ = cortes_ls.rondas(lp, D, n, esc.dias, 3, L)
    raiz = {}
    for nombre, agregar in [("gurobi_sin_ls", False), ("gurobi_con_ls", True)]:
        m = capturar(mm.resolver, esc, est, ins); m.Params.OutputFlag = 0; m.Params.NodeLimit = 1; m.Params.TimeLimit = 600
        if agregar: cortes_ls.agregar(m, cortes, D, L)
        m.optimize(); raiz[nombre] = m.ObjBound
    d = lambda c: 100 * (mejor - c) / mejor
    fila = {"hospitales": n, "mejor_conocida": mejor, "basica": d(basica), "basica_con_ls": d(con_ls),
            "gurobi_raiz_sin_ls": d(raiz["gurobi_sin_ls"]), "gurobi_raiz_con_ls": d(raiz["gurobi_con_ls"])}
    filas.append(fila); print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in fila.items()}, flush=True)
json.dump(filas, open(RESULTADOS / "comparar_raiz.json", "w"), indent=1)
