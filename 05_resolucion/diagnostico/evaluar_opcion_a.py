"""Evaluación de la opción a (topes de lo útil, `modelo_comun_mejorado.topes_utiles`).
Etapas (cada resultado se guarda al terminar; si se corta, sigue donde quedó):
 1. Instancias chicas, gap 0: ¿da el mismo óptimo que el original?
 2. Estimación (cota continua): distancia al óptimo con y sin topes, en las
    instancias chicas y a escala completa (semana 1, 22 hospitales, 30 escenarios).
 3. Instancias medianas (8, 9 y 10 hospitales, 3 vehículos, 3 escenarios, 10 min):
    topes solos y topes + desigualdades. El original y las desigualdades solas ya
    están en resultados/medir_mejoras.json (mismas instancias y parámetros).
 4. Escala completa, 1 h: topes + desigualdades. El original y las desigualdades
    solas ya están en resultados/confirmar_escala_completa.json.
Entre corridas pesadas descansa 5 minutos para no sobrecalentar el computador."""
import sys, pathlib, json, time, functools
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
import numpy as np
from codigo import configuracion as cfg
from codigo import modelo_comun, verificacion as v
from codigo.horizonte_rodante import estado_inicial_semana
from codigo.escenarios import generar_escenarios
from comun import capturar, insumos_y_semana, escenarios_calendario
import modelo_comun_mejorado as mm

SALIDA = RESULTADOS / "evaluar_opcion_a.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {}
def guardar(clave, valor):
    R[clave] = valor; json.dump(R, open(SALIDA, "w"), indent=1, default=str)
    print(clave, valor, flush=True)

def cota(esc, est, ins, **op):
    m = capturar(functools.partial(mm.resolver, **op), esc, est, ins).relax()
    m.Params.OutputFlag = 0; m.optimize(); return m.ObjVal

CHICAS = [
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría"], 2, 3),
    (["Hospital San José", "Hospital Calvo Mackenna", "Instituto Geriatría", "Hospital Sótero del Río"], 3, 2),
    (["Hospital del Salvador", "Hospital Barros Luco", "Hospital El Pino", "Hospital San José de Maipo", "Ex Posta Central"], 2, 2),
    (["Hospital del Salvador", "Instituto Neurocirugía", "Instituto Geriatría", "Hospital Calvo Mackenna", "Ex Posta Central", "Instituto Traumatológico"], 3, 1),
]
# Etapas 1 y 2 en chicas
cfg.GAP_MIP = 0.0
for nombres, nv, ne in CHICAS:
    clave = f"chica_{len(nombres)}"
    if clave in R: continue
    ins = v._insumos_chicos(nombres, n_vehiculos=nv)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, ne, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(len(nombres)))
    est = estado_inicial_semana(ins)
    orig = mm.resolver(esc, est, ins).info_resolucion
    top = mm.resolver(esc, est, ins, topes_de_lo_util=True).info_resolucion
    optimo = orig["valor_objetivo"]
    c0, c1, c2 = cota(esc, est, ins), cota(esc, est, ins, topes_de_lo_util=True), cota(esc, est, ins, topes_de_lo_util=True, desigualdades=True)
    d = lambda c: 100 * (optimo - c) / optimo
    guardar(clave, {"optimo": optimo, "optimo_con_topes": top["valor_objetivo"],
                    "mismo_optimo": top["optimo"] and abs(top["valor_objetivo"] - optimo) <= 1e-6 * optimo + 1e-3,
                    "t_original": orig["tiempo_s"], "t_topes": top["tiempo_s"],
                    "distancia_cota_original_pct": d(c0), "distancia_cota_topes_pct": d(c1), "distancia_cota_topes_desig_pct": d(c2)})

# Etapa 2 a escala completa (la mejor solución conocida de la semana 1 es 1.419.918, de b_en_comun.json)
if "cota_escala_completa" not in R:
    ins, semana = insumos_y_semana(0); esc = escenarios_calendario(ins, semana); est = estado_inicial_semana(ins)
    mejor = 1419917.7
    c0, c1, c2 = cota(esc, est, ins), cota(esc, est, ins, topes_de_lo_util=True), cota(esc, est, ins, topes_de_lo_util=True, desigualdades=True)
    guardar("cota_escala_completa", {"mejor_conocida": mejor, "cota_original": c0, "cota_topes": c1, "cota_topes_desig": c2,
                                     "distancia_original_pct": 100 * (mejor - c0) / mejor, "distancia_topes_pct": 100 * (mejor - c1) / mejor,
                                     "distancia_topes_desig_pct": 100 * (mejor - c2) / mejor})

# Etapa 3: medianas, 10 min
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 600
base = v._base()
dist = base.distancias.loc[base.nombre_banco, list(base.hospitales["nombre"])].sort_values()
for n in (8, 9, 10):
    ins = v._insumos_chicos(list(dist.index[:n]), n_vehiculos=3)
    esc = generar_escenarios(7, 9, ins.distribuciones_demanda, 3, distribuciones_oferta=ins.distribuciones_oferta, rng=np.random.default_rng(7))
    est = estado_inicial_semana(ins)
    for nombre, op in [("topes", dict(topes_de_lo_util=True)), ("topes+desigualdades", dict(topes_de_lo_util=True, desigualdades=True))]:
        clave = f"mediana_{n}_{nombre}"
        if clave in R: continue
        r = mm.resolver(esc, est, ins, **op).info_resolucion
        guardar(clave, {k: r[k] for k in ["tiempo_s", "optimo", "gap", "valor_objetivo", "cota", "trabajo"]})
        print("pausa 5 min", flush=True); time.sleep(300)

# Etapa 4: escala completa, 1 h
if "escala_completa_topes+desigualdades" not in R:
    cfg.TIEMPO_LIMITE_CALENDARIO_S = 3600
    ins, semana = insumos_y_semana(0); esc = escenarios_calendario(ins, semana); est = estado_inicial_semana(ins)
    r = mm.resolver(esc, est, ins, topes_de_lo_util=True, desigualdades=True).info_resolucion
    guardar("escala_completa_topes+desigualdades", {k: r[k] for k in ["tiempo_s", "optimo", "gap", "valor_objetivo", "cota", "trabajo"]})
print("FIN")
