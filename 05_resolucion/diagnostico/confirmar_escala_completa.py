"""Confirmación a escala completa (semana 1 del año simulado, 22 hospitales,
5 vehículos, 30 escenarios, semilla 2024): modelo común original y con
desigualdades válidas, 1 h cada uno y 10 min de pausa entre ellos. Guarda cada
resultado al terminar y, si se corta, sigue con el que falta."""
import sys, pathlib, json, time
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 3600
from comun import insumos_y_semana, escenarios_calendario
from codigo.horizonte_rodante import estado_inicial_semana
import modelo_comun_mejorado as mm

SALIDA = RESULTADOS / "confirmar_escala_completa.json"
filas = json.load(open(SALIDA)) if SALIDA.exists() else []
hechas = {f["variante"] for f in filas}
ins, semana = insumos_y_semana(0); esc = escenarios_calendario(ins, semana); est = estado_inicial_semana(ins)
for i, (nombre, op) in enumerate([("original", {}), ("desigualdades", dict(desigualdades=True))]):
    if nombre in hechas:
        continue
    if i > 0 and filas:
        print("pausa de 10 min", flush=True); time.sleep(600)
    r = mm.resolver(esc, est, ins, **op).info_resolucion
    filas.append({"variante": nombre, **{k: r[k] for k in ["tiempo_s", "optimo", "gap", "valor_objetivo", "cota", "trabajo"]}})
    json.dump(filas, open(SALIDA, "w"), indent=1)
    print(filas[-1], flush=True)
print("FIN")
