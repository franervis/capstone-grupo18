"""Corre `sin_orden_zonas` con los MISMOS escenarios que el modelo común libre
de `01_datos/linea_base_2024-09-02.json` (la "línea base" oficial del
proyecto). Permite medir cuánto se aleja la propuesta sin orden con zonas de
la heurística B del óptimo del modelo común.

Setup replicado:
- 22 hospitales, 5 vehículos (los reales).
- Semana del 2024-09-02 del archivo `01_datos/semanas_simuladas_2024_5xmes.csv`
  (`indice=240`, oferta del lunes = 707). Es la que usó la corrida de 12 h del
  modelo común en `linea_base_2024-09-02.json` (valor 1.845.145,73; cota
  1.571.515,54; gap 14,83 %).
- 30 escenarios del modelo con `generador_modelo_semana(2024, 240)`.
- Gap MIP 1 %, 1 h de límite.

NO reemplaza las otras corridas del `sin_orden_zonas` (que están con 3 esc,
mes 9, semilla 7 y los primeros n hospitales por distancia). Esta corrida es
la que se compara directamente con la línea base."""
import sys, pathlib, json, time
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA)); sys.path.insert(0, str(CARPETA.parents[1]))
RESULTADOS = CARPETA.parents[1] / "resultados"
from codigo import configuracion as cfg
cfg.GAP_MIP = 0.01; cfg.TIEMPO_LIMITE_CALENDARIO_S = 3600
from codigo import heuristica_b, simulacion
from codigo.horizonte_rodante import preparar_insumos, estado_inicial_semana
from codigo.escenarios import generar_escenarios
from codigo.sin_orden_zonas import resolver_sin_orden_zonas

SALIDA = RESULTADOS / "comparar_sin_orden_zonas_vs_linea_base.json"
R = json.load(open(SALIDA)) if SALIDA.exists() else {}

if "resultado" not in R:
    ins = preparar_insumos()
    semanas = simulacion.cargar_semanas("01_datos/semanas_simuladas_2024_5xmes.csv", ins)
    semana = next(s for s in semanas if str(s.fecha_lunes.date()) == "2024-09-02")
    rng = simulacion.generador_modelo_semana(2024, semana.indice)
    esc = generar_escenarios(7, semana.fecha_lunes.month, ins.distribuciones_demanda, 30,
                             distribuciones_oferta=ins.distribuciones_oferta,
                             oferta_hoy=int(semana.oferta[0]), rng=rng)
    est = estado_inicial_semana(ins)
    zonas = heuristica_b.particion_compacta(ins)
    inicio = time.time()
    sol = resolver_sin_orden_zonas(esc, est, ins, zonas)
    r = sol.info_resolucion
    linea = json.load(open(CARPETA.parents[1] / "01_datos" / "linea_base_2024-09-02.json"))["calendario_info"]
    R["resultado"] = {
        "semana": "2024-09-02", "indice": semana.indice, "oferta_hoy": int(semana.oferta[0]),
        "hospitales": len(ins.hospitales), "vehiculos": len(ins.flota), "n_escenarios": 30, "semilla": 2024,
        "zonas": zonas, "tam_zonas": [len(z) for z in zonas],
        "tiempo_s": time.time() - inicio, "optimo_zonas": r["optimo"], "gap_mip": r["gap"],
        "valor_zonas": r["valor_objetivo"], "cota_zonas": r["cota"],
        "visitas": len(sol.visitas), "km": sol.distancia_km,
        "linea_base_valor": linea["valor_objetivo"], "linea_base_cota": linea["cota"],
        "linea_base_gap_mip": linea["gap"], "linea_base_tiempo_s": linea["tiempo_s"],
        "gap_zonas_vs_valor_linea_base": (r["valor_objetivo"] - linea["valor_objetivo"]) / linea["valor_objetivo"],
        "gap_zonas_vs_cota_linea_base": (r["valor_objetivo"] - linea["cota"]) / linea["cota"],
    }
    json.dump(R, open(SALIDA, "w"), indent=1)
    print(json.dumps(R["resultado"], indent=1, default=str), flush=True)
print("FIN")
