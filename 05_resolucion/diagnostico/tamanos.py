import sys, time, json
import sys, pathlib
CARPETA = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
sys.path.insert(0, str(CARPETA.parents[1]))  # raíz del repo, para importar el paquete codigo
RESULTADOS = CARPETA.parents[1] / "resultados"
from comun import *
from codigo.modelo_comun import SolucionCalendario
ins, semana = insumos_y_semana(0)
esc = escenarios_calendario(ins, semana)
estado = estado_inicial_semana(ins)
salida = {}
for nombre, f in [("comun_calendario", modelo_comun.resolver_modelo_calendario),
                  ("A_inventario", heuristica_a.resolver_modelo_calendario),
                  ("B_calendario", heuristica_b.resolver_modelo_calendario)]:
    t = time.time(); m = capturar(f, esc, estado, ins)
    salida[nombre] = {**tamano(m), "grupos": por_grupo(m), "construccion_s": round(time.time()-t, 1)}
    print(nombre, salida[nombre], flush=True)
# modelo de cantidades del lunes con un calendario cualquiera (el de visitar todo, como la A)
cal = SolucionCalendario(visitas={(t, i, k) for t in range(7) for i in range(22) for k in range(5)})
m = capturar(modelo_comun.resolver_modelo_cantidades, esc, estado, cal, ins)
salida["cantidades_lunes"] = {**tamano(m), "grupos": por_grupo(m)}
print("cantidades_lunes", salida["cantidades_lunes"], flush=True)
json.dump(salida, open(RESULTADOS / "tamanos.json", "w"), indent=1)
