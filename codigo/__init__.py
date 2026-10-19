"""Paquete `codigo`: sus módulos están repartidos en las carpetas numeradas del
repositorio (03 a 08), en el mismo orden en que se presentan en el informe."""
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
__path__ += [str(c) for c in sorted(_RAIZ.glob("0[3-8]_*"))]
