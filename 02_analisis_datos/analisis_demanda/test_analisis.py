"""Pruebas de fórmulas, soporte, bootstrap y separación temporal."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from analisis_demanda import analisis as cfg
from analisis_demanda.analisis import (
    ajustar_distribucion_demanda, ajustar_holm, estadisticos_discretos,
    validar_hospital, validar_demanda, NormalDiscreta, ajustar_alternativas,
    crps_discreto, costo_un_dia, evaluar_modelo, intervalos_pareados, comparar,
)
from codigo.cargador_datos import cargar_demanda_historica, cargar_parametros, tabla_hospitales


class EstadisticosTests(unittest.TestCase):
    def test_bernoulli_calculo_manual(self):
        # Fn(0)=0.6, F(0)=0.5: KS=0.1, AD=100*.1²*.5/.25=2.
        # Pearson: (60-50)²/50 + (40-50)²/50 = 4.
        x = np.r_[np.zeros(60), np.ones(40)]
        t, clases = estadisticos_discretos(x, stats.binom(1, .5))
        np.testing.assert_allclose(t, [.1, 2., 4.], atol=1e-12)
        self.assertEqual(int(clases.observado.sum()), 100)

    def test_ks_evalua_saltos_no_observados(self):
        x = np.r_[np.zeros(25), np.full(75, 4)]
        d = stats.poisson(2)
        t, _ = estadisticos_discretos(x, d)
        grid = np.arange(50)
        esperado = max(abs(np.mean(x <= k) - d.cdf(k)) for k in grid)
        self.assertAlmostEqual(t[0], esperado)
        self.assertGreater(t[0], max(abs(np.mean(x <= k) - d.cdf(k)) for k in np.unique(x)))

    def test_ad_suma_directa_y_colas_pearson(self):
        x = np.random.default_rng(123).poisson(5, 300)
        d = stats.poisson(5)
        t, clases = estadisticos_discretos(x, d)
        ad = 300 * sum((np.mean(x <= k) - d.cdf(k)) ** 2 * (d.pmf(k) + d.pmf(k+1)) / 2
                       / d.cdf(k) / d.sf(k) for k in range(30))
        self.assertAlmostEqual(t[1], ad, places=11)
        self.assertAlmostEqual(clases.esperado.sum(), len(x), places=10)
        self.assertEqual(clases.observado.sum(), len(x))
        self.assertGreaterEqual(clases.esperado.min(), 5 - 1e-10)
        self.assertEqual(clases.iloc[0].desde, 0)
        self.assertTrue(np.isinf(clases.iloc[-1].hasta))
        for _, c in clases.iterrows():
            self.assertEqual(c.observado, np.sum((x >= c.desde) & (x <= c.hasta)))

    def test_ad_pesos_discretos_dgof_no_uniformes(self):
        # Referencia dgof A2: masas (.49,.42,.09), pesos (.455,.255),
        # residuos de conteos acumulados (-19,-11); se excluye F=1.
        x = np.repeat([0, 1, 2], [30, 50, 20])
        t, _ = estadisticos_discretos(x, stats.binom(2, .3))
        esperado = (19**2 * .455 / (.49*.51) + 11**2 * .255 / (.91*.09)) / 100
        self.assertAlmostEqual(t[1], esperado, places=12)

    def test_fuera_soporte_no_desaparece(self):
        x = np.r_[np.zeros(30), np.ones(69), 2]
        t, _ = estadisticos_discretos(x, stats.binom(1, .5))
        self.assertTrue(np.isinf(t[1:]).all())

    def test_holm_con_valores_conocidos(self):
        np.testing.assert_allclose(ajustar_holm([.03, .001, .01, .5]), [.06, .004, .03, .5])

    def test_entradas_invalidas(self):
        for x in [[], [1], [1, np.nan], [2, np.inf], [-1, 2], [1, 1.000001], [[1, 2]]]:
            with self.subTest(x=x), self.assertRaises(ValueError):
                validar_hospital(x, "poisson", replicas=1)
        with self.assertRaises(ValueError):
            estadisticos_discretos([0, 1], stats.poisson(1))


class BootstrapTests(unittest.TestCase):
    def test_reajusta_todas_las_replicas_y_reproduce(self):
        x = np.random.default_rng(4).poisson(8, 150)
        with patch("analisis_demanda.analisis.ajustar_distribucion_demanda",
                   wraps=ajustar_distribucion_demanda) as ajustar:
            a, _ = validar_hospital(x, "poisson", replicas=39, rng=91)
            self.assertEqual(ajustar.call_count, 40)
            self.assertTrue(all(c.kwargs["validar_dispersion"] is False for c in ajustar.call_args_list))
        b, _ = validar_hospital(x, "poisson", replicas=39, rng=91)
        self.assertEqual(a, b)
        for p in ["ks", "ad", "chi2"]:
            self.assertEqual(a[f"{p}_p"], (1+a[f"{p}_excedencias"])/40)

    def test_detecta_alternativa_muy_distinta(self):
        x = np.r_[np.zeros(100), np.full(100, 20)]
        r, _ = validar_hospital(x, "poisson", replicas=99, rng=123)
        for p in ["ks", "ad", "chi2"]:
            self.assertLessEqual(r[f"{p}_p"], .02)

    def test_no_rechaza_replicas_por_dispersion(self):
        # Una realización NB puede ser subdispersa y una binomial sobredispersa.
        for x, tipo in [(np.tile([9, 10, 11], 10), "binomial_negativa"),
                        (np.tile([0, 20], 10), "binomial")]:
            with self.assertRaises(ValueError):
                ajustar_distribucion_demanda(x, tipo)
            d = ajustar_distribucion_demanda(x, tipo, validar_dispersion=False)
            self.assertTrue(np.isfinite(d.distribucion.logpmf(x)).all())

    def test_duplicados_y_cobertura_hospitales(self):
        datos = cargar_demanda_historica()
        with self.assertRaises(ValueError):
            validar_demanda(datos.iloc[:-1].assign(hospital="X"), replicas=1)
        with self.assertRaises(ValueError):
            validar_demanda(datos.iloc[np.r_[0, np.arange(len(datos))]], replicas=1)

    def test_integracion_los_22_hospitales(self):
        tabla, clases = validar_demanda(cargar_demanda_historica(), replicas=9)
        self.assertEqual(len(tabla), 22)
        self.assertEqual(list(tabla.hospital), list(cfg.TIPO_DISTRIBUCION_DEMANDA))
        self.assertTrue((clases.groupby("hospital").observado.sum() == 1096).all())
        self.assertTrue((clases.esperado >= 5 - 1e-10).all())
        self.assertIn("holm_sensible_mc", tabla)
        for p in ["ks", "ad", "chi2"]:
            self.assertTrue((tabla[f"{p}_p_holm"] >= tabla[f"{p}_p"]).all())

    def test_procesos_conservan_semillas_y_orden(self):
        datos = cargar_demanda_historica()
        tipos = dict(list(cfg.TIPO_DISTRIBUCION_DEMANDA.items())[:3])
        datos = datos.loc[datos.hospital.isin(tipos)]
        sec, clases_sec = validar_demanda(datos, tipos=tipos, replicas=9, trabajadores=1)
        par, clases_par = validar_demanda(datos, tipos=tipos, replicas=9, trabajadores=2)
        pd.testing.assert_frame_equal(sec, par)
        pd.testing.assert_frame_equal(clases_sec, clases_par)

    def test_ajustes_siguen_reproduciendo_tabla_a3(self):
        datos = cargar_demanda_historica()
        # Transcripción del P95 modelo de la imagen A.3, en orden de cfg.
        p95 = [42, 83, 51, 63, 48, 57, 42, 47, 44, 53, 22, 55, 31, 13,
               15, 62, 64, 42, 10, 20, 17, 36]
        for (hospital, tipo), percentil in zip(cfg.TIPO_DISTRIBUCION_DEMANDA.items(), p95):
            with self.subTest(hospital=hospital):
                x = datos.loc[datos.hospital == hospital, "demanda"].to_numpy()
                d = ajustar_distribucion_demanda(x, tipo)
                self.assertEqual(d.distribucion.ppf(.95), percentil)
                # Contrasta contra la verosimilitud sin comprimir del notebook.
                if tipo == "binomial_negativa":
                    opt = minimize_scalar(lambda z: -stats.nbinom.logpmf(
                        x, np.exp(z), np.exp(z)/(np.exp(z)+x.mean())).sum(),
                        bounds=cfg.COTAS_LOG_R_BINOMIAL_NEGATIVA, method="bounded")
                    np.testing.assert_allclose(d.parametros["n"], np.exp(opt.x), rtol=1e-4)
                elif tipo == "binomial":
                    ns = range(int(x.max()), max(int(x.max())+cfg.HOLGURA_N_BINOMIAL,
                                                 int(x.max())*cfg.FACTOR_N_BINOMIAL)+1)
                    mejor = max(ns, key=lambda n: stats.binom.logpmf(x, n, x.mean()/n).sum())
                    self.assertEqual(d.parametros["n"], mejor)


class ComparacionTests(unittest.TestCase):
    def test_normal_discreta_masa_y_cuantiles(self):
        for mu, sigma in [(7, 1.5), (1, 4), (-2, 3)]:
            d = NormalDiscreta(mu, sigma)
            k = np.arange(100)
            self.assertAlmostEqual(d.pmf(k).sum(), 1, places=12)
            np.testing.assert_allclose(d.pmf(k).cumsum(), d.cdf(k), atol=1e-12)
            np.testing.assert_allclose(d.cdf(k)+d.sf(k), 1, atol=1e-12)
            self.assertAlmostEqual(d.pmf(0), stats.norm.cdf((.5-mu)/sigma))
            for tau in [.1, .5, .77, .95, .99]:
                q = d.ppf(tau)
                self.assertGreaterEqual(d.cdf(q), tau)
                self.assertLess(d.cdf(q-1), tau)

    def test_crps_coincide_formula_energia(self):
        x = np.array([0, 2, 2, 6])
        valores, n = np.unique(x, return_counts=True)
        d = stats.rv_discrete(values=(valores, n/len(x)))
        y = np.array([0, 1, 8])
        esperado = [np.abs(x-v).mean()-.5*np.abs(x[:, None]-x).mean() for v in y]
        np.testing.assert_allclose(crps_discreto(d, y), esperado, atol=1e-12)

    def test_stock_proxy_minimiza_costo_de_entrenamiento(self):
        d = stats.poisson(8)
        costos = {"costo_shortage": 2300, "costo_holding_hospital": 680}
        y = np.arange(60)
        r, _ = evaluar_modelo(d, y, 12, costos)
        q = r["q_proxy"]
        perdidas = [np.dot(d.pmf(y), costo_un_dia(y, s, 2300, 680)) for s in range(13)]
        self.assertEqual(q, int(np.argmin(perdidas)))
        r, _ = evaluar_modelo(d, y, 2, costos)
        self.assertEqual(r["q_proxy"], 2)

    def test_bloques_conservan_diferencias_constantes(self):
        diferencias = np.tile([0., -3., 2.], (40, 1))
        ci = intervalos_pareados(diferencias, rng=np.random.default_rng(1), replicas=99,
                                bloque=7, n_contrastes=20)
        np.testing.assert_allclose(ci, np.tile([0., -3., 2.], (4, 1)), atol=1e-12)

    def test_empirica_conserva_frecuencias_y_ceros(self):
        modelos = ajustar_alternativas([0, 1, 1, 2, 2, 2], "poisson")
        d = modelos["empirica"][0]
        np.testing.assert_allclose(d.pmf([0, 1, 2]), [1/6, 2/6, 3/6])
        self.assertEqual(d.pmf(3), 0)

    def test_prueba_no_cambia_parametros_ni_stock(self):
        datos = cargar_demanda_historica()
        h = "Hospital San José"
        datos = datos.loc[datos.hospital.eq(h)].copy()
        modificado = datos.copy()
        modificado.loc[modificado.fecha >= cfg.CORTE_COMPARACION, "demanda"] += 50
        capacidades = tabla_hospitales().set_index("nombre").capacidad_maxima.to_dict()
        with patch.object(cfg, "HOSPITALES_COMPARACION", (h,)), patch.object(cfg, "REPLICAS_COMPARACION", 19):
            a, _, _, ajustes_a = comparar(datos, cargar_parametros(), capacidades)
            b, _, _, ajustes_b = comparar(modificado, cargar_parametros(), capacidades)
        self.assertEqual(ajustes_a, ajustes_b)
        np.testing.assert_array_equal(a.q_proxy, b.q_proxy)
        self.assertFalse(np.allclose(a.crps, b.crps))
        self.assertTrue((a.n_entrenamiento == 730).all())
        self.assertTrue((a.n_prueba == 366).all())


if __name__ == "__main__":
    unittest.main()
