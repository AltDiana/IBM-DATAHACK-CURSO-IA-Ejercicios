import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# ============================================================
#   Clase 1: GeneradorSeries
# ============================================================
class GeneradorSeries:
    """
    Genera combinaciones de lotería con 6 números únicos entre 1 y 49.
    """

    def generar_series(self, cantidad: int) -> np.ndarray:
        series = []
        for _ in range(cantidad):
            combinacion = np.random.choice(range(1, 50), size=6, replace=False)
            combinacion.sort()
            series.append(combinacion)
        return np.array(series)


# ============================================================
#   Clase 2: DatosLoteria
# ============================================================
class DatosLoteria:
    """
    Genera datos históricos de combinaciones con etiquetas de éxito (1) o fracaso (0).
    """

    def __init__(self):
        self.generador = GeneradorSeries()

    def generar_datos_entrenamiento(self, cantidad: int = 1000) -> pd.DataFrame:
        series = self.generador.generar_series(cantidad)

        # 10% de éxitos (1), 90% fracasos (0)
        etiquetas = np.zeros(cantidad, dtype=int)
        exitos_indices = np.random.choice(cantidad, size=int(cantidad * 0.10), replace=False)
        etiquetas[exitos_indices] = 1

        df = pd.DataFrame(series, columns=[f"N{i}" for i in range(1, 7)])
        df["Exito"] = etiquetas

        return df


# ============================================================
#   Clase 3: ModeloLoteria
# ============================================================
class ModeloLoteria:
    """
    Entrena un modelo RandomForestClassifier para predecir probabilidad de éxito.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.modelo = RandomForestClassifier(n_estimators=300, random_state=42)

    def entrenar(self, X: pd.DataFrame, y: pd.Series):
        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)

    def predecir_probabilidades(self, X_nuevas: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_nuevas)
        probabilidades = self.modelo.predict_proba(X_scaled)[:, 1]  # prob de éxito
        return probabilidades


# ============================================================
#   Clase 4: VisualizadorResultados
# ============================================================
class VisualizadorResultados:
    """
    Gráfica las combinaciones más prometedoras según el modelo.
    """

    def graficar_top_combinaciones(self, series, probabilidades, top_n=10):
        # Ordenar de mejor a peor
        idx_top = np.argsort(probabilidades)[-top_n:]
        series_top = series[idx_top]
        prob_top = probabilidades[idx_top]

        etiquetas = [str(list(s)) for s in series_top]

        plt.figure(figsize=(8, 6))
        plt.barh(etiquetas, prob_top)
        plt.xlabel("Probabilidad de éxito")
        plt.title(f"Top {top_n} combinaciones más prometedoras")
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()


# ============================================================
#   Clase 5: EjecutarSimulacion
# ============================================================
class EjecutarSimulacion:
    """
    Integra todo: genera datos, entrena modelo, crea nuevas series,
    predice y muestra resultados.
    """

    def ejecutar(self):
        print("\n🔄 Generando datos de entrenamiento...")
        datos = DatosLoteria().generar_datos_entrenamiento(1000)

        X = datos[[f"N{i}" for i in range(1, 7)]]
        y = datos["Exito"]

        print("🧠 Entrenando modelo...")
        modelo = ModeloLoteria()
        modelo.entrenar(X, y)

        print("🎲 Generando nuevas combinaciones a evaluar...")
        generador = GeneradorSeries()
        series_nuevas = generador.generar_series(200)

        print("📊 Calculando probabilidades...")
        probabilidades = modelo.predecir_probabilidades(series_nuevas)

        # Mejor combinación
        mejor_idx = np.argmax(probabilidades)
        mejor_serie = series_nuevas[mejor_idx]
        mejor_prob = probabilidades[mejor_idx]

        print("\n🎯 Mejor serie encontrada:")
        print(f"Números: {list(mejor_serie)}")
        print(f"Probabilidad estimada de éxito: {mejor_prob:.4f}")

        # Mostrar las 10 mejores
        visualizador = VisualizadorResultados()
        visualizador.graficar_top_combinaciones(series_nuevas, probabilidades, top_n=10)


# ============================================================
#   EJECUCIÓN DEL PROGRAMA
# ============================================================
if __name__ == "__main__":
    simulacion = EjecutarSimulacion()
    simulacion.ejecutar()
