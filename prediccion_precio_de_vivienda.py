import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =====================================================
# 🔹 Clase 1: SimuladorViviendas
# =====================================================

class SimuladorViviendas:
    """
    Genera un conjunto de datos sintético de viviendas.
    """

    def generar_datos(self, n=200, seed=42):
        np.random.seed(seed)

        superficie = np.random.uniform(50, 200, n)                # m2
        habitaciones = np.random.randint(1, 6, n)                 # 1 a 5 habitaciones
        antiguedad = np.random.randint(0, 50, n)                  # años
        distancia = np.random.uniform(1, 20, n)                   # km al centro
        baños = np.random.randint(1, 4, n)                        # 1 a 3 baños

        # Fórmula sintética para el precio
        precio = (
            superficie * 2500 +
            habitaciones * 15000 -
            antiguedad * 1200 -
            distancia * 800 +
            baños * 10000 +
            np.random.normal(0, 30000, n)  # ruido
        )

        df = pd.DataFrame({
            "Superficie": superficie,
            "Habitaciones": habitaciones,
            "Antigüedad": antiguedad,
            "Distancia_centro": distancia,
            "Baños": baños,
            "Precio": precio
        })

        return df



# =====================================================
# 🔹 Clase 2: ModeloPrecioVivienda
# =====================================================

class ModeloPrecioVivienda:
    """
    Entrena un modelo de regresión lineal para predecir precios de viviendas.
    """

    def __init__(self):
        self.modelo = LinearRegression()
        self.X_test = None
        self.y_test = None

    def entrenar(self, data: pd.DataFrame):
        X = data.drop("Precio", axis=1)
        y = data["Precio"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.modelo.fit(X_train, y_train)

        self.X_test = X_test
        self.y_test = y_test

        print("Modelo entrenado correctamente.\n")

    def evaluar(self):
        pred = self.modelo.predict(self.X_test)

        mse = mean_squared_error(self.y_test, pred)
        r2 = r2_score(self.y_test, pred)

        print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
        print(f"R² del modelo: {r2:.2f}\n")

    def predecir(self, nueva_vivienda: pd.DataFrame) -> float:
        return float(self.modelo.predict(nueva_vivienda)[0])



# =====================================================
# 🔹 Clase 3: TestModeloPrecio
# =====================================================

class TestModeloPrecio:
    """
    Ejecuta el sistema completo para probar la predicción de precios.
    """

    def ejecutar(self):

        print("📌 Generando datos simulados...\n")
        simulador = SimuladorViviendas()
        df = simulador.generar_datos()

        print("Primeras filas de datos simulados:")
        print(df.head())
        print("\n")

        print("📌 Entrenando el modelo...\n")
        modelo = ModeloPrecioVivienda()
        modelo.entrenar(df)

        print("📌 Evaluando el modelo...\n")
        modelo.evaluar()

        print("📌 Predicción con una vivienda de ejemplo...\n")

        nueva = pd.DataFrame([{
            "Superficie": 120,
            "Habitaciones": 3,
            "Antigüedad": 10,
            "Distancia_centro": 5,
            "Baños": 2
        }])

        precio_estimado = modelo.predecir(nueva)

        print(f"El precio estimado de la vivienda es: ${precio_estimado:,.2f}")


test = TestModeloPrecio()
test.ejecutar()
