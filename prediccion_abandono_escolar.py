import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


class SimuladorDatos:
    
    def __init__(self, n=200, seed=42):
        self.n = n
        self.seed = seed
    
    def generar(self):
        np.random.seed(self.seed)

        edad = np.random.randint(18, 30, self.n)
        horas_estudio = np.random.uniform(0, 30, self.n)
        asistencia = np.random.uniform(50, 100, self.n)
        promedio = np.random.uniform(5, 10, self.n)
        uso_online = np.random.uniform(0, 15, self.n)

        # Probabilidad de abandono: 30%
        abandono = np.random.choice([0, 1], size=self.n, p=[0.7, 0.3])

        df = pd.DataFrame({
            "Edad": edad,
            "Horas_estudio": horas_estudio,
            "Asistencia": asistencia,
            "Promedio": promedio,
            "Uso_online": uso_online,
            "Abandono": abandono
        })

        return df



class ModeloAbandono:

    def __init__(self, max_depth=4, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.modelo = None
        self.X_test = None
        self.y_test = None

    def entrenar(self, data):
        X = data.drop("Abandono", axis=1)
        y = data["Abandono"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        self.modelo = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )

        self.modelo.fit(X_train, y_train)

        self.X_test = X_test
        self.y_test = y_test

    def evaluar(self):
        predicciones = self.modelo.predict(self.X_test)

        print("🔎 Precisión del modelo:")
        print(accuracy_score(self.y_test, predicciones))

        print("\n📄 Reporte de clasificación:")
        print(classification_report(self.y_test, predicciones))

    def predecir_estudiante(self, estudiante_df):
        pred = self.modelo.predict(estudiante_df)[0]

        return "Abandonará" if pred == 1 else "Seguirá estudiando"


class TestBasicoModeloAbandono:

    def ejecutar(self):

        print("📌 Generando datos simulados...")
        sim = SimuladorDatos()
        df = sim.generar()
        print(df.head(), "\n")

        print("📌 Entrenando el modelo...")
        modelo = ModeloAbandono()
        modelo.entrenar(df)

        print("\n📌 Evaluando el modelo...")
        modelo.evaluar()

        print("\n📌 Probando predicción con un nuevo estudiante...")

        nuevo = pd.DataFrame([{
            "Edad": 22,
            "Horas_estudio": 12,
            "Asistencia": 78,
            "Promedio": 7.5,
            "Uso_online": 5
        }])

        resultado = modelo.predecir_estudiante(nuevo)
        print("\nResultado de la predicción:", resultado)


test = TestBasicoModeloAbandono()
test.ejecutar()