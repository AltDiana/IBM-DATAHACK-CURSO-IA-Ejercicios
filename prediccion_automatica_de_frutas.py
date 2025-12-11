import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class GeneradorFrutas:
    """Genera datos sintéticos de frutas basados en peso y tamaño."""
 
    def __init__(self):
        self.frutas = ['Manzana', 'Plátano', 'Naranja']
 
    def generar(self, num_muestras):
        datos = []
        etiquetas = []
 
        for _ in range(num_muestras):
            fruta = random.choice(self.frutas)
 
            if fruta == 'Manzana':
                peso = random.uniform(120, 200)
                tamaño = random.uniform(7, 9)
            elif fruta == 'Plátano':
                peso = random.uniform(100, 150)
                tamaño = random.uniform(12, 20)
            elif fruta == 'Naranja':
                peso = random.uniform(150, 250)
                tamaño = random.uniform(8, 12)
 
            datos.append([peso, tamaño])
            etiquetas.append(fruta)
 
        return np.array(datos), np.array(etiquetas)
class ClasificadorFrutas:
    """Entrena un clasificador KNN para predecir frutas."""
 
    def __init__(self, k=3):
        self.k = k
        self.modelo = KNeighborsClassifier(n_neighbors=self.k)
 
    def entrenar(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        self.modelo.fit(self.X_train, self.y_train)
        self.y_pred = self.modelo.predict(self.X_test)
 
    def evaluar(self):
        precision = accuracy_score(self.y_test, self.y_pred)
        print(f"🔍 Precisión del modelo: {precision * 100:.2f}%")
        return precision
 
    def predecir(self, peso, tamaño):
        prediccion = self.modelo.predict([[peso, tamaño]])
        return prediccion[0]

class VisualizadorFrutas:
    """Genera visualizaciones para los datos de frutas."""
 
    def graficar_datos(self, X, y):
        colores = {'Manzana': 'red', 'Plátano': 'yellow', 'Naranja': 'orange'}
        plt.figure(figsize=(8, 6))
 
        for fruta in np.unique(y):
            indices = y == fruta
            plt.scatter(X[indices, 0], X[indices, 1],
                        label=fruta, color=colores[fruta], edgecolors='k')
 
        plt.xlabel("Peso (g)")
        plt.ylabel("Tamaño (cm)")
        plt.title("Distribución de frutas por peso y tamaño")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class SimuladorFrutas:
    """Clase principal que ejecuta todo el flujo."""
 
    def ejecutar(self):
        # Generar datos
        generador = GeneradorFrutas()
        X, y = generador.generar(100)
 
        # Visualizar los datos
        visualizador = VisualizadorFrutas()
        visualizador.graficar_datos(X, y)
 
        # Entrenar el modelo
        clasificador = ClasificadorFrutas(k=3)
        clasificador.entrenar(X, y)
        clasificador.evaluar()
 
        # Hacer una predicción
        peso_nuevo = 140
        tamaño_nuevo = 18
        prediccion = clasificador.predecir(peso_nuevo, tamaño_nuevo)
        print(f"🍎 La fruta predicha para peso={peso_nuevo}g y tamaño={tamaño_nuevo}cm es: {prediccion}")