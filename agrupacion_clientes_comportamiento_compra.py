import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SimuladorClientes:
    """Simula un conjunto de datos de clientes con variables de comportamiento."""
    
    def __init__(self, num_muestras=200, seed=42):
        self.num_muestras = num_muestras
        self.seed = seed
 
    def generar_datos(self):
        np.random.seed(self.seed)
        monto_gastado = np.random.uniform(100, 10000, self.num_muestras)
        frecuencia_compras = np.random.randint(1, 100, self.num_muestras)
        categorias_preferidas = np.random.randint(1, 6, size=(self.num_muestras, 3))
 
        datos = np.column_stack((
            monto_gastado,
            frecuencia_compras,
            categorias_preferidas.sum(axis=1)
        ))
 
        return datos

class ModeloSegmentacionClientes:
    """Entrena un modelo KMeans para agrupar clientes y visualizar resultados."""
 
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.modelo = KMeans(n_clusters=self.n_clusters, random_state=42)
 
    def entrenar(self, datos):
        self.datos_escalados = self.scaler.fit_transform(datos)
        self.modelo.fit(self.datos_escalados)
        print(f"Modelo entrenado con {self.n_clusters} clusters.")
 
    def predecir(self, cliente_nuevo):
        cliente_scaled = self.scaler.transform([cliente_nuevo])
        cluster_predicho = self.modelo.predict(cliente_scaled)
        return int(cluster_predicho[0])

class TestSegmentacionClientes:
    """Integra todo el proceso: simulación, entrenamiento, predicción y visualización."""
 
    def ejecutar(self):
        # Paso 1: Simular datos
        simulador = SimuladorClientes()
        datos = simulador.generar_datos()
        print("Primeros 5 registros de datos simulados:")
        print(datos[:5])
 
        # Paso 2: Entrenar modelo
        modelo = ModeloSegmentacionClientes(n_clusters=3)
        modelo.entrenar(datos)
 
        # Paso 4: Predecir un nuevo cliente
        cliente_nuevo = [2000, 10, 12]  # Monto gastado, frecuencia, categorías preferidas
        cluster = modelo.predecir(cliente_nuevo)
        print(f"El nuevo cliente pertenece al cluster: {cluster}")
 
        # Paso 5: Graficar
        etiquetas = modelo.modelo.labels_
        plt.figure(figsize=(10, 6))
        plt.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='viridis', s=50)
        plt.xlabel("Monto gastado")
        plt.ylabel("Frecuencia de compras")
        plt.title("Visualización de Clusters de Clientes")
        plt.colorbar(label="Cluster")
        plt.grid(True)
        plt.show()


    #test = TestSegmentacionClientes()
    #test.ejecutar()
