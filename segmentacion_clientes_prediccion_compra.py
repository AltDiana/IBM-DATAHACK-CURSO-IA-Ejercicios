import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================
# 1. GENERADOR DE DATOS
# ============================================================
class CustomerDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
 
    def generate(self, n=300):
        total_spent = np.random.uniform(50, 1500, n)
        total_purchases = np.random.randint(1, 51, n)
        purchase_frequency = np.random.uniform(0.5, 10, n)
 
        will_buy_next_month = []
        for spent, freq in zip(total_spent, purchase_frequency):
            if spent > 500 and freq > 4:
                will_buy_next_month.append(1)
            else:
                will_buy_next_month.append(0)
 
        data = pd.DataFrame({
            'total_spent': total_spent,
            'total_purchases': total_purchases,
            'purchase_frequency': purchase_frequency,
            'will_buy_next_month': will_buy_next_month
        })
 
        return data


# ============================================================
# 2. MODELO DE SEGMENTACIÓN Y PREDICCIÓN
# ============================================================
class CustomerSegmentationModel:
    def __init__(self, data):
        self.data = data.copy()
        self.kmeans = None
        self.model = None
        self.accuracy = None
        self.conf_matrix = None

    def segment_customers(self, n_clusters=3):
        X_clustering = self.data[['total_spent', 'total_purchases', 'purchase_frequency']]
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.data['customer_segment'] = self.kmeans.fit_predict(X_clustering)

    def train_model(self):
        data_encoded = pd.get_dummies(self.data, columns=['customer_segment'], drop_first=True)
    
        X = data_encoded.drop(columns=['will_buy_next_month'])
        y = data_encoded['will_buy_next_month']
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X_train, y_train)
    
        y_pred = self.model.predict(X_test)
    
        self.accuracy = accuracy_score(y_test, y_pred)
        self.conf_matrix = confusion_matrix(y_test, y_pred)
        
    def get_accuracy(self):
        return self.accuracy
 
    def get_confusion_matrix(self):
        return self.conf_matrix
    
# ============================================================
# 3. VISUALIZACIONES
# ============================================================
def graficar_segmentos(data):
    plt.figure(figsize=(8, 6))
    colores = ['red', 'green', 'blue', 'purple', 'orange']
    segmentos = data['customer_segment'].unique()
 
    for seg in segmentos:
        grupo = data[data['customer_segment'] == seg]
        plt.scatter(grupo['total_spent'], grupo['purchase_frequency'],
                    label=f'Segmento {seg}', alpha=0.6, color=colores[seg % len(colores)])
 
    plt.xlabel("Total gastado (€)")
    plt.ylabel("Frecuencia de compra (mensual)")
    plt.title("Segmentación de clientes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graficar_probabilidad_compra(modelo):
    total_spent = np.linspace(50, 1500, 100)
    total_purchases = np.full(100, 25)
    purchase_frequency = np.full(100, 5)
 
    X_input = pd.DataFrame({
        'total_spent': total_spent,
        'total_purchases': total_purchases,
        'purchase_frequency': purchase_frequency
    })
 
    X_input['customer_segment_1'] = 0
    X_input['customer_segment_2'] = 0
 
    probs = modelo.predict_proba(X_input)[:, 1]
 
    plt.figure(figsize=(8, 5))
    plt.plot(total_spent, probs, color='green', marker='o', markersize=4)
    plt.title("Probabilidad de compra según total gastado")
    plt.xlabel("Total gastado (€)")
    plt.ylabel("Probabilidad de compra")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================================
# 4. EJEMPLO DE USO
# ============================================================
    generador = CustomerDataGenerator()
    datos_clientes = generador.generate(300)
     
    modelo = CustomerSegmentationModel(datos_clientes)
    modelo.segment_customers()
    modelo.train_model()
     
    print("Precisión del modelo:", modelo.get_accuracy())
    print("Matriz de confusión:\n", modelo.get_confusion_matrix())
     
    graficar_segmentos(modelo.data)
    
    graficar_probabilidad_compra(modelo.model)