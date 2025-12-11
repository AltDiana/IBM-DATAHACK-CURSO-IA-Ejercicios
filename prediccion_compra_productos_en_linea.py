import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def generar_datos_compras(num_muestras):
    datos = []
    etiquetas = []
    
    for _ in range(num_muestras):
        num_paginas_vistas = random.randint(1, 20)
        tiempo_en_sitio = round(random.uniform(0, 30), 2)
        
        if num_paginas_vistas > 5 and tiempo_en_sitio > 10:
            etiqueta = 1
        else:
            etiqueta = 0
        
        datos.append([num_paginas_vistas, tiempo_en_sitio])
        etiquetas.append(etiqueta)
    
    return np.array(datos), np.array(etiquetas)
    
def entrenar_modelo(datos, etiquetas):
    X_train, X_test, y_train, y_test = train_test_split(
        datos, etiquetas, test_size=0.3, random_state=42)
    
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {precision:.2f}")
    
    return modelo

def evaluar_modelo(modelo, datos, etiquetas):
    X_train, X_test, y_train, y_test = train_test_split(
        datos, etiquetas, test_size=0.3, random_state=42)
    
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión en el conjunto de prueba: {precision:.2f}")
    
#Usamos el modelo para predecir si un nuevo visitante comprará o no:
def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
    caracteristicas = np.array([[num_paginas_vistas, tiempo_en_sitio]])
    prediccion = modelo.predict(caracteristicas)
    
    if prediccion == 1:
        return "El usuario comprará el producto."
    else:
        return "El usuario no comprará el producto."
        
def graficar_datos(datos, etiquetas):
    plt.figure(figsize=(8, 6))
 
    etiquetas = np.array(etiquetas)
 
    compraron = etiquetas == 1
    plt.scatter(
        datos[compraron, 0],
        datos[compraron, 1],
        color='green',
        label='Compró',
        alpha=0.6
    )
 
    no_compraron = etiquetas == 0
    plt.scatter(
        datos[no_compraron, 0],
        datos[no_compraron, 1],
        color='red',
        label='No compró',
        alpha=0.6
    )
 
    plt.xlabel("Número de páginas vistas")
    plt.ylabel("Tiempo en el sitio (minutos)")
    plt.title("Comportamiento de usuarios y decisión de compra")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def graficar_funcion_prediccion(modelo):
    paginas = np.arange(1, 21)
    tiempo_fijo = 15  # tiempo constante
    
    entradas = np.array([[p, tiempo_fijo] for p in paginas])
    probabilidades = modelo.predict_proba(entradas)[:, 1]
    
    plt.figure(figsize=(8, 5))
    plt.plot(paginas, probabilidades, marker='o', color='green')
    plt.title("Probabilidad de compra según páginas vistas (tiempo fijo = 15 min)")
    plt.xlabel("Páginas vistas")
    plt.ylabel("Probabilidad de compra")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()
