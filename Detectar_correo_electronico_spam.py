import numpy as np
import pandas as pd
from sklearn.svm import SVC

# ---------------------------------------------------
# 1. Generar datos sintéticos para emails
# ---------------------------------------------------
def generar_datos_emails(num_muestras):
    np.random.seed(42)  # Para reproducibilidad

    longitud_mensaje = np.random.randint(50, 501, num_muestras)
    frecuencia_palabra_clave = np.random.rand(num_muestras)
    cantidad_enlaces = np.random.randint(0, 11, num_muestras)

    # Regla simple para etiquetas (SPAM = 1)
    etiqueta_spam = (
        (frecuencia_palabra_clave > 0.6) |
        (cantidad_enlaces > 5) |
        (longitud_mensaje < 120)
    ).astype(int)

    datos = pd.DataFrame({
        "longitud_mensaje": longitud_mensaje,
        "frecuencia_palabra_clave": frecuencia_palabra_clave,
        "cantidad_enlaces": cantidad_enlaces
    })

    return datos, etiqueta_spam


# ---------------------------------------------------
# 2. Entrenar modelo SVM
# ---------------------------------------------------
def entrenar_modelo_svm(datos, etiquetas):
    # SVM lineal rápido, sin probabilities (más ligero)
    modelo = SVC(kernel='linear')
    modelo.fit(datos, etiquetas)
    return modelo


# ---------------------------------------------------
# 3. Predecir si un email es SPAM o NO SPAM
# ---------------------------------------------------
def predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces):
    nuevo = np.array([[longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces]])
    pred = modelo.predict(nuevo)[0]
    return "El email es Spam" if pred == 1 else "El email no es Spam"


# ---------------------------------------------------
# 4. Pipeline completo
# ---------------------------------------------------
# Generar datos
datos, etiquetas = generar_datos_emails(300)   # tamaño reducido = evita timeout

# Entrenar modelo
modelo_svm = entrenar_modelo_svm(datos, etiquetas)

# Ejemplo de predicción:
resultado = predecir_email(
    modelo_svm,
    longitud_mensaje=90,
    frecuencia_palabra_clave=0.75,
    cantidad_enlaces=4
)

print("Resultado del email:", resultado)
