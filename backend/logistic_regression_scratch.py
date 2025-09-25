# -*- coding: utf-8 -*-
"""
Implementación desde cero de Regresión Logística
Python 3.7.x compatible
"""

import numpy as np
import matplotlib
# Configurar backend para entornos sin display gráfico
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import pickle
import warnings
warnings.filterwarnings('ignore')

class ScratchLogisticRegression():
    """
    Implementación desde cero de Regresión Logística
    """

    def __init__(self, num_iter=1000, lr=0.01, bias=True, verbose=False, lambda_reg=0.1):
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.lambda_reg = lambda_reg
        
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None
        
    def _add_bias(self, X):
        """Añade columna de bias (intercept)"""
        if self.bias:
            return np.c_[np.ones((X.shape[0], 1)), X]
        return X
    
    def _sigmoid(self, z):
        """Función sigmoide - Problema 1"""
        z_clipped = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_clipped))
    
    def _hypothesis(self, X):
        """Función hipótesis - Problema 1"""
        z = np.dot(X, self.coef_)
        return self._sigmoid(z)
    
    def _compute_loss(self, X, y):
        """Función de costo - Problema 4"""
        m = len(y)
        h = self._hypothesis(X)
        
        epsilon = 1e-8
        error_term = -y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)
        
        if self.bias:
            reg_term = (self.lambda_reg / (2.0 * m)) * np.sum(self.coef_[1:]**2)
        else:
            reg_term = (self.lambda_reg / (2.0 * m)) * np.sum(self.coef_**2)
        
        return np.mean(error_term) + reg_term
    
    def _gradient_descent(self, X, y):
        """Descenso de gradiente - Problema 2"""
        m = len(y)
        h = self._hypothesis(X)
        error = h - y
        gradient = np.dot(X.T, error) / m
        
        if self.bias:
            reg_gradient = (self.lambda_reg / m) * self.coef_
            reg_gradient[0] = 0
        else:
            reg_gradient = (self.lambda_reg / m) * self.coef_
        
        gradient += reg_gradient
        self.coef_ -= self.lr * gradient
        return gradient
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Entrena el modelo"""
        X_with_bias = self._add_bias(X)
        n_features = X_with_bias.shape[1]
        
        np.random.seed(42)
        self.coef_ = np.random.normal(0, 0.01, n_features)
        
        if X_val is not None and y_val is not None:
            X_val_with_bias = self._add_bias(X_val)
        else:
            X_val_with_bias = None
        
        for i in range(self.iter):
            self._gradient_descent(X_with_bias, y)
            self.loss[i] = self._compute_loss(X_with_bias, y)
            
            if X_val_with_bias is not None and y_val is not None:
                self.val_loss[i] = self._compute_loss(X_val_with_bias, y_val)
            
            if self.verbose and i % 100 == 0:
                if X_val is not None and y_val is not None:
                    print("Iteración {}: Loss_train={:.4f}, Loss_val={:.4f}".format(
                        i, self.loss[i], self.val_loss[i]))
                else:
                    print("Iteración {}: Loss={:.4f}".format(i, self.loss[i]))
    
    def predict_proba(self, X):
        """Predice probabilidades - Problema 3"""
        X_with_bias = self._add_bias(X)
        return self._hypothesis(X_with_bias)
    
    def predict(self, X, threshold=0.5):
        """Predice etiquetas - Problema 3"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def save_weights(self, filename):
        """Guarda los pesos del modelo"""
        if self.coef_ is None:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        weights_dict = {
            'coef': self.coef_,
            'bias': self.bias,
            'lambda_reg': self.lambda_reg
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(weights_dict, f)
        print("Pesos guardados en {}".format(filename))
    
    def load_weights(self, filename):
        """Carga los pesos del modelo"""
        with open(filename, 'rb') as f:
            weights_dict = pickle.load(f)
        
        self.coef_ = weights_dict['coef']
        self.bias = weights_dict['bias']
        self.lambda_reg = weights_dict.get('lambda_reg', 0.1)
        print("Pesos cargados desde {}".format(filename))

# =============================================================================
# PROBLEMA 1: Demostración Función Hipótesis
# =============================================================================

def problema_1():
    """Demuestra la función hipótesis y sigmoide"""
    print("=" * 70)
    print("PROBLEMA 1: Función Hipótesis y Sigmoide")
    print("=" * 70)
    
    model = ScratchLogisticRegression()
    
    print("\n1. FUNCIÓN SIGMOIDE:")
    print("g(z) = 1 / (1 + e^(-z))")
    
    z_values = np.array([-10, -5, -1, 0, 1, 5, 10])
    sigmoid_values = model._sigmoid(z_values)
    
    for z, sig in zip(z_values, sigmoid_values):
        print("g({:3d}) = {:.6f}".format(z, sig))
    
    print("\n2. FUNCIÓN HIPÓTESIS:")
    print("hθ(x) = g(θᵀ·x) = 1 / (1 + e^(-θᵀ·x))")
    
    return model

# =============================================================================
# PROBLEMA 2: Demostración Descenso de Gradiente
# =============================================================================

def problema_2():
    """Demuestra el descenso de gradiente"""
    print("\n" + "=" * 70)
    print("PROBLEMA 2: Descenso de Gradiente")
    print("=" * 70)
    
    print("Fórmula del descenso de gradiente:")
    print("θⱼ := θⱼ - α * ∂J(θ)/∂θⱼ")
    print("Donde ∂J(θ)/∂θⱼ = (1/m) * Σ[(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾] + (λ/m)*θⱼ (para j ≥ 1)")
    
    return None

# =============================================================================
# PROBLEMA 3: Demostración Estimación
# =============================================================================

def problema_3():
    """Demuestra las funciones de predicción"""
    print("\n" + "=" * 70)
    print("PROBLEMA 3: Estimación (predict y predict_proba)")
    print("=" * 70)
    
    print("1. predict_proba(): Devuelve probabilidades P(y=1|x)")
    print("2. predict(): Aplica umbral (default 0.5) para obtener etiquetas binarias")
    
    return None

# =============================================================================
# PROBLEMA 4: Demostración Función de Costo
# =============================================================================

def problema_4():
    """Demuestra la función de costo"""
    print("\n" + "=" * 70)
    print("PROBLEMA 4: Función de Costo")
    print("=" * 70)
    
    print("Fórmula de la función de costo:")
    print("J(θ) = -(1/m) * Σ[y⁽ⁱ⁾log(hθ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-hθ(x⁽ⁱ⁾))] + (λ/2m) * Σθⱼ²")
    
    return None

# =============================================================================
# PROBLEMA 5: Entrenamiento y evaluación
# =============================================================================

def problema_5():
    """Entrenamiento y evaluación con dataset Iris"""
    print("\n" + "=" * 70)
    print("PROBLEMA 5: Entrenamiento y evaluación")
    print("=" * 70)
    
    # Cargar datos
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Filtrar para clasificación binaria (clases 1 y 2)
    mask = (y == 1) | (y == 2)
    X_binary = X[mask]
    y_binary = y[mask] - 1
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    # Nuestro modelo
    our_model = ScratchLogisticRegression(
        num_iter=1500, 
        lr=0.1,
        verbose=True, 
        lambda_reg=0.1
    )
    
    our_model.fit(X_train, y_train, X_test, y_test)
    y_pred_our = our_model.predict(X_test)
    
    # Modelo de scikit-learn
    sklearn_model = SklearnLogisticRegression(random_state=42, C=1.0/0.1, solver='lbfgs')
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    # Métricas
    accuracy_our = accuracy_score(y_test, y_pred_our)
    precision_our = precision_score(y_test, y_pred_our, zero_division=0)
    recall_our = recall_score(y_test, y_pred_our, zero_division=0)
    
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    precision_sklearn = precision_score(y_test, y_pred_sklearn, zero_division=0)
    recall_sklearn = recall_score(y_test, y_pred_sklearn, zero_division=0)
    
    print("\nCOMPARACIÓN DE RESULTADOS:")
    print("NUESTRO MODELO - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
        accuracy_our, precision_our, recall_our))
    print("SCI-KIT LEARN - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
        accuracy_sklearn, precision_sklearn, recall_sklearn))
    
    return our_model, X_train, X_test, y_train, y_test, X_binary, y_binary

# =============================================================================
# PROBLEMA 6: Curvas de aprendizaje
# =============================================================================

def problema_6(model):
    """Grafica curvas de aprendizaje"""
    print("\n" + "=" * 70)
    print("PROBLEMA 6: Curvas de aprendizaje")
    print("=" * 70)
    
    # Crear y guardar gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss, label='Pérdida entrenamiento', linewidth=2)
    if np.any(model.val_loss != 0):
        plt.plot(model.val_loss, label='Pérdida validación', linewidth=2)
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.title('Curvas de Aprendizaje - Regresión Logística')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Guardar la gráfica
    plt.savefig('curvas_aprendizaje.png', dpi=300, bbox_inches='tight')
    plt.close()  # Cerrar la figura para liberar memoria
    
    print("✓ Gráfica de curvas de aprendizaje guardada como 'curvas_aprendizaje.png'")
    print("Pérdida inicial: {:.4f}".format(model.loss[0]))
    print("Pérdida final: {:.4f}".format(model.loss[-1]))
    print("Reducción: {:.2f}%".format((1 - model.loss[-1]/model.loss[0]) * 100))

# =============================================================================
# PROBLEMA 7: Visualización de región de decisión
# =============================================================================

def problema_7(X_binary, y_binary):
    """Visualiza región de decisión"""
    print("\n" + "=" * 70)
    print("PROBLEMA 7: Región de decisión")
    print("=" * 70)
    
    # Usar solo 2 características para visualización
    X_2d = X_binary[:, [1, 2]]  # sepal width y petal length
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    
    # Entrenar modelo con 2 características
    model_2d = ScratchLogisticRegression(num_iter=2000, lr=0.1, verbose=False)
    model_2d.fit(X_train, y_train)
    
    # Crear meshgrid para región de decisión
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predecir en el grid
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Crear y guardar gráfica
    plt.figure(figsize=(10, 8))
    
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_binary, 
                         cmap=cmap_bold, edgecolors='k', s=50)
    plt.colorbar(scatter)
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Región de Decisión - Regresión Logística')
    
    # Guardar la gráfica
    plt.savefig('region_decision.png', dpi=300, bbox_inches='tight')
    plt.close()  # Cerrar la figura
    
    print("✓ Gráfica de región de decisión guardada como 'region_decision.png'")
    print("Accuracy en datos 2D: {:.4f}".format(
        accuracy_score(y_test, model_2d.predict(X_test))))

# =============================================================================
# PROBLEMA 8: Guardar y cargar pesos
# =============================================================================

def problema_8(model, X_test, y_test):
    """Demo de guardar y cargar pesos"""
    print("\n" + "=" * 70)
    print("PROBLEMA 8: Guardar y cargar pesos")
    print("=" * 70)
    
    # Guardar modelo
    model.save_weights('logistic_regression_weights.pkl')
    
    # Crear nuevo modelo y cargar pesos
    new_model = ScratchLogisticRegression()
    new_model.load_weights('logistic_regression_weights.pkl')
    
    # Verificar que las predicciones son iguales
    y_pred_original = model.predict(X_test)
    y_pred_loaded = new_model.predict(X_test)
    
    accuracy_original = accuracy_score(y_test, y_pred_original)
    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
    
    print("Accuracy modelo original: {:.4f}".format(accuracy_original))
    print("Accuracy modelo cargado:  {:.4f}".format(accuracy_loaded))
    print("¿Predicciones iguales? {}".format(np.array_equal(y_pred_original, y_pred_loaded)))

# =============================================================================
# FUNCIÓN PARA MOSTRAR GRÁFICAS EN CODESPACES
# =============================================================================

def mostrar_graficas():
    """Intenta mostrar las gráficas si el entorno lo permite"""
    try:
        # Intentar usar backend interactivo
        import matplotlib.pyplot as plt
        from IPython.display import display, Image
        
        print("\n" + "=" * 70)
        print("INTENTANDO MOSTRAR GRÁFICAS...")
        print("=" * 70)
        
        # Mostrar curvas de aprendizaje
        try:
            img1 = Image(filename='curvas_aprendizaje.png')
            display(img1)
            print("✓ Gráfica 1 mostrada: Curvas de Aprendizaje")
        except:
            print("✗ No se pudo mostrar la gráfica 1")
        
        # Mostrar región de decisión
        try:
            img2 = Image(filename='region_decision.png')
            display(img2)
            print("✓ Gráfica 2 mostrada: Región de Decisión")
        except:
            print("✗ No se pudo mostrar la gráfica 2")
            
    except ImportError:
        print("\nPara ver las gráficas interactivamente, ejecuta en Jupyter Notebook")
        print("Las gráficas se han guardado como archivos PNG:")
        print(" - curvas_aprendizaje.png")
        print(" - region_decision.png")

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    """Función principal para ejecutar todos los problemas"""
    print("INICIANDO IMPLEMENTACIÓN DE REGRESIÓN LOGÍSTICA DESDE CERO")
    print("=" * 70)
    
    try:
        # Ejecutar todos los problemas
        problema_1()
        problema_2()
        problema_3()
        problema_4()
        
        model, X_train, X_test, y_train, y_test, X_binary, y_binary = problema_5()
        problema_6(model)
        problema_7(X_binary, y_binary)
        problema_8(model, X_test, y_test)
        
        # Intentar mostrar gráficas
        mostrar_graficas()
        
        print("\n" + "=" * 70)
        print("¡TODOS LOS PROBLEMAS COMPLETADOS EXITOSAMENTE!")
        print("=" * 70)
        print("\nARCHIVOS GENERADOS:")
        print(" - curvas_aprendizaje.png (Problema 6)")
        print(" - region_decision.png (Problema 7)") 
        print(" - logistic_regression_weights.pkl (Problema 8)")
        
    except Exception as e:
        print("Error durante la ejecución:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()