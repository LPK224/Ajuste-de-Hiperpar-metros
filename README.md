# Heart Failure Prediction - Ajuste de Hiperparámetros

## 📊 Análisis Exploratorio de Datos (EDA) - Predicción de Falla Cardíaca

### 📌 Descripción del Proyecto
EDA completo y pipeline de machine learning del dataset "Heart Failure Prediction" que contiene 12 variables clínicas para 918 pacientes. El objetivo es desarrollar y comparar modelos predictivos para diagnóstico de enfermedad cardíaca implementando diferentes técnicas de ajuste de hiperparámetros.

---

## 🔍 Hallazgos Clave del EDA

### 1. **Calidad de Datos Excepcional**
- **Dataset completo**: 0 valores nulos en todas las variables
- **Balance moderado**: 55.3% con enfermedad cardíaca vs 44.7% sanos
- **Consistencia médica**: Valores dentro de rangos clínicos esperados

### 2. **Distribución de Variables**
- **7 variables numéricas**: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, HeartDisease
- **5 variables categóricas**: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
- **Edad promedio**: 53.5 años (rango: 28-77 años)

### 3. **Preprocesamiento Implementado**
```python
# Codificación de variables categóricas
Sex: F=0, M=1
ChestPainType: ASY=0, ATA=1, NAP=2, TA=3
RestingECG: LVH=0, Normal=1, ST=2
ExerciseAngina: N=0, Y=1
ST_Slope: Down=0, Flat=1, Up=2
```

---

## 🛠️ Metodología de Modelado

### **Pipeline de Machine Learning**
1. **Preprocesamiento**: StandardScaler para todas las características
2. **Partición**: 80% entrenamiento (734 muestras) - 20% prueba (184 muestras)
3. **Estratificación**: Proporción balanceada mantenida en ambos conjuntos

### **Técnicas de Ajuste de Hiperparámetros**
- **Modelo Baseline**: Logistic Regression (parámetros por defecto)
- **Búsqueda Aleatoria**: 250 combinaciones evaluadas con RandomizedSearchCV
- **Optimización Bayesiana**: 50 trials con Optuna

---

## 📈 Resultados de Modelado

### **Comparación de Desempeño**

| Modelo | ROC-AUC (CV) | ROC-AUC (Test) | Accuracy |
|--------|--------------|----------------|----------|
| Baseline (Logistic Regression) | 0.9126 | 0.8971 | 0.8696 |
| **Random Forest (Búsqueda Aleatoria)** | **0.9328** | **0.9297** | **0.8913** |
| Random Forest (Optuna) | 0.9337 | 0.9265 | 0.8804 |

### **Mejora Lograda**
- **+0.0326 puntos en ROC-AUC** vs baseline
- **+2.17% en accuracy** vs baseline
- **Reducción de variabilidad** en validación cruzada

---

## 🎯 Características Más Importantes

### **Top 5 Predictores de Enfermedad Cardíaca**

| Variable | Importancia | Significado Clínico |
|----------|-------------|---------------------|
| **ST_Slope** | 31.0% | Pendiente del segmento ST en ECG |
| **ChestPainType** | 15.2% | Tipo de dolor torácico |
| **ExerciseAngina** | 10.4% | Angina inducida por ejercicio |
| **Oldpeak** | 9.8% | Depresión del ST inducida por ejercicio |
| **MaxHR** | 9.4% | Frecuencia cardíaca máxima alcanzada |

---

## 💡 Conclusiones Principales

### **1. Efectividad del Ajuste de Hiperparámetros**
- La optimización mejoró significativamente el desempeño predictivo
- Random Forest superó consistentemente a Logistic Regression
- Ambos métodos (búsqueda aleatoria y Optuna) demostraron utilidad

### **2. Relevancia Clínica**
- Las características más importantes coinciden con factores de riesgo médicos establecidos
- Variables de electrocardiograma (ST_Slope) son los predictores más fuertes
- El modelo muestra excelente balance entre precision y recall

### **3. Capacidad Predictiva**
- **ROC-AUC de 0.9297** en datos no vistos indica alta capacidad discriminativa
- **F1-score balanceado**: 0.87 (sanos) y 0.90 (enfermos)
- Modelo confiable para asistencia en diagnóstico médico

---

## 🚀 Cómo Reproducir el Análisis

### **Requisitos**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn optuna
```

### **Estructura del Proyecto**
```
heart-failure-prediction/
│
├── data/
│   └── heart.csv                    # Dataset original
├── notebooks/
│   └── heart_failure_analysis.ipynb # Análisis completo
├── src/
│   ├── data_loader.py              # Carga de datos
│   ├── preprocessing.py            # Preprocesamiento
│   └── model_training.py           # Entrenamiento de modelos
├── results/
│   ├── eda_visualizations/         # Gráficos del EDA
│   ├── model_performance/          # Métricas de modelos
│   └── feature_importance/         # Análisis de características
└── README.md
```
---

## 📚 Referencias

- [Dataset Original - Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
---
