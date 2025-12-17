export interface CodeExample {
  title: string;
  code: string;
  explanation: string;
}

export interface Stage {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  description: string;
  duration: string;
  objectives: string[];
  keyTopics: string[];
  practicalExamples: CodeExample[];
  resources: { title: string; url: string }[];
  proTips: string[];
}

export const curriculum: Stage[] = [
  {
    id: 'python-basics',
    icon: '🐍',
    title: 'Python Esencial',
    subtitle: 'Fundamentos del lenguaje',
    description:
      'Domina los fundamentos de Python necesarios para Machine Learning. Aprende estructuras de datos, funciones, programación orientada a objetos y manejo de errores.',
    duration: '2-3 semanas',
    objectives: [
      'Escribir código Python limpio y eficiente',
      'Trabajar con listas, diccionarios y comprensiones',
      'Crear funciones y clases reutilizables',
      'Manejar errores y excepciones correctamente',
    ],
    keyTopics: [
      'Variables, tipos de datos y operadores',
      'Estructuras de control (if, for, while)',
      'Funciones y lambda expressions',
      'Listas, tuplas, sets y diccionarios',
      'Comprensiones de listas y diccionarios',
      'Programación orientada a objetos básica',
      'Manejo de archivos y excepciones',
    ],
    practicalExamples: [
      {
        title: 'Comprensión de Listas',
        code: `# Filtrar y transformar datos
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Obtener cuadrados de números pares
squares = [x**2 for x in numbers if x % 2 == 0]
print(squares)  # [4, 16, 36, 64, 100]

# Crear diccionario de números y sus cuadrados
num_squares = {x: x**2 for x in numbers}
print(num_squares)`,
        explanation:
          'Las comprensiones son fundamentales en Python para procesar datos de forma concisa. Las usarás constantemente en ML.',
      },
      {
        title: 'Clases y Objetos',
        code: `class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)
    
    def process(self):
        """Procesa los datos almacenados"""
        return [x * 2 for x in self.data]

# Uso
processor = DataProcessor("MyProcessor")
processor.add_data(10)
processor.add_data(20)
print(processor.process())  # [20, 40]`,
        explanation:
          'Las clases te permiten organizar código. Frameworks como PyTorch usan POO extensivamente.',
      },
    ],
    resources: [
      { title: 'Python.org Tutorial Oficial', url: 'https://docs.python.org/3/tutorial/' },
      { title: 'Real Python - Beginner Tutorials', url: 'https://realpython.com/tutorials/basics/' },
    ],
    proTips: [
      'Las comprensiones de listas son 2-3x más rápidas que loops equivalentes',
      'Usa f-strings (f"...") para formatear texto - son más legibles y rápidas',
      'Los diccionarios son perfectos para mapear categorías en ML',
      'Practica debugging con print() y type() para entender el flujo de datos'
    ],
  },
  {
    id: 'numpy-matplotlib',
    icon: '�',
    title: 'NumPy & Matplotlib',
    subtitle: 'Computación numérica y visualización',
    description:
      'NumPy es la base de todo el ecosistema de ML en Python. Aprende a manipular arrays multidimensionales y crear visualizaciones con Matplotlib.',
    duration: '2-3 semanas',
    objectives: [
      'Trabajar eficientemente con arrays de NumPy',
      'Realizar operaciones vectorizadas y broadcasting',
      'Crear visualizaciones claras con Matplotlib',
      'Preparar datos para Machine Learning',
    ],
    keyTopics: [
      'Arrays de NumPy y operaciones básicas',
      'Indexing, slicing y fancy indexing',
      'Broadcasting y operaciones vectorizadas',
      'Funciones universales (ufuncs)',
      'Álgebra lineal básica',
      'Gráficos de líneas, barras y scatter plots',
      'Customización de plots (colores, etiquetas, leyendas)',
    ],
    practicalExamples: [
      {
        title: 'Broadcasting en NumPy',
        code: `import numpy as np

# Normalizar datos (mean=0, std=1)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Calcular media y std por columna
mean = data.mean(axis=0)  # [4. 5. 6.]
std = data.std(axis=0)    # [2.45 2.45 2.45]

# Normalizar usando broadcasting
normalized = (data - mean) / std
print(normalized)
# [[-1.22 -1.22 -1.22]
#  [ 0.    0.    0.  ]
#  [ 1.22  1.22  1.22]]`,
        explanation:
          'Broadcasting permite operaciones entre arrays de diferentes formas. Esencial para preprocesamiento de datos.',
      },
      {
        title: 'Visualización de Datos',
        code: `import matplotlib.pyplot as plt
import numpy as np

# Generar datos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Crear visualización
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='#ff6b35', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='#4ecdc4', linewidth=2)
plt.title('Funciones Trigonométricas', fontsize=16)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()`,
        explanation:
          'Matplotlib es crucial para visualizar resultados de modelos y entender datos. Base para exploración de datos.',
      },
    ],
    resources: [
      { title: 'NumPy Documentation', url: 'https://numpy.org/doc/stable/' },
      { title: 'Matplotlib Tutorials', url: 'https://matplotlib.org/stable/tutorials/index.html' },
    ],    proTips: [
      'Usa operaciones vectorizadas de NumPy - son hasta 100x más rápidas que loops',
      'Broadcasting automático evita código repetitivo: (3,1) + (3,) funciona',
      'plt.style.use("seaborn") mejora visualizaciones sin esfuerzo extra',
      'Siempre normaliza datos con (x - mean) / std antes de entrenar modelos'
    ],  },
  {
    id: 'pandas',
    icon: '🐼',
    title: 'Pandas',
    subtitle: 'Manipulación de datos',
    description:
      'Pandas es la herramienta principal para limpiar, transformar y analizar datos tabulares. Aprende a trabajar con DataFrames y preparar datos para modelos.',
    duration: '2-3 semanas',
    objectives: [
      'Cargar y explorar datasets reales',
      'Limpiar datos faltantes o incorrectos',
      'Transformar y agrupar datos eficientemente',
      'Crear features para Machine Learning',
    ],
    keyTopics: [
      'DataFrames y Series',
      'Carga de datos (CSV, Excel, JSON)',
      'Filtrado, selección y indexación',
      'Manejo de valores faltantes',
      'Agrupación (groupby) y agregaciones',
      'Merge, join y concatenación',
      'Análisis exploratorio de datos (EDA)',
    ],
    practicalExamples: [
      {
        title: 'Limpieza de Datos',
        code: `import pandas as pd
import numpy as np

# Cargar dataset
df = pd.DataFrame({
    'edad': [25, 30, np.nan, 35, 28],
    'salario': [50000, 60000, 55000, np.nan, 52000],
    'ciudad': ['NY', 'LA', 'NY', 'LA', 'NY']
})

# Rellenar valores faltantes
df['edad'].fillna(df['edad'].mean(), inplace=True)
df['salario'].fillna(df['salario'].median(), inplace=True)

# Crear nueva feature
df['salario_por_edad'] = df['salario'] / df['edad']

print(df)`,
        explanation:
          'La limpieza de datos es el 80% del trabajo en ML. Pandas hace este proceso manejable.',
      },
      {
        title: 'Análisis Exploratorio',
        code: `import pandas as pd
import numpy as np

# Crear dataset de ejemplo
data = {
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Valencia', 'Barcelona'],
    'edad': [25, 30, 35, np.nan, 28],
    'salario': [30000, 45000, np.nan, 35000, 42000]
}
df = pd.DataFrame(data)

# Exploración básica
print('Dataset:')
print(df)
print('\\nInformación:')
print(df.info())
print('\\nEstadísticas:')
print(df.describe())

# Agrupar y analizar
city_stats = df.groupby('ciudad').agg({
    'salario': ['mean', 'count'],
    'edad': 'mean'
})

print('\\nEstadísticas por ciudad:')
print(city_stats)

# Filtrar salarios altos
high_earners = df[df['salario'] > 40000]
print(f"\\nPersonas con salario > 40k: {len(high_earners)}")`,
        explanation:
          'EDA te ayuda a entender tus datos antes de entrenar modelos. Crucial para feature engineering.',
      },
    ],
    resources: [
      { title: 'Pandas Documentation', url: 'https://pandas.pydata.org/docs/' },
      { title: 'Pandas Cheat Sheet', url: 'https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf' },
    ],
    proTips: [
      'df.describe() + df.info() son tus primeros pasos en CUALQUIER dataset',
      'groupby() es la clave para feature engineering: agrupa y crea nuevas columnas',
      'Usa .isnull().sum() para detectar valores faltantes antes de entrenar',
      'pd.get_dummies() convierte categorías a números automáticamente para ML'
    ],
  },
  {
    id: 'scikit-learn',
    icon: '🤖',
    title: 'scikit-learn',
    subtitle: 'Machine Learning clásico',
    description:
      'Aprende los fundamentos de Machine Learning con scikit-learn: clasificación, regresión, clustering y evaluación de modelos.',
    duration: '4-5 semanas',
    objectives: [
      'Entrenar modelos de clasificación y regresión',
      'Evaluar modelos correctamente (train/test split, cross-validation)',
      'Realizar feature engineering y scaling',
      'Optimizar hiperparámetros',
    ],
    keyTopics: [
      'Regresión lineal y logística',
      'Árboles de decisión y Random Forests',
      'Support Vector Machines (SVM)',
      'K-Nearest Neighbors (KNN)',
      'Clustering (K-Means, DBSCAN)',
      'Preprocesamiento (StandardScaler, OneHotEncoder)',
      'Validación cruzada y métricas de evaluación',
      'Pipeline y GridSearchCV',
    ],
    practicalExamples: [
      {
        title: 'Clasificación con Random Forest',
        code: `from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))`,
        explanation:
          'Random Forest es uno de los modelos más usados. Funciona bien out-of-the-box en muchos problemas.',
      },
      {
        title: 'Pipeline Completo',
        code: `from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris

# Cargar y preparar datos
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Crear pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Definir hiperparámetros a optimizar
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'linear']
}

# Grid search con validación cruzada
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor score: {grid_search.best_score_:.2f}")`,
        explanation:
          'Los pipelines aseguran que el preprocesamiento se aplique correctamente en train y test.',
      },
    ],
    resources: [
      { title: 'scikit-learn Documentation', url: 'https://scikit-learn.org/stable/' },
      { title: 'scikit-learn User Guide', url: 'https://scikit-learn.org/stable/user_guide.html' },
    ],    proTips: [
      'SIEMPRE divide train/test ANTES de cualquier preprocesamiento para evitar data leakage',
      'Cross-validation (cv=5) da métricas más confiables que un solo train_test_split',
      'Pipeline() automatiza preprocesamiento + modelo en un solo objeto',
      'GridSearchCV encuentra hiperparámetros óptimos automáticamente - úsalo siempre'
    ],  },
  {
    id: 'opencv',
    icon: '👁️',
    title: 'OpenCV',
    subtitle: 'Visión por Computadora clásica',
    description:
      'Domina las técnicas fundamentales de procesamiento de imágenes y visión por computadora con OpenCV: detección de bordes, filtros, transformaciones y detección de objetos.',
    duration: '3-4 semanas',
    objectives: [
      'Cargar, manipular y guardar imágenes',
      'Aplicar filtros y transformaciones',
      'Detectar bordes, contornos y características',
      'Implementar detección de objetos clásica',
    ],
    keyTopics: [
      'Carga y visualización de imágenes',
      'Espacios de color (RGB, HSV, Grayscale)',
      'Filtros (blur, sharpening, edge detection)',
      'Transformaciones geométricas',
      'Detección de contornos',
      'Detección de características (SIFT, ORB)',
      'Detección de rostros (Haar Cascades)',
      'Template matching',
    ],
    practicalExamples: [
      {
        title: 'Detección de Bordes',
        code: `import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

# Descargar imagen de mineral de cobre desde el servidor
url = 'https://ml-cv-roadmap.vercel.app/oxido-de-cobre.jpg'
urllib.request.urlretrieve(url, 'mineral.jpg')
print('✓ Imagen descargada: oxido-de-cobre.jpg')

# Cargar imagen
img = cv2.imread('mineral.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar Gaussian Blur para reducir ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detección de bordes con Canny
edges = cv2.Canny(blurred, 50, 150)

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original - Mineral de Cobre')
axes[0].axis('off')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')
axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Bordes (Canny)')
axes[2].axis('off')
plt.tight_layout()
plt.show()

print(f"Forma de la imagen: {img.shape}")
print(f"Bordes detectados: {np.sum(edges > 0)} píxeles")`,
        explanation:
          'La detección de bordes es fundamental en visión por computadora. Canny es el algoritmo más usado. Usa la imagen del mineral de cobre (roca turquesa) para ver cómo detecta los contornos y texturas naturales.',
      },
      {
        title: 'Detección de Rostros',
        code: `import cv2
import urllib.request
from google.colab.patches import cv2_imshow

# Descargar imagen de familia desde el servidor
url = 'https://ml-cv-roadmap.vercel.app/familia_deteccion_rostro.jpg'
urllib.request.urlretrieve(url, 'familia.jpg')
print('✓ Imagen descargada: familia_deteccion_rostro.jpg')

# Cargar clasificador pre-entrenado
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Cargar imagen
img = cv2.imread('familia.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar rostros con parámetros optimizados
# minNeighbors más alto = menos falsos positivos
# minSize más grande = ignora detecciones pequeñas erróneas
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.2,
    minNeighbors=6,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Dibujar rectángulos en rostros detectados
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 107, 53), 3)

print(f"✓ Rostros detectados: {len(faces)}")
for i, (x, y, w, h) in enumerate(faces):
    print(f"  Rostro {i+1}: posición ({x}, {y}), tamaño {w}x{h}")

# Mostrar resultado
cv2_imshow(img)`,
        explanation:
          'Haar Cascades son un método clásico para detección de objetos. Base para entender detección moderna. Usa la foto familiar para detectar rostros humanos automáticamente.',
      },
    ],
    resources: [
      { title: 'OpenCV Documentation', url: 'https://docs.opencv.org/' },
      { title: 'OpenCV Python Tutorials', url: 'https://docs.opencv.org/master/d6/d00/tutorial_py_root.html' },
    ],
    proTips: [
      'Convierte a escala de grises SIEMPRE que el color no sea importante - acelera 3x',
      'GaussianBlur() antes de Canny reduce ruido y mejora detección de bordes',
      'Haar Cascades son rápidos pero menos precisos que deep learning (YOLO)',
      'Ajusta minNeighbors en detectMultiScale para balance precisión/falsos positivos'
    ],
  },
  {
    id: 'pytorch',
    icon: '🔥',
    title: 'PyTorch Esencial',
    subtitle: 'Deep Learning fundamentals',
    description:
      'Aprende PyTorch, el framework líder en investigación de Deep Learning. Domina tensores, autograd, construcción de redes neuronales y el ciclo de entrenamiento.',
    duration: '4-5 semanas',
    objectives: [
      'Trabajar con tensores de PyTorch y operaciones',
      'Construir redes neuronales con torch.nn',
      'Entrenar modelos con optimizadores y funciones de pérdida',
      'Usar GPU para acelerar entrenamiento',
    ],
    keyTopics: [
      'Tensores y operaciones básicas',
      'Autograd y backpropagation automática',
      'torch.nn: capas, activaciones, pérdidas',
      'torch.optim: optimizadores (SGD, Adam)',
      'DataLoader y Dataset para carga eficiente',
      'Entrenamiento en GPU (CUDA)',
      'Guardar y cargar modelos',
      'Transfer learning básico',
    ],
    practicalExamples: [
      {
        title: 'Red Neuronal Simple',
        code: `# ════════════════════════════════════════════════════════
# ⚠️  IMPORTANTE: PyTorch ya viene en Google Colab
# Si usas otro entorno, ejecuta primero:
# !pip install torch torchvision
# ════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim

# Definir modelo
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Crear modelo
model = SimpleNN(input_size=784, hidden_size=128, num_classes=10)

# Pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)`,
        explanation:
          'Esta es la estructura básica de una red neuronal en PyTorch. nn.Module es la base de todo.',
      },
      {
        title: 'Ciclo de Entrenamiento',
        code: `# ════════════════════════════════════════════════════════
# 📝 EJECUTA EN UNA CELDA SEPARADA (debajo del anterior)
# ════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Crear datos de ejemplo
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
X_test = torch.randn(200, 784)
y_test = torch.randint(0, 10, (200,))

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Modelo, pérdida y optimizador (del ejemplo anterior)
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ENTRENAMIENTO
num_epochs = 3
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

# EVALUACIÓN
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    print(f'\\n✓ Accuracy: {100 * correct / total:.2f}%')`,
        explanation:
          'El ciclo de entrenamiento es el corazón del deep learning. Aprende este patrón: forward pass → calcular pérdida → backward pass → actualizar pesos.',
      },
    ],
    resources: [
      { title: 'PyTorch Documentation', url: 'https://pytorch.org/docs/stable/index.html' },
      { title: 'PyTorch Tutorials', url: 'https://pytorch.org/tutorials/' },
    ],
    proTips: [
      'Usa .to(device) para mover modelo y datos a GPU - acelera 10-100x',
      'DataLoader con batch_size=32-128 optimiza velocidad de entrenamiento',
      'torch.no_grad() en validación ahorra memoria y acelera inferencia',
      'lr_scheduler ajusta learning rate automáticamente para mejor convergencia'
    ],
  },
  {
    id: 'yolo',
    icon: '🎯',
    title: 'YOLO',
    subtitle: 'Detección de Objetos en Tiempo Real',
    description:
      'Domina YOLO (You Only Look Once), el algoritmo líder en detección de objetos en tiempo real. Aprende a entrenar modelos para detectar y clasificar objetos con alta velocidad y precisión.',
    duration: '4-6 semanas',
    objectives: [
      'Entender arquitecturas YOLO (YOLOv5, YOLOv8)',
      'Entrenar modelos YOLO personalizados',
      'Optimizar detección para tiempo real',
      'Implementar detección en imágenes y video',
    ],
    keyTopics: [
      'Arquitectura YOLO y evolución',
      'Detección en una sola pasada (Single-shot detection)',
      'Anchor boxes y predicción de bounding boxes',
      'Non-Maximum Suppression (NMS)',
      'Entrenamiento con datasets personalizados',
      'Transfer learning con modelos pre-entrenados',
      'Optimización para velocidad vs precisión',
      'Implementación en producción',
    ],
    practicalExamples: [
      {
        title: 'Detección con YOLOv8',
        code: `# ════════════════════════════════════════════════════════
# ⚠️  PASO 1: Ejecuta primero en una celda separada
# !pip install -q ultralytics
# ════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════
# 📝 PASO 2: Ejecuta este código en la siguiente celda
# ════════════════════════════════════════════════════════

from ultralytics import YOLO
import cv2
import urllib.request

# Descargar imagen de ejemplo (calle con coches y personas)
url = 'https://ultralytics.com/images/bus.jpg'
urllib.request.urlretrieve(url, 'bus.jpg')
print('✓ Imagen descargada')

# Cargar modelo pre-entrenado YOLOv8
model = YOLO('yolov8n.pt')  # n=nano, s=small, m=medium, l=large, x=xlarge

# Realizar detección en la imagen
results = model('bus.jpg', conf=0.5)

# Procesar resultados
for result in results:
    boxes = result.boxes  # Bounding boxes
    print(f"\\n✓ Objetos detectados: {len(boxes)}")
    
    for i, box in enumerate(boxes):
        # Obtener coordenadas y confianza
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        
        print(f"\\n  Objeto {i+1}:")
        print(f"    Clase: {model.names[int(cls)]}")
        print(f"    Confianza: {conf:.2%}")
        print(f"    Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Mostrar imagen con detecciones
from google.colab.patches import cv2_imshow
annotated = results[0].plot()
cv2_imshow(annotated)`,
        explanation:
          'YOLOv8 de Ultralytics es extremadamente fácil de usar y ofrece excelente balance entre velocidad y precisión. Detecta 80 clases diferentes (personas, coches, animales, etc.).',
      },
      {
        title: 'Entrenar YOLO Personalizado',
        code: `from ultralytics import YOLO

# Cargar modelo base para transfer learning
model = YOLO('yolov8n.pt')

# Opción 1: Entrenar con dataset de ejemplo COCO128
# Dataset pequeño con 128 imágenes para pruebas rápidas
results = model.train(
    data='coco128.yaml',  # Dataset de ejemplo incluido
    epochs=3,  # Solo 3 epochs para demo rápida
    imgsz=640,
    batch=16,
    name='demo_detector',
    device=0  # GPU 0, o 'cpu' para CPU
)

print("\\n✓ Entrenamiento completado!")

# Validar modelo entrenado
metrics = model.val()
print(f"\\nmAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Opción 2: Para tu dataset personalizado, crea un archivo YAML:
# Ejemplo de mi_dataset.yaml:
# path: /path/to/dataset
# train: images/train
# val: images/val
# names:
#   0: clase1
#   1: clase2
#
# Anota imágenes con herramientas como:
# - Roboflow: https://roboflow.com
# - LabelImg: https://github.com/heartexlabs/labelImg
# - CVAT: https://cvat.org

# Exportar modelo optimizado para producción
model.export(format='onnx')  # Más rápido para inferencia
print("\\n✓ Modelo exportado a ONNX")`,
        explanation:
          'Entrenar YOLO con transfer learning es muy efectivo. COCO128 es perfecto para practicar. Para datasets propios, usa Roboflow para anotar imágenes fácilmente.',
      },
      {
        title: 'Detección en Video/Webcam',
        code: `from ultralytics import YOLO
import cv2
import urllib.request

# Cargar modelo
model = YOLO('yolov8n.pt')

# Opción 1: Procesar video de ejemplo
url = 'https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4'
urllib.request.urlretrieve(url, 'demo_video.mp4')
print('✓ Video descargado')

video_path = 'demo_video.mp4'
cap = cv2.VideoCapture(video_path)

# Opción 2: Para usar webcam en Google Colab:
# from IPython.display import display, Javascript
# from google.colab.output import eval_js
# from base64 import b64decode
# # Requiere permisos de cámara en el navegador

frame_count = 0
max_frames = 50  # Procesar solo 50 frames para demo

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar detección (stream=True optimiza memoria)
    results = model(frame, conf=0.5, verbose=False)
    
    # Contar objetos detectados
    num_objects = len(results[0].boxes)
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Frame {frame_count}: {num_objects} objetos detectados")

cap.release()
print(f"\\n✓ Procesados {frame_count} frames")
print(f"FPS del video: {cap.get(cv2.CAP_PROP_FPS):.1f}")

# Para obtener video con anotaciones:
# results = model(video_path, save=True, conf=0.5)
# El video se guarda en runs/detect/predict/`,
        explanation:
          'YOLO procesa video frame por frame con alta velocidad. Para producción real, considera usar threading para captura y procesamiento paralelo.',
      },
    ],
    resources: [
      { title: 'Ultralytics YOLOv8 Documentation', url: 'https://docs.ultralytics.com/' },
      { title: 'YOLO GitHub Repository', url: 'https://github.com/ultralytics/ultralytics' },
      { title: 'Train Custom YOLO Model', url: 'https://docs.ultralytics.com/modes/train/' },
    ],
    proTips: [
      'Empieza con YOLOv8n (nano) para pruebas rápidas, luego sube a v8m/v8l',
      'conf=0.5 (50% confianza) es buen balance - ajusta según falsos positivos',
      'Fine-tuning con 100-500 imágenes propias supera modelos genéricos',
      'Usa augmentación (flip, rotate, scale) para mejorar con pocos datos'
    ],
  },
];
