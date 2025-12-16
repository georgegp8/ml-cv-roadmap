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
    ],
  },
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

# Cargar dataset
df = pd.read_csv('data.csv')

# Exploración básica
print(df.head())
print(df.info())
print(df.describe())

# Agrupar y analizar
city_stats = df.groupby('ciudad').agg({
    'salario': ['mean', 'median', 'std'],
    'edad': 'mean'
})

print(city_stats)

# Filtrar datos
high_earners = df[df['salario'] > 60000]
print(f"Personas con salario > 60k: {len(high_earners)}")`,
        explanation:
          'EDA te ayuda a entender tus datos antes de entrenar modelos. Crucial para feature engineering.',
      },
    ],
    resources: [
      { title: 'Pandas Documentation', url: 'https://pandas.pydata.org/docs/' },
      { title: 'Pandas Cheat Sheet', url: 'https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf' },
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
from sklearn.model_selection import GridSearchCV

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
    ],
  },
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

# Cargar imagen
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar Gaussian Blur para reducir ruido
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detección de bordes con Canny
edges = cv2.Canny(blurred, 50, 150)

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Edges (Canny)')
plt.show()`,
        explanation:
          'La detección de bordes es fundamental en visión por computadora. Canny es el algoritmo más usado.',
      },
      {
        title: 'Detección de Rostros',
        code: `import cv2

# Cargar clasificador pre-entrenado
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Cargar imagen
img = cv2.imread('people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

# Dibujar rectángulos
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 107, 53), 2)

print(f"Rostros detectados: {len(faces)}")
cv2.imshow('Face Detection', img)
cv2.waitKey(0)`,
        explanation:
          'Haar Cascades son un método clásico para detección de objetos. Base para entender detección moderna.',
      },
    ],
    resources: [
      { title: 'OpenCV Documentation', url: 'https://docs.opencv.org/' },
      { title: 'OpenCV Python Tutorials', url: 'https://docs.opencv.org/master/d6/d00/tutorial_py_root.html' },
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
        code: `import torch
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
        code: `# Entrenamiento
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
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

# Evaluación
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')`,
        explanation:
          'El ciclo de entrenamiento es el corazón del deep learning. Aprende este patrón, lo usarás siempre.',
      },
    ],
    resources: [
      { title: 'PyTorch Documentation', url: 'https://pytorch.org/docs/stable/index.html' },
      { title: 'PyTorch Tutorials', url: 'https://pytorch.org/tutorials/' },
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
        code: `from ultralytics import YOLO
import cv2

# Cargar modelo pre-entrenado YOLOv8
model = YOLO('yolov8n.pt')  # n=nano, s=small, m=medium, l=large, x=xlarge

# Realizar detección en una imagen
results = model('imagen.jpg')

# Procesar resultados
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        # Obtener coordenadas y confianza
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
        
        print(f"Clase: {model.names[int(cls)]}")
        print(f"Confianza: {conf:.2f}")
        print(f"Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Mostrar imagen con detecciones
annotated = results[0].plot()
cv2.imshow('YOLOv8', annotated)
cv2.waitKey(0)`,
        explanation:
          'YOLOv8 de Ultralytics es extremadamente fácil de usar y ofrece excelente balance entre velocidad y precisión.',
      },
      {
        title: 'Entrenar YOLO Personalizado',
        code: `from ultralytics import YOLO

# Cargar modelo base
model = YOLO('yolov8n.pt')

# Entrenar con tu dataset
# Dataset debe estar en formato YOLO (txt con: clase x_center y_center width height)
results = model.train(
    data='mi_dataset.yaml',  # Configuración del dataset
    epochs=100,
    imgsz=640,
    batch=16,
    name='mi_detector',
    patience=50,  # Early stopping
    save=True,
    device=0  # GPU 0, o 'cpu' para CPU
)

# Validar modelo entrenado
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Exportar modelo optimizado
model.export(format='onnx')  # Para producción`,
        explanation:
          'Entrenar YOLO con tus propios datos es sencillo. El formato YOLO facilita la anotación de datasets.',
      },
      {
        title: 'Detección en Video en Tiempo Real',
        code: `from ultralytics import YOLO
import cv2

# Cargar modelo
model = YOLO('yolov8n.pt')

# Abrir video o webcam (0 para webcam)
cap = cv2.VideoCapture('video.mp4')  # o VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Realizar detección
    results = model(frame, stream=True)
    
    # Dibujar resultados
    for result in results:
        annotated = result.plot()
        cv2.imshow('YOLO Real-time', annotated)
    
    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("FPS promedio:", cap.get(cv2.CAP_PROP_FPS))`,
        explanation:
          'YOLO es ideal para detección en tiempo real. Procesa video frame por frame con alta velocidad.',
      },
    ],
    resources: [
      { title: 'Ultralytics YOLOv8 Documentation', url: 'https://docs.ultralytics.com/' },
      { title: 'YOLO GitHub Repository', url: 'https://github.com/ultralytics/ultralytics' },
      { title: 'Train Custom YOLO Model', url: 'https://docs.ultralytics.com/modes/train/' },
    ],
  },
];
