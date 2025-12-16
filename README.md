# 🎮 ML & CV Roadmap - Roadmap Educativo Retro

Una interfaz web educativa moderna con estética retro gaming que presenta un roadmap completo para aprender **Machine Learning** con enfoque en **Visión por Computadora**.

## 🌟 Características

- **🎨 Estética Retro Gaming**: Paleta de colores naranja y negro, tipografía pixel art (Press Start 2P)
- **🛤️ Camino de Aprendizaje**: Path visual con nodos conectados estilo Duolingo
- **🔓 Sistema de Progresión**: Etapas bloqueadas/desbloqueadas/completadas
- **📚 Contenido Educativo**: Explicaciones claras, ejemplos en Python y recursos
- **💻 Code Snippets**: Bloques de código con syntax highlighting (PrismJS)
- **📱 Responsive Design**: Adaptado para móviles y escritorio
- **🎯 Enfoque Práctico**: "Aprender haciendo" con ejemplos reales

## 🗺️ Roadmap Completo

### 1. 🐍 Python Esencial (2-3 semanas)
Fundamentos del lenguaje: estructuras de datos, funciones, POO

### 2. 🔢 NumPy & Matplotlib (2-3 semanas)
Computación numérica con arrays y visualización de datos

### 3. 🐼 Pandas (2-3 semanas)
Manipulación y análisis de datos tabulares

### 4. 🤖 scikit-learn (4-5 semanas)
Machine Learning clásico: clasificación, regresión, clustering

### 5. 👁️ OpenCV (3-4 semanas)
Visión por Computadora clásica: filtros, detección de bordes, Haar Cascades

### 6. 🔥 PyTorch Esencial (4-5 semanas)
Deep Learning: tensores, autograd, redes neuronales

### 7. 🖼️ torchvision (4-6 semanas)
Deep Learning para CV: CNNs, transfer learning, detección de objetos

## 🚀 Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## 🛠️ Tecnologías

- **Framework**: Next.js 16 (App Router)
- **Lenguaje**: TypeScript
- **Estilos**: Tailwind CSS v4
- **Tipografía**: 
  - Press Start 2P (pixel art)
  - Space Mono (monospace)
- **Syntax Highlighting**: PrismJS
- **Iconos**: Emojis nativos

## 📁 Estructura del Proyecto

```
my-ml-cv-roadmap/
├── src/
│   ├── app/
│   │   ├── globals.css         # Estilos globales con tema retro
│   │   ├── layout.tsx          # Layout principal
│   │   └── page.tsx            # Página principal con roadmap
│   ├── components/
│   │   ├── ui/
│   │   │   ├── Button.tsx      # Botón con estilo pixel art
│   │   │   └── Badge.tsx       # Badge para estados
│   │   ├── CodeBlock.tsx       # Bloque de código con syntax
│   │   ├── LearningPath.tsx    # SVG path conectando nodos
│   │   ├── PathNode.tsx        # Nodo individual del roadmap
│   │   ├── ProgressHeader.tsx  # Header con progreso
│   │   ├── RetroBackground.tsx # Fondo con grid y scanlines
│   │   └── StageModal.tsx      # Modal con contenido detallado
│   └── data/
│       └── curriculum.ts       # Datos del roadmap completo
├── package.json
└── README.md
```

## 🎨 Paleta de Colores

```css
--retro-black: #0a0a0a;      /* Fondo principal */
--retro-orange: #ff6b35;     /* Color primario (CTAs, acentos) */
--retro-orange-dim: #cc552a; /* Orange oscuro (hover) */
--retro-gray: #2a2a2a;       /* Fondo secundario */
```

## 🎮 Uso

1. **Navega el Roadmap**: Haz scroll para ver todas las etapas
2. **Haz Click en un Nodo**: Abre el modal con contenido detallado
3. **Lee el Contenido**: Objetivos, temas clave, ejemplos prácticos
4. **Copia los Ejemplos**: Botón de copiar en cada código
5. **Marca como Completado**: Desbloquea la siguiente etapa
6. **Trackea tu Progreso**: Barra de progreso en el header

## 📚 Contenido por Etapa

Cada etapa incluye:

- ✅ **Objetivos**: Qué aprenderás
- 📖 **Temas Clave**: Conceptos principales
- 💻 **Ejemplos Prácticos**: Código Python comentado
- 💡 **Explicaciones**: Por qué es importante
- 🔗 **Recursos**: Links a documentación oficial

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
