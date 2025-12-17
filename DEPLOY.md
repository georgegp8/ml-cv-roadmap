# 🚀 Guía de Despliegue en Vercel

## Pasos para Desplegar

### 1. Preparar el Proyecto
```bash
# Verificar que todo está en el repositorio
git status
git push
```

### 2. Desplegar en Vercel

#### Opción A: Desde la Web (Recomendado)
1. Ve a [vercel.com](https://vercel.com)
2. Click en "Add New Project"
3. Importa tu repositorio: `georgegp8/ml-cv-roadmap`
4. Configura el proyecto:
   - **Framework Preset**: Next.js
   - **Root Directory**: `./` (por defecto)
   - **Build Command**: `npm run build` (detectado automáticamente)
   - **Output Directory**: `.next` (detectado automáticamente)
5. Click en "Deploy"

#### Opción B: Desde CLI
```bash
# Instalar Vercel CLI
npm i -g vercel

# Hacer login
vercel login

# Desplegar
vercel --prod
```

### 3. Configuración (Opcional)

#### Variables de Entorno
No se requieren variables de entorno para este proyecto.

#### Dominios Personalizados
1. En Vercel Dashboard → tu proyecto
2. Settings → Domains
3. Añade tu dominio personalizado

## ✅ Características Listas para Producción

- ✅ **Next.js 16** configurado correctamente
- ✅ **Imágenes optimizadas** con Next.js Image
- ✅ **Responsive design** optimizado para móvil
- ✅ **Loading states** para mejor UX
- ✅ **Efectos visuales** (confetti, toasts)
- ✅ **Python en el navegador** con Pyodide (CDN)

## 📱 Optimizaciones Móviles Incluidas

- Modal de altura completa en móvil (90vh)
- Botones táctiles de mínimo 44x44px
- Tabs optimizados para tocar
- Línea vertical simple en lugar de paths SVG curvos
- Editor de código con altura reducida
- Texto responsivo para pantallas pequeñas

## 🎨 Experiencia de Usuario

- **Confeti** al completar cada etapa
- **Toast notifications** con mensajes de éxito
- **Tooltips** en stages bloqueados
- **Scroll suave** entre etapas
- **Loading states** para Pyodide
- **Animaciones** en iconos desbloqueados

## 🔍 Verificación Post-Deploy

Después del despliegue, verifica:

1. ✅ Todos los iconos cargan correctamente
2. ✅ Modal abre y cierra sin problemas
3. ✅ Tabs funcionan (Resumen, Código, Playground)
4. ✅ Pyodide se carga y ejecuta código
5. ✅ Confeti aparece al completar stages
6. ✅ Responsive funciona en móvil
7. ✅ Paths/líneas se ven correctamente

## 📊 Performance

El proyecto está optimizado para:
- **First Contentful Paint**: < 1.8s
- **Largest Contentful Paint**: < 2.5s
- **Time to Interactive**: < 3.5s

Nota: Pyodide (~15-20MB) se carga bajo demanda solo cuando el usuario hace click en "Ejecutar Código".

## 🐛 Troubleshooting

### Problema: Imágenes no cargan
**Solución**: Verifica que `next.config.ts` tiene los dominios correctos en `remotePatterns`.

### Problema: Error de build
**Solución**: Ejecuta `npm run build` localmente primero para detectar errores.

### Problema: Pyodide no funciona
**Solución**: Pyodide se carga desde CDN, asegúrate que el navegador permite scripts externos.

## 🔗 Enlaces Útiles

- [Vercel Dashboard](https://vercel.com/dashboard)
- [Next.js Documentation](https://nextjs.org/docs)
- [Pyodide Documentation](https://pyodide.org/)
- [Repositorio GitHub](https://github.com/georgegp8/ml-cv-roadmap)
