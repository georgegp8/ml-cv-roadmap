'use client';

import React from 'react';

interface LearningPathProps {
  totalStages: number;
}

export const LearningPath: React.FC<LearningPathProps> = ({ totalStages }) => {
  // Helper to calculate path coordinates
  const getPathCoordinates = () => {
    const points: { x: number; y: number; isStart?: boolean }[] = [];
    const stageHeight = 180;
    const svgWidth = 800;
    const nodeRadius = 40; // Radio del círculo del nodo
    
    for (let i = 0; i < totalStages; i++) {
      const centerY = i * stageHeight + 90;
      let xPercent = 50;
      
      // Desktop S-curve pattern
      const pos = i % 4;
      if (pos === 0) xPercent = 50;
      else if (pos === 1) xPercent = 75;
      else if (pos === 2) xPercent = 50;
      else if (pos === 3) xPercent = 25;
      
      const x = (xPercent / 100) * svgWidth;
      
      // Para el primer nodo, empezar desde el borde inferior
      if (i === 0) {
        points.push({ x, y: centerY + nodeRadius, isStart: true });
      } else {
        // Para los demás nodos, agregar dos puntos: entrada (arriba) y salida (abajo)
        points.push({ x, y: centerY - nodeRadius }); // Entrada por arriba
        if (i < totalStages - 1) {
          points.push({ x, y: centerY + nodeRadius }); // Salida por abajo
        }
      }
    }
    
    return points;
  };
  
  const pathPoints = getPathCoordinates();
  
  // Construir path con curvas suaves usando Cubic Bezier
  const pathData = pathPoints.map((point, index) => {
    if (index === 0) {
      return `M ${point.x} ${point.y}`;
    }
    
    const prevPoint = pathPoints[index - 1];
    const deltaY = point.y - prevPoint.y;
    
    // Control points para curvas suaves
    const controlPointOffset = deltaY * 0.5;
    
    const cp1x = prevPoint.x;
    const cp1y = prevPoint.y + controlPointOffset;
    const cp2x = point.x;
    const cp2y = point.y - controlPointOffset;
    
    return `C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${point.x} ${point.y}`;
  }).join(' ');
  
  return (
    <svg 
      className="absolute top-0 left-0 w-full pointer-events-none hidden md:block"
      style={{ height: `${totalStages * 180}px` }}
      viewBox="0 0 800 1260"
      preserveAspectRatio="xMidYMin meet"
    >
      <defs>
        <linearGradient id="pathGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#ff6b35" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#ff6b35" stopOpacity="0.6" />
        </linearGradient>
      </defs>
      
      {/* Draw dotted path connecting nodes */}
      <path
        d={pathData}
        stroke="url(#pathGradient)"
        strokeWidth="3"
        strokeDasharray="10,10"
        fill="none"
        strokeLinecap="round"
      />
      
      {/* Glow effect */}
      <path
        d={pathData}
        stroke="#ff6b35"
        strokeWidth="6"
        strokeDasharray="10,10"
        fill="none"
        strokeLinecap="round"
        opacity="0.2"
        filter="blur(4px)"
      />
    </svg>
  );
};
