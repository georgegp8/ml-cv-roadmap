'use client';

import React from 'react';

interface LearningPathProps {
  totalStages: number;
}

export const LearningPath: React.FC<LearningPathProps> = ({ totalStages }) => {
  // Helper to calculate path coordinates
  const getPathCoordinates = () => {
    const points: { x: number; y: number }[] = [];
    const stageHeight = 180;
    const svgWidth = 800; // Approximate SVG width
    
    for (let i = 0; i < totalStages; i++) {
      const y = i * stageHeight + 90;
      let xPercent = 50; // Center by default
      
      // Desktop S-curve pattern
      const pos = i % 4;
      if (pos === 0) xPercent = 50;
      else if (pos === 1) xPercent = 75;
      else if (pos === 2) xPercent = 50;
      else if (pos === 3) xPercent = 25;
      
      // Convert percentage to actual pixel value
      const x = (xPercent / 100) * svgWidth;
      points.push({ x, y });
    }
    
    return points;
  };
  
  const pathPoints = getPathCoordinates();
  
  // Construir path con curvas suaves usando Cubic Bezier (C command)
  const pathData = pathPoints.map((point, index) => {
    if (index === 0) {
      return `M ${point.x} ${point.y}`;
    }
    
    const prevPoint = pathPoints[index - 1];
    const controlPointOffset = 60; // Distancia del control point
    
    // Control points para crear curvas suaves
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
