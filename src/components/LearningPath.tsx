'use client';

import React, { useMemo } from 'react';

interface LearningPathProps {
  totalStages: number;
}

export const LearningPath: React.FC<LearningPathProps> = ({ totalStages }) => {
  const stageHeight = 180;
  const svgWidth = 800;

  // PathNode: w-24 h-24 => 96px -> radio 48
  const nodeRadius = 48;
  const holeRadius = nodeRadius + 10;

  const { pathData, vbHeight, centers } = useMemo(() => {
    const getXPercent = (i: number) => {
      const pos = i % 4;
      if (pos === 0) return 50;
      if (pos === 1) return 75;
      if (pos === 2) return 50;
      return 25;
    };

    // Centro EXACTO del círculo según tu PathNode
    const pts = Array.from({ length: totalStages }, (_, i) => ({
      x: (getXPercent(i) / 100) * svgWidth,
      y: i * stageHeight + nodeRadius + 110, // y*180 + 40 + 10
    }));

    let d = `M ${pts[0].x} ${pts[0].y}`;
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1];
      const cur = pts[i];
      const cpOffset = stageHeight * 0.45;

      d += ` C ${prev.x} ${prev.y + cpOffset}, ${cur.x} ${cur.y - cpOffset}, ${cur.x} ${cur.y}`;
    }

    // Extra para que el último nodo (YOLO) no quede fuera
    const height = (totalStages - 1) * stageHeight + nodeRadius * 2 + 140;

    return { pathData: d, vbHeight: height, centers: pts };
  }, [totalStages]);

  return (
    <>
      {/* Línea vertical simple para móvil */}
      <div className="md:hidden absolute left-1/2 top-0 w-1 -translate-x-1/2 pointer-events-none z-10"
        style={{ 
          height: `${vbHeight}px`,
          background: 'linear-gradient(180deg, rgba(255,107,53,0.3) 0%, rgba(255,107,53,0.6) 100%)',
          backgroundSize: '100% 20px',
          backgroundRepeat: 'repeat-y'
        }}
      />
      
      {/* Paths curvos para desktop */}
      <svg
        className="absolute top-0 left-0 w-full pointer-events-none hidden md:block z-10"
        style={{ height: `${vbHeight}px` }}
        viewBox={`0 0 ${svgWidth} ${vbHeight}`}
        preserveAspectRatio="none"
      >
      <defs>
        <linearGradient id="pathGradient" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#ff6b35" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#ff6b35" stopOpacity="0.7" />
        </linearGradient>

        <filter id="glow">
          <feGaussianBlur stdDeviation="3.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        {/* Recorta la línea debajo de cada icono */}
        <mask id="pathMask">
          <rect x="0" y="0" width={svgWidth} height={vbHeight} fill="white" />
          {centers.map((p, idx) => (
            <circle key={idx} cx={p.x} cy={p.y} r={holeRadius} fill="black" />
          ))}
        </mask>
      </defs>

      {/* Glow detrás */}
      <path
        d={pathData}
        stroke="#ff6b35"
        strokeWidth="8"
        strokeDasharray="10 10"
        strokeLinecap="round"
        fill="none"
        opacity="0.16"
        filter="url(#glow)"
        mask="url(#pathMask)"
      />

      {/* Línea principal */}
      <path
        d={pathData}
        stroke="url(#pathGradient)"
        strokeWidth="3"
        strokeDasharray="10 10"
        strokeLinecap="round"
        fill="none"
        mask="url(#pathMask)"
      />
      </svg>
    </>
  );
};
