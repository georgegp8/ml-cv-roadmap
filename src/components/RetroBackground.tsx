'use client';

import React from 'react';

interface RetroBackgroundProps {}

export const RetroBackground: React.FC<RetroBackgroundProps> = () => {
  return (
    <>
      {/* Animated grid background */}
      <div className="fixed inset-0 z-0 opacity-20">
        <div 
          className="absolute inset-0" 
          style={{
            backgroundImage: `
              linear-gradient(rgba(255, 107, 53, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255, 107, 53, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px',
          }}
        />
      </div>
      
      {/* Scanline effect */}
      <div className="scanline" />
      
      {/* Vignette */}
      <div 
        className="fixed inset-0 z-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, transparent 0%, rgba(0,0,0,0.4) 100%)',
        }}
      />
    </>
  );
};
