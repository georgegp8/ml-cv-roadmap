'use client';

import React from 'react';

interface RetroBackgroundProps {}

export const RetroBackground: React.FC<RetroBackgroundProps> = () => {
  return (
    <>
      {/* Animated grid background */}
      <div className="fixed inset-0 z-0 opacity-20">
        <div className="absolute inset-0 retro-grid" />
      </div>
      
      {/* Scanline effect */}
      <div className="scanline" />
      
      {/* Vignette */}
      <div className="fixed inset-0 z-0 pointer-events-none retro-vignette" />
    </>
  );
};
