'use client';

import React from 'react';

interface ProgressHeaderProps {
  completedCount: number;
  totalCount: number;
}

export const ProgressHeader: React.FC<ProgressHeaderProps> = ({
  completedCount,
  totalCount,
}) => {
  const percentage = Math.round((completedCount / totalCount) * 100);
  
  return (
    <div className="fixed top-0 left-0 right-0 z-40 bg-retro-black/95 backdrop-blur-sm border-b-2 border-retro-orange">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Title */}
          <div>
            <h1 className="font-pixel text-xs md:text-sm text-retro-orange">
              ML & CV ROADMAP
            </h1>
            <p className="text-xs text-gray-400 mt-1 hidden md:block">
              Aprende Machine Learning y Computer Vision
            </p>
          </div>
          
          {/* Progress */}
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="font-pixel text-xs text-white">
                {completedCount}/{totalCount}
              </div>
              <div className="text-xs text-gray-400">
                {percentage}% Completado
              </div>
            </div>
            
            {/* Progress Bar */}
            <div className="w-32 md:w-48 h-4 bg-retro-gray pixel-corners overflow-hidden">
              <div
                className="h-full bg-retro-orange transition-all duration-500"
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
