'use client';

import React from 'react';
import { Badge } from './ui/Badge';
import { StageIcon } from './StageIcon';

interface PathNodeProps {
  id: string;
  icon: string;
  title: string;
  status: 'locked' | 'unlocked' | 'completed';
  x: number; // Percentage
  y: number; // Index
  onClick: () => void;
  isLeft?: boolean;
}

export const PathNode: React.FC<PathNodeProps> = ({
  id,
  icon,
  title,
  status,
  x,
  y,
  onClick,
  isLeft = false,
}) => {
  const statusConfig = {
    locked: {
      bg: 'bg-gray-800',
      border: 'border-gray-600',
      opacity: 'opacity-50',
      cursor: 'cursor-not-allowed',
      hover: '',
      animate: '',
    },
    unlocked: {
      bg: 'bg-retro-gray',
      border: 'border-retro-orange',
      opacity: 'opacity-100',
      cursor: 'cursor-pointer',
      hover: 'hover:shadow-lg hover:shadow-retro-orange/50',
      animate: 'animate-bounce',
    },
    completed: {
      bg: 'bg-retro-orange',
      border: 'border-retro-orange',
      opacity: 'opacity-100',
      cursor: 'cursor-pointer',
      hover: 'hover:shadow-lg hover:shadow-retro-orange/30',
      animate: '',
    },
  };
  
  const config = statusConfig[status];
  const canClick = status !== 'locked';
  
  return (
    <div
      className="absolute transform -translate-x-1/2"
      style={{
        left: `${x}%`,
        top: `${y * 180 - 20}px`,
      }}
    >
      {/* Node Container */}
      <div className="flex flex-col items-center">
        {/* Icon Square */}
        <div
          className={`
            w-24 h-24 rounded-xl border-4 flex items-center justify-center
            ${config.bg} ${config.border} ${config.opacity} ${config.cursor}
            transition-all duration-300 ${config.hover} ${config.animate}
            relative z-20
          `}
          onClick={canClick ? onClick : undefined}
        >
          <StageIcon stageId={id} size={56} />
          
          {/* Completion checkmark */}
          {status === 'completed' && (
            <div className="absolute -top-2 -right-2 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center border-2 border-black">
              <span className="text-white text-sm">✓</span>
            </div>
          )}
          
          {/* Lock icon */}
          {status === 'locked' && (
            <div className="absolute -bottom-2 w-6 h-6 bg-gray-700 rounded flex items-center justify-center">
              <span className="text-gray-400 text-xs">🔒</span>
            </div>
          )}
        </div>
        
        {/* Title */}
        <div className={`mt-3 ${isLeft ? 'md:mr-32' : 'md:ml-32'}`}>
          <Badge variant={status === 'completed' ? 'success' : status === 'locked' ? 'locked' : 'default'}>
            {title}
          </Badge>
        </div>
      </div>
    </div>
  );
};
