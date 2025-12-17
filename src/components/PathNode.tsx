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
  isLast?: boolean;
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
  isLast = false,
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
      hover: 'hover:border-8 hover:shadow-[0_0_20px_rgba(255,107,53,0.8)]',
      animate: 'animate-bounce',
    },
    completed: {
      bg: 'bg-retro-orange',
      border: 'border-retro-orange',
      opacity: 'opacity-100',
      cursor: 'cursor-pointer',
      hover: 'hover:border-8 hover:shadow-[0_0_20px_rgba(255,107,53,0.6)]',
      animate: '',
    },
  };
  
  const config = statusConfig[status];
  const canClick = status !== 'locked';
  
  // Tooltip messages
  const getTooltip = () => {
    if (status === 'completed') return 'Completado - Click para revisar';
    if (status === 'unlocked') return 'Click para comenzar';
    return 'Completa las etapas anteriores primero';
  };
  
  return (
    <div
      className="path-node absolute left-1/2 transform -translate-x-1/2"
      style={{
        '--node-x': `${x}%`,
        top: `${y * 180 - 20}px`,
      } as React.CSSProperties & { '--node-x': string }}
    >
      {/* Node Container */}
      <div className="flex flex-col items-center group">
        {/* Icon Square */}
        <div
          className={`
            w-24 h-24 rounded-xl border-4 flex items-center justify-center
            ${config.bg} ${config.border} ${config.opacity} ${config.cursor}
            transition-all duration-300 ${config.hover} ${config.animate}
            relative z-20
          `}
          onClick={canClick ? onClick : undefined}
          title={getTooltip()}
        >
          <StageIcon stageId={id} size={56} />
          
          {/* Tooltip on hover */}
          {status === 'locked' && (
            <div className="absolute -top-16 left-1/2 -translate-x-1/2 bg-retro-black border-2 border-gray-600 px-3 py-2 rounded text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-30">
              <div className="text-gray-400">{getTooltip()}</div>
              <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-retro-black border-b-2 border-r-2 border-gray-600 rotate-45"></div>
            </div>
          )}
          
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
        <div className={`${id === 'opencv' ? '-mt-2' : 'mt-3'} ${isLeft ? 'md:mr-32' : id === 'opencv' ? 'md:ml-48' : 'md:ml-32'}`}>
          <div className="px-3 py-1 border-2 border-retro-orange bg-retro-black rounded font-pixel text-xs text-white shadow-lg">
            {title}
          </div>
        </div>
        
        {/* Flecha animada para móvil (no mostrar en el último nodo) */}
        {!isLast && (
          <div className="md:hidden mt-6 flex flex-col items-center gap-1 animate-bounce">
            <svg 
              width="24" 
              height="24" 
              viewBox="0 0 24 24" 
              fill="none" 
              className="text-retro-orange"
            >
              <path 
                d="M12 5V19M12 19L5 12M12 19L19 12" 
                stroke="currentColor" 
                strokeWidth="3" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
            </svg>
            <div className="w-1 h-8 bg-gradient-to-b from-retro-orange/60 to-transparent"></div>
          </div>
        )}
      </div>
    </div>
  );
};
