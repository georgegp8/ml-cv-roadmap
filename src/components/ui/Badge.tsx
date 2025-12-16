import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'locked';
}

export const Badge: React.FC<BadgeProps> = ({ 
  children, 
  variant = 'default' 
}) => {
  const variants = {
    default: 'bg-retro-gray text-white',
    success: 'bg-green-600 text-white',
    warning: 'bg-yellow-600 text-black',
    locked: 'bg-gray-700 text-gray-400',
  };
  
  return (
    <span className={`px-2 py-1 text-xs font-pixel ${variants[variant]} pixel-corners inline-block`}>
      {children}
    </span>
  );
};
