'use client';

import React from 'react';
import Image from 'next/image';

interface StageIconProps {
  stageId: string;
  size?: number;
  className?: string;
}

const iconMap: Record<string, { src: string; alt: string }> = {
  'python-basics': { 
    src: 'https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg',
    alt: 'Python'
  },
  'numpy-matplotlib': { 
    src: 'https://icon.icepanel.io/Technology/svg/NumPy.svg',
    alt: 'NumPy'
  },
  'pandas': { 
    src: 'https://upload.wikimedia.org/wikipedia/commons/2/22/Pandas_mark.svg',
    alt: 'Pandas'
  },
  'scikit-learn': { 
    src: 'https://icon.icepanel.io/Technology/svg/scikit-learn.svg',
    alt: 'scikit-learn'
  },
  'opencv': { 
    src: 'https://icon.icepanel.io/Technology/svg/OpenCV.svg',
    alt: 'OpenCV'
  },
  'pytorch': { 
    src: 'https://icon.icepanel.io/Technology/svg/PyTorch.svg',
    alt: 'PyTorch'
  },
  'yolo': { 
    src: 'https://storage.googleapis.com/organization-image-assets/ultralytics-botAvatarSrcUrl-1729379860806.svg',
    alt: 'YOLO'
  },
};

export const StageIcon: React.FC<StageIconProps> = ({ stageId, size = 40, className = '' }) => {
  const iconInfo = iconMap[stageId];
  
  if (!iconInfo) {
    return <span className={className} style={{ fontSize: size }}>📚</span>;
  }

  // Python es el primer stage, marcar como LCP con loading eager
  const isPython = stageId === 'python-basics';

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size }}>
      <Image
        src={iconInfo.src}
        alt={iconInfo.alt}
        fill
        className="object-contain"
        unoptimized
        loading={isPython ? 'eager' : 'lazy'}
        priority={isPython}
      />
    </div>
  );
};
