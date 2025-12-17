'use client';

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Trophy, Award, Star, Zap } from 'lucide-react';

interface VictoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  completedCount: number;
  totalCount: number;
}

export const VictoryModal: React.FC<VictoryModalProps> = ({
  isOpen,
  onClose,
  completedCount,
  totalCount,
}) => {
  // Efecto de sonido (opcional)
  useEffect(() => {
    if (isOpen) {
      // Aquí podrías agregar un sonido de victoria
      console.log('🎉 ¡Todas las misiones completadas!');
    }
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center p-4">
          {/* Backdrop con animación */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/90 backdrop-blur-md"
          />

          {/* Modal Content */}
          <motion.div
            initial={{ scale: 0.5, opacity: 0, y: 100 }}
            animate={{ 
              scale: 1, 
              opacity: 1, 
              y: 0,
              transition: {
                type: "spring",
                duration: 0.7,
                bounce: 0.4
              }
            }}
            exit={{ scale: 0.5, opacity: 0, y: 100 }}
            className="relative w-full max-w-2xl bg-gradient-to-br from-retro-black via-retro-gray to-retro-black border-4 border-retro-orange rounded-xl overflow-hidden shadow-[0_0_100px_rgba(255,107,53,0.8)]"
          >
            {/* Efectos de fondo */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              {[...Array(20)].map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-2 h-2 bg-retro-orange rounded-full"
                  initial={{ 
                    x: Math.random() * 100 + '%',
                    y: '100%',
                    opacity: 0 
                  }}
                  animate={{
                    y: [null, '-100%'],
                    opacity: [0, 1, 0],
                    transition: {
                      duration: 2 + Math.random() * 2,
                      repeat: Infinity,
                      delay: Math.random() * 2
                    }
                  }}
                />
              ))}
            </div>

            {/* Content */}
            <div className="relative p-8 md:p-12 text-center">
              {/* Trofeo animado */}
              <motion.div
                initial={{ scale: 0, rotate: -180 }}
                animate={{ 
                  scale: 1, 
                  rotate: 0,
                  transition: {
                    type: "spring",
                    delay: 0.3,
                    duration: 0.8
                  }
                }}
                className="mb-6 flex justify-center"
              >
                <div className="relative">
                  <Trophy className="w-24 h-24 md:w-32 md:h-32 text-retro-orange drop-shadow-[0_0_20px_rgba(255,107,53,0.8)]" />
                  
                  {/* Estrellas giratorias */}
                  {[0, 120, 240].map((rotation, i) => (
                    <motion.div
                      key={i}
                      className="absolute top-1/2 left-1/2"
                      animate={{
                        rotate: [rotation, rotation + 360],
                        scale: [1, 1.2, 1],
                      }}
                      transition={{
                        duration: 3,
                        repeat: Infinity,
                        ease: "linear"
                      }}
                    >
                      <Star
                        className="w-6 h-6 text-yellow-400 -translate-x-1/2 -translate-y-1/2"
                        style={{ 
                          transform: `translate(-50%, -50%) translateY(-60px) rotate(${-rotation}deg)` 
                        }}
                        fill="currentColor"
                      />
                    </motion.div>
                  ))}
                </div>
              </motion.div>

              {/* Título */}
              <motion.h2
                initial={{ opacity: 0, y: 20 }}
                animate={{ 
                  opacity: 1, 
                  y: 0,
                  transition: { delay: 0.5 }
                }}
                className="text-lg md:text-2xl lg:text-4xl font-pixel text-retro-orange mb-4 leading-tight px-4"
              >
                ¡FELICITACIONES!
              </motion.h2>

              {/* Mensaje */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ 
                  opacity: 1, 
                  y: 0,
                  transition: { delay: 0.7 }
                }}
                className="space-y-4 mb-8"
              >
                <p className="text-xl md:text-2xl text-white font-bold">
                  🎉 ¡Has completado todas las misiones! 🎉
                </p>
                <p className="text-gray-300 text-lg">
                  Has dominado <span className="text-retro-orange font-bold">{totalCount} etapas</span> del roadmap de ML/CV
                </p>
                <div className="flex items-center justify-center gap-4 mt-6">
                  <Award className="w-8 h-8 text-yellow-400" />
                  <Zap className="w-8 h-8 text-retro-orange" />
                  <Award className="w-8 h-8 text-yellow-400" />
                </div>
              </motion.div>

              {/* Stats */}
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ 
                  opacity: 1, 
                  scale: 1,
                  transition: { delay: 0.9 }
                }}
                className="bg-retro-gray/50 border-2 border-retro-orange rounded-lg p-6 mb-6"
              >
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-3xl font-pixel text-retro-orange">{totalCount}</div>
                    <div className="text-xs text-gray-400 mt-1">Etapas</div>
                  </div>
                  <div>
                    <div className="text-3xl font-pixel text-green-400">100%</div>
                    <div className="text-xs text-gray-400 mt-1">Completo</div>
                  </div>
                  <div>
                    <div className="text-3xl font-pixel text-yellow-400">⭐</div>
                    <div className="text-xs text-gray-400 mt-1">Maestro ML</div>
                  </div>
                </div>
              </motion.div>

              {/* Mensaje motivacional */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ 
                  opacity: 1,
                  transition: { delay: 1.1 }
                }}
                className="text-gray-400 text-sm italic mb-6"
              >
                "El aprendizaje es un viaje continuo. ¡Sigue practicando y construyendo proyectos increíbles!"
              </motion.div>

              {/* Botón de cierre */}
              <motion.button
                initial={{ opacity: 0, y: 20 }}
                animate={{ 
                  opacity: 1, 
                  y: 0,
                  transition: { delay: 1.3 }
                }}
                onClick={onClose}
                className="px-8 py-4 bg-retro-orange text-black font-pixel text-sm hover:bg-white transition-all hover:scale-105 rounded-lg shadow-lg"
              >
                ¡CONTINUAR APRENDIENDO!
              </motion.button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
