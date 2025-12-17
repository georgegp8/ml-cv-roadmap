'use client';

import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Download } from 'lucide-react';

interface ImageModalProps {
  imageUrl: string;
  isOpen: boolean;
  onClose: () => void;
  title?: string;
}

export const ImageModal: React.FC<ImageModalProps> = ({
  imageUrl,
  isOpen,
  onClose,
  title = 'Visualización',
}) => {
  // Cerrar con tecla ESC
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = 'grafica.png';
    link.click();
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/90 backdrop-blur-sm"
          />

          {/* Modal Content */}
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            className="relative w-full max-w-5xl bg-retro-black border-4 border-retro-orange rounded-lg overflow-hidden shadow-[0_0_50px_rgba(255,107,53,0.5)]"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b-4 border-retro-orange bg-retro-gray/30">
              <h3 className="text-retro-orange font-pixel text-sm md:text-base">
                📊 {title}
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={handleDownload}
                  className="p-2 hover:bg-retro-orange hover:text-black transition-colors rounded"
                  title="Descargar imagen"
                >
                  <Download size={20} />
                </button>
                <button
                  onClick={onClose}
                  className="p-2 hover:bg-retro-orange hover:text-black transition-colors rounded"
                  title="Cerrar"
                >
                  <X size={20} />
                </button>
              </div>
            </div>

            {/* Image */}
            <div className="p-6 bg-white max-h-[80vh] overflow-auto">
              <img
                src={imageUrl}
                alt="Gráfica generada"
                className="w-full h-auto"
              />
            </div>

            {/* Footer */}
            <div className="p-4 border-t-4 border-retro-gray bg-retro-gray/30 text-center">
              <p className="text-gray-400 text-xs">
                Presiona ESC o haz clic fuera para cerrar
              </p>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
