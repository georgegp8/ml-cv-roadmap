'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Play, Code, BookOpen, CheckCircle } from 'lucide-react';
import { Stage } from '../data/curriculum';
import { CodePlayground } from './CodePlayground';
import { StageIcon } from './StageIcon';

interface StageModalProps {
  stage: Stage | null;
  isOpen: boolean;
  onClose: () => void;
  onComplete: (id: string) => void;
  isCompleted: boolean;
}

export const StageModal: React.FC<StageModalProps> = ({
  stage,
  isOpen,
  onClose,
  onComplete,
  isCompleted,
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'code' | 'demo'>('overview');
  
  // Reset tab to overview when stage changes
  useEffect(() => {
    if (stage) {
      setActiveTab('overview');
    }
  }, [stage?.id]);
  
  if (!stage) return null;
  
  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8">
          {/* Backdrop */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
          />
          
          {/* Modal Content */}
          <motion.div 
            initial={{ scale: 0.9, opacity: 0, y: 20 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 20 }}
            className="relative w-full max-w-4xl h-[90vh] md:h-[80vh] bg-retro-black border-2 md:border-4 border-retro-orange flex flex-col overflow-hidden shadow-[0_0_50px_rgba(255,107,53,0.2)]"
            style={{ borderRadius: '8px' }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 md:p-6 border-b-2 md:border-b-4 border-retro-gray bg-retro-gray/20">
              <div className="flex items-center gap-2 md:gap-4">
                <StageIcon stageId={stage.id} size={48} className="md:w-16 md:h-16" />
                <div>
                  <h2 className="text-base md:text-2xl font-pixel text-retro-orange">{stage.title}</h2>
                  <p className="text-gray-400 font-mono text-xs md:text-sm hidden md:block">{stage.subtitle}</p>
                </div>
              </div>
              <button 
                onClick={onClose}
                className="p-3 md:p-2 hover:bg-retro-orange hover:text-black transition-colors rounded min-w-[44px] min-h-[44px] flex items-center justify-center"
              >
                <X size={20} className="md:w-6 md:h-6" />
              </button>
            </div>
            
            {/* Tabs */}
            <div className="flex border-b-2 md:border-b-4 border-retro-gray bg-retro-black">
              {[
                { id: 'overview', label: 'Resumen', icon: BookOpen },
                { id: 'code', label: 'Código', icon: Code },
                { id: 'demo', label: 'Playground', icon: Play },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`
                    flex-1 py-4 md:py-4 min-h-[52px] flex items-center justify-center gap-1 md:gap-2 font-pixel text-[10px] md:text-sm transition-colors
                    ${activeTab === tab.id 
                      ? 'bg-retro-orange text-black' 
                      : 'bg-retro-black text-gray-500 hover:text-white hover:bg-retro-gray'
                    }
                  `}
                >
                  <tab.icon size={16} className="md:w-4 md:h-4" />
                  <span className="hidden xs:inline">{tab.label}</span>
                </button>
              ))}
            </div>
            
            {/* Content Area */}
            <div className="flex-1 overflow-y-auto p-4 md:p-8 bg-retro-black custom-scrollbar">
              {activeTab === 'overview' && (
                <div className="space-y-8 animate-in fade-in duration-300">
                  <div>
                    <h3 className="text-retro-orange font-bold mb-4 text-lg border-b border-retro-gray pb-2">Descripción de la Misión</h3>
                    <p className="text-gray-300 leading-relaxed text-lg">{stage.description}</p>
                  </div>
                  
                  <div>
                    <h3 className="text-retro-orange font-bold mb-4 text-lg border-b border-retro-gray pb-2">Objetivos</h3>
                    <ul className="space-y-3">
                      {stage.objectives.map((obj, idx) => (
                        <li key={idx} className="flex items-start gap-3 bg-retro-gray/30 p-3 rounded border border-retro-gray">
                          <span className="text-retro-orange mt-1">▸</span>
                          <span className="text-gray-300">{obj}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="text-retro-orange font-bold mb-4 text-lg border-b border-retro-gray pb-2">Conceptos Clave</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {stage.keyTopics.map((topic, idx) => (
                        <div key={idx} className="flex items-center gap-3 bg-retro-gray/30 p-3 rounded border border-retro-gray">
                          <div className="w-2 h-2 bg-retro-orange rounded-full flex-shrink-0" />
                          <span className="text-gray-300 text-sm">{topic}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-retro-orange font-bold mb-4 text-lg border-b border-retro-gray pb-2">Recursos</h3>
                    <div className="space-y-2">
                      {stage.resources.map((resource, idx) => (
                        <a
                          key={idx}
                          href={resource.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center gap-2 text-retro-orange hover:text-white hover:underline transition-colors p-2 hover:bg-retro-gray/20 rounded"
                        >
                          <span>→</span>
                          <span>{resource.title}</span>
                        </a>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm text-gray-500 mt-8 p-4 bg-retro-gray/20 rounded border border-retro-gray">
                    <span>⏱️ Tiempo Estimado:</span>
                    <span className="text-white font-semibold">{stage.duration}</span>
                  </div>
                </div>
              )}
              
              {activeTab === 'code' && (
                <div className="space-y-8 animate-in fade-in duration-300">
                  {stage.practicalExamples.map((example, idx) => (
                    <div key={idx} className="space-y-2">
                      <h4 className="text-retro-orange font-bold text-lg mb-3">{example.title}</h4>
                      <div className="bg-[#1e1e1e] p-4 rounded border-2 border-retro-orange/30 font-mono text-sm overflow-x-auto">
                        <pre className="text-gray-300">
                          <code>{example.code}</code>
                        </pre>
                      </div>
                      <p className="text-gray-400 text-sm italic bg-retro-gray/20 p-3 rounded border-l-4 border-retro-orange">
                        💡 {example.explanation}
                      </p>
                    </div>
                  ))}
                </div>
              )}
              
              {activeTab === 'demo' && (
                <div className="space-y-6 animate-in fade-in duration-300">
                  
                  {/* Interactive Code Playground */}
                  {stage.practicalExamples.length > 0 && (
                    <div className="space-y-4">
                      <div className="bg-retro-orange/10 border-l-4 border-retro-orange p-3 rounded">
                        <p className="text-sm text-gray-300">
                          ⚡ <strong>¡Pruébalo en vivo!</strong> Edita y ejecuta código Python directamente en tu navegador.
                        </p>
                      </div>
                      
                      <CodePlayground
                        initialCode={stage.practicalExamples[0].code}
                        title={`${stage.title} - ${stage.practicalExamples[0].title}`}
                      />
                      
                      <div className="bg-retro-gray/20 p-4 rounded border border-retro-gray">
                        <p className="text-gray-400 text-sm italic">
                          💡 {stage.practicalExamples[0].explanation}
                        </p>
                      </div>
                    </div>
                  )}
                  
                  {/* Additional Examples */}
                  {stage.practicalExamples.length > 1 && (
                    <div className="mt-6">
                      <h4 className="text-retro-orange font-bold mb-3">Más Ejemplos para Probar:</h4>
                      <div className="grid grid-cols-1 gap-3">
                        {stage.practicalExamples.slice(1).map((example, idx) => (
                          <details key={idx} className="bg-retro-gray/20 p-4 rounded border border-retro-gray cursor-pointer">
                            <summary className="font-semibold text-white hover:text-retro-orange">
                              {example.title}
                            </summary>
                            <div className="mt-3">
                              <CodePlayground
                                initialCode={example.code}
                                title={example.title}
                              />
                            </div>
                          </details>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Pro Tips */}
                  <div className="bg-retro-orange/10 border-l-4 border-retro-orange p-4 rounded mt-6">
                    <h4 className="text-retro-orange font-bold mb-2 flex items-center gap-2">
                      <span>🚀</span> Consejos Pro
                    </h4>
                    <ul className="space-y-2 text-sm text-gray-300">
                      <li>• Modifica el código y ve los resultados instantáneamente</li>
                      <li>• Usa print() para depurar y entender el flujo</li>
                      <li>• Prueba diferentes valores para experimentar</li>
                      <li>• Haz clic en "Resetear" para restaurar el código original</li>
                    </ul>
                  </div>
                </div>
              )}
            </div>
            
            {/* Footer Actions */}
            <div className="p-4 md:p-6 border-t-2 md:border-t-4 border-retro-gray bg-retro-black flex flex-col md:flex-row justify-between items-stretch md:items-center gap-3">
              <button
                onClick={onClose}
                className="px-4 py-3 md:py-2 text-gray-400 hover:text-white transition-colors text-center min-h-[44px]"
              >
                ← Volver al Roadmap
              </button>
              <button
                onClick={() => {
                  onComplete(stage.id);
                  onClose();
                }}
                disabled={isCompleted}
                className={`
                  px-6 py-4 md:py-3 font-pixel text-xs md:text-sm flex items-center justify-center gap-2 transition-all min-h-[52px]
                  ${isCompleted 
                    ? 'bg-green-600 text-white cursor-default' 
                    : 'bg-retro-orange text-black hover:bg-white hover:scale-105'
                  }
                `}
                style={{ borderRadius: '4px' }}
              >
                {isCompleted ? (
                  <>
                    <CheckCircle size={18} />
                    <span>Misión Completa</span>
                  </>
                ) : (
                  <>
                    <span>Completar Misión</span>
                    <span className="animate-pulse">_</span>
                  </>
                )}
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
