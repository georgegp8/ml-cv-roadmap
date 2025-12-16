'use client';

import React from 'react';
import { Stage } from '../data/curriculum';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';
import { CodeBlock } from './CodeBlock';

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
  if (!isOpen || !stage) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
      <div className="bg-retro-black border-4 border-retro-orange w-full max-w-4xl max-h-[90vh] overflow-y-auto pixel-corners">
        {/* Header */}
        <div className="sticky top-0 bg-retro-orange p-6 border-b-4 border-black z-10">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-4xl">{stage.icon}</span>
                <div>
                  <h2 className="font-pixel text-xl text-black">{stage.title}</h2>
                  <p className="text-sm text-black/80 mt-1">{stage.subtitle}</p>
                </div>
              </div>
              <div className="flex gap-2 mt-3">
                <Badge variant={isCompleted ? 'success' : 'default'}>
                  {isCompleted ? '✓ Completado' : stage.duration}
                </Badge>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-black hover:text-retro-black text-2xl font-bold ml-4"
            >
              ✕
            </button>
          </div>
        </div>
        
        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Description */}
          <div>
            <p className="text-gray-300 leading-relaxed">{stage.description}</p>
          </div>
          
          {/* Objectives */}
          <div>
            <h3 className="font-pixel text-sm text-retro-orange mb-3">
              🎯 OBJETIVOS
            </h3>
            <ul className="space-y-2">
              {stage.objectives.map((obj, idx) => (
                <li key={idx} className="flex items-start gap-2 text-gray-300">
                  <span className="text-retro-orange mt-1">▸</span>
                  <span>{obj}</span>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Key Topics */}
          <div>
            <h3 className="font-pixel text-sm text-retro-orange mb-3">
              📚 TEMAS CLAVE
            </h3>
            <div className="flex flex-wrap gap-2">
              {stage.keyTopics.map((topic, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-retro-gray text-gray-300 text-sm rounded"
                >
                  {topic}
                </span>
              ))}
            </div>
          </div>
          
          {/* Practical Examples */}
          <div>
            <h3 className="font-pixel text-sm text-retro-orange mb-3">
              💻 EJEMPLOS PRÁCTICOS
            </h3>
            {stage.practicalExamples.map((example, idx) => (
              <div key={idx} className="mb-6">
                <h4 className="text-white font-semibold mb-2">{example.title}</h4>
                <CodeBlock 
                  code={example.code} 
                  language="python"
                  title={example.title}
                />
                <p className="text-gray-400 text-sm mt-2 italic">
                  💡 {example.explanation}
                </p>
              </div>
            ))}
          </div>
          
          {/* Resources */}
          <div>
            <h3 className="font-pixel text-sm text-retro-orange mb-3">
              🔗 RECURSOS
            </h3>
            <ul className="space-y-2">
              {stage.resources.map((resource, idx) => (
                <li key={idx}>
                  <a
                    href={resource.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-retro-orange hover:underline flex items-center gap-2"
                  >
                    <span>→</span>
                    {resource.title}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Actions */}
          <div className="flex gap-3 pt-4 border-t border-retro-gray">
            {!isCompleted && (
              <Button
                variant="primary"
                onClick={() => {
                  onComplete(stage.id);
                  onClose();
                }}
              >
                ✓ Marcar como Completado
              </Button>
            )}
            <Button variant="secondary" onClick={onClose}>
              Cerrar
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
