'use client';

import React, { useEffect, useState } from 'react';
import { RetroBackground } from '../components/RetroBackground';
import { LearningPath } from '../components/LearningPath';
import { PathNode } from '../components/PathNode';
import { StageModal } from '../components/StageModal';
import { ProgressHeader } from '../components/ProgressHeader';
import { ConfettiEffect } from '../components/ConfettiEffect';
import { Toast } from '../components/Toast';
import { curriculum, Stage } from '../data/curriculum';

export default function Home() {
  // State
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [selectedStage, setSelectedStage] = useState<Stage | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [showToast, setShowToast] = useState(false);

  // Calculate status for each stage
  const getStageStatus = (index: number, id: string) => {
    if (completedStages.includes(id)) return 'completed';
    // First stage is unlocked by default, others need previous one completed
    if (index === 0 || completedStages.includes(curriculum[index - 1].id))
      return 'unlocked';
    return 'locked';
  };

  const handleNodeClick = (stage: Stage) => {
    setSelectedStage(stage);
    setIsModalOpen(true);
  };

  // Smooth scroll to stage
  const scrollToStage = (index: number) => {
    const element = document.getElementById(`stage-${index}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const handleComplete = (id: string) => {
    if (!completedStages.includes(id)) {
      setCompletedStages([...completedStages, id]);
      
      // Trigger confetti and toast
      setShowConfetti(true);
      const stage = curriculum.find(s => s.id === id);
      setToastMessage(`¡Misión completada: ${stage?.title || 'Stage'}! 🎉`);
      setShowToast(true);
      
      // Reset confetti after animation
      setTimeout(() => setShowConfetti(false), 3500);
    }
  };

  // Helper to determine X position based on index for S-curve
  const getXPosition = (index: number) => {
    // Desktop: 50 -> 75 -> 50 -> 25 -> 50 pattern
    // Mobile will always center (50%) via CSS
    const pos = index % 4;
    if (pos === 0) return 50;
    if (pos === 1) return 75;
    if (pos === 2) return 50;
    if (pos === 3) return 25;
    return 50;
  };

  return (
    <div className="min-h-screen font-mono text-white relative overflow-x-hidden">
      <RetroBackground />

      <ProgressHeader
        completedCount={completedStages.length}
        totalCount={curriculum.length}
      />

      <main className="relative pt-32 pb-32 min-h-screen max-w-4xl mx-auto">
        {/* Path SVG */}
        <LearningPath totalStages={curriculum.length} />

        {/* Nodes */}
        <div
          className="relative h-full"
          style={{
            height: `${curriculum.length * 180}px`,
          }}
        >
          {curriculum.map((stage, index) => {
            const status = getStageStatus(index, stage.id);
            const xPos = getXPosition(index);
            return (
              <div key={stage.id} id={`stage-${index}`}>
                <PathNode
                  id={stage.id}
                  icon={stage.icon}
                  title={stage.title}
                  status={status}
                  x={xPos}
                  y={index}
                  onClick={() => handleNodeClick(stage)}
                  isLeft={xPos < 50}
                  isLast={index === curriculum.length - 1}
                />
              </div>
            );
          })}
        </div>

        {/* Start/End Markers */}
        <div className="absolute top-12 left-1/2 -translate-x-1/2 text-center">
          <div className="font-pixel text-xs text-retro-orange mb-2 animate-bounce">
            START
          </div>
        </div>
      </main>


      <ConfettiEffect trigger={showConfetti} />
      
      <Toast
        message={toastMessage}
        isVisible={showToast}
        onClose={() => setShowToast(false)}
      />
      <StageModal
        stage={selectedStage}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onComplete={handleComplete}
        isCompleted={
          selectedStage ? completedStages.includes(selectedStage.id) : false
        }
      />
    </div>
  );
}
