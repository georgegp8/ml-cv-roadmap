'use client';

import React, { useEffect, useState } from 'react';
import { RetroBackground } from '../components/RetroBackground';
import { LearningPath } from '../components/LearningPath';
import { PathNode } from '../components/PathNode';
import { StageModal } from '../components/StageModal';
import { ProgressHeader } from '../components/ProgressHeader';
import { curriculum, Stage } from '../data/curriculum';

export default function Home() {
  // State
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [selectedStage, setSelectedStage] = useState<Stage | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [windowWidth, setWindowWidth] = useState(
    typeof window !== 'undefined' ? window.innerWidth : 1024
  );

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

  const handleComplete = (id: string) => {
    if (!completedStages.includes(id)) {
      setCompletedStages([...completedStages, id]);
    }
  };

  // Helper to determine X position based on index for S-curve
  const getXPosition = (index: number) => {
    // Mobile: always center (50%)
    if (windowWidth < 768) return 50;
    // Desktop: 50 -> 75 -> 50 -> 25 -> 50 pattern
    const pos = index % 4;
    if (pos === 0) return 50;
    if (pos === 1) return 75;
    if (pos === 2) return 50;
    if (pos === 3) return 25;
    return 50;
  };

  // Handle resize for responsive positioning
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

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
              <PathNode
                key={stage.id}
                id={stage.id}
                icon={stage.icon}
                title={stage.title}
                status={status}
                x={xPos}
                y={index}
                onClick={() => handleNodeClick(stage)}
                isLeft={xPos < 50}
              />
            );
          })}
        </div>

        {/* Start/End Markers */}
        <div className="absolute top-24 left-1/2 -translate-x-1/2 text-center">
          <div className="font-pixel text-xs text-retro-orange mb-2 animate-bounce">
            START
          </div>
        </div>
      </main>

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
