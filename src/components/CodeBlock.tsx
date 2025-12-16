'use client';

import React, { useState } from 'react';
import Prism from 'prismjs';
import 'prismjs/themes/prism-tomorrow.css';
import 'prismjs/components/prism-python';

interface CodeBlockProps {
  code: string;
  language?: string;
  title?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ 
  code, 
  language = 'python',
  title 
}) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  React.useEffect(() => {
    Prism.highlightAll();
  }, [code]);
  
  return (
    <div className="my-4 bg-[#1e1e1e] rounded-lg overflow-hidden border-2 border-retro-orange/30">
      {/* Header */}
      {title && (
        <div className="bg-retro-gray px-4 py-2 flex items-center justify-between border-b border-retro-orange/30">
          <span className="font-pixel text-xs text-retro-orange">{title}</span>
          <button
            onClick={handleCopy}
            className="text-xs text-gray-400 hover:text-retro-orange transition-colors"
          >
            {copied ? '✓ Copiado' : '📋 Copiar'}
          </button>
        </div>
      )}
      
      {/* Code */}
      <div className="overflow-x-auto">
        <pre className="!m-0 !bg-transparent">
          <code className={`language-${language} !text-sm`}>
            {code.trim()}
          </code>
        </pre>
      </div>
    </div>
  );
};
