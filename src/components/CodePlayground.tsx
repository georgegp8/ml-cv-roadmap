'use client';

import React, { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { Copy, Check, ExternalLink } from 'lucide-react';

interface CodePlaygroundProps {
  initialCode: string;
  title?: string;
  stageId?: string;
}

export const CodePlayground: React.FC<CodePlaygroundProps> = ({ 
  initialCode, 
  title = 'Python Playground',
  stageId = ''
}) => {
  const [code, setCode] = useState(initialCode);
  const [copied, setCopied] = useState(false);

  // Copiar código al portapapeles
  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Determinar el editor apropiado según el stageId
  const getEditorUrl = () => {
    if (stageId.includes('python') || stageId === 'python') {
      return 'https://www.online-python.com/';
    } else if (stageId.includes('matplotlib') || stageId.includes('numpy')) {
      return 'https://www.tutorialspoint.com/compilers/online-matplotlib-compiler.htm';
    } else if (stageId.includes('pandas')) {
      return 'https://python-fiddle.com/examples/pandas';
    } else if (stageId.includes('scikit')) {
      return 'https://python-fiddle.com/examples/sklearn';
    } else {
      // OpenCV, PyTorch, YOLO -> Google Colab
      return null; // Usar función de Colab
    }
  };

  // Abrir en Google Colab
  const openInColab = () => {
    const notebook = {
      cells: [
        {
          cell_type: 'code',
          source: code.split('\n'),
          metadata: {},
          outputs: [],
          execution_count: null
        }
      ],
      metadata: {
        kernelspec: {
          display_name: 'Python 3',
          language: 'python',
          name: 'python3'
        }
      },
      nbformat: 4,
      nbformat_minor: 0
    };
    
    const notebookJson = JSON.stringify(notebook);
    const encoded = encodeURIComponent(notebookJson);
    window.open(`https://colab.research.google.com/notebook#create=true&notebook=${encoded}`, '_blank');
  };

  // Abrir en editor externo
  const openInExternalEditor = () => {
    const editorUrl = getEditorUrl();
    if (editorUrl) {
      window.open(editorUrl, '_blank');
    } else {
      openInColab();
    }
  };

  const editorUrl = getEditorUrl();
  const editorName = editorUrl 
    ? (editorUrl.includes('online-python') ? 'Online-Python'
      : editorUrl.includes('matplotlib') ? 'Matplotlib Online' 
      : editorUrl.includes('pandas') ? 'Python Fiddle (Pandas)'
      : 'Python Fiddle (Sklearn)')
    : 'Google Colab';

  return (
    <div className="bg-retro-black border-2 border-retro-orange/30 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-retro-gray/30 px-3 md:px-4 py-3 border-b border-retro-orange/30">
        <div className="flex items-center justify-between gap-2 mb-3">
          <span className="text-retro-orange font-pixel text-[10px] md:text-xs flex-1 truncate">{title}</span>
        </div>
        
        {/* Botones de acción */}
        <div className="flex flex-wrap gap-2">
          {/* Copiar código */}
          <button
            onClick={copyToClipboard}
            className="flex-1 min-w-[120px] px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded text-xs font-semibold transition-all min-h-[44px] flex items-center justify-center gap-2"
          >
            {copied ? (
              <>
                <Check size={14} className="text-green-400" />
                <span>¡Copiado!</span>
              </>
            ) : (
              <>
                <Copy size={14} />
                <span>Copiar Código</span>
              </>
            )}
          </button>
          
          {/* Editor externo principal */}
          <button
            onClick={openInExternalEditor}
            className="flex-1 min-w-[140px] px-3 py-2 bg-retro-orange hover:bg-white text-black rounded text-xs font-semibold transition-all min-h-[44px] flex items-center justify-center gap-2"
          >
            <ExternalLink size={14} />
            <span>Abrir en {editorName}</span>
          </button>
        </div>
        
        <div className="mt-2 text-[10px] text-gray-500">
          💡 Copia el código o ábrelo directamente en el editor online
        </div>
      </div>

      {/* Editor */}
      <div className="border-b border-retro-orange/30">
        <CodeMirror
          value={code}
          height="200px"
          className="text-xs md:text-sm"
          theme="dark"
          extensions={[python()]}
          onChange={(value) => setCode(value)}
          basicSetup={{
            lineNumbers: true,
            highlightActiveLineGutter: true,
            highlightActiveLine: true,
            foldGutter: true,
          }}
        />
      </div>

      {/* Info adicional */}
      <div className="bg-[#1e1e1e] p-3 md:p-4">
        <div className="text-[10px] md:text-xs text-gray-500 mb-2 font-pixel">INSTRUCCIONES:</div>
        <div className="text-gray-400 text-xs space-y-1">
          <p>• Copia el código con el botón "Copiar"</p>
          <p>• Ábrelo en {editorName} para ejecutarlo</p>
          <p>• Modifica y experimenta con el código</p>
        </div>
      </div>
    </div>
  );
};
