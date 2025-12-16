'use client';

import React, { useState, useEffect, useRef } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { Play, RotateCcw, Loader2 } from 'lucide-react';

interface CodePlaygroundProps {
  initialCode: string;
  title?: string;
}

export const CodePlayground: React.FC<CodePlaygroundProps> = ({ 
  initialCode, 
  title = 'Python Playground' 
}) => {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const pyodideRef = useRef<any>(null);

  // Cargar Pyodide cuando el componente se monta
  useEffect(() => {
    const loadPyodide = async () => {
      try {
        setIsLoading(true);
        setOutput('Loading Python environment...\n');
        
        // Cargar el script de Pyodide dinámicamente
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        script.async = true;
        
        script.onload = async () => {
          try {
            // @ts-ignore
            if (typeof window.loadPyodide === 'function') {
              // @ts-ignore
              const pyodide = await window.loadPyodide({
                indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
              });
              pyodideRef.current = pyodide;
              setPyodideReady(true);
              setOutput('✓ Python ready! Click "Run Code" to execute.\n');
            }
          } catch (err) {
            console.error('Error initializing Pyodide:', err);
            setOutput('Error initializing Python. Please refresh the page.');
          } finally {
            setIsLoading(false);
          }
        };
        
        script.onerror = () => {
          setOutput('Error loading Python environment. Please check your internet connection.');
          setIsLoading(false);
        };
        
        document.head.appendChild(script);
      } catch (error) {
        setOutput('Error loading Python environment. Please refresh the page.');
        console.error('Pyodide loading error:', error);
        setIsLoading(false);
      }
    };

    loadPyodide();
  }, []);

  const runCode = async () => {
    if (!pyodideRef.current) {
      setOutput('Python environment not ready yet. Please wait...');
      return;
    }

    setIsLoading(true);
    setOutput('Running...\n');

    try {
      // Capturar stdout
      await pyodideRef.current.runPythonAsync(`
import sys
from io import StringIO
sys.stdout = StringIO()
`);

      // Ejecutar el código del usuario
      await pyodideRef.current.runPythonAsync(code);

      // Obtener la salida
      const stdout = await pyodideRef.current.runPythonAsync('sys.stdout.getvalue()');
      
      setOutput(stdout || '✓ Code executed successfully (no output)');
    } catch (error: any) {
      setOutput(`Error:\n${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetCode = () => {
    setCode(initialCode);
    setOutput('✓ Code reset to original example.\n');
  };

  return (
    <div className="bg-retro-black border-2 border-retro-orange/30 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-retro-gray/30 px-4 py-3 border-b border-retro-orange/30 flex items-center justify-between">
        <span className="text-retro-orange font-pixel text-xs">{title}</span>
        <div className="flex gap-2">
          <button
            onClick={resetCode}
            className="p-2 hover:bg-retro-orange/20 rounded transition-colors text-gray-400 hover:text-white"
            title="Reset code"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={runCode}
            disabled={isLoading || !pyodideReady}
            className={`
              px-4 py-2 rounded flex items-center gap-2 text-xs font-semibold transition-all
              ${isLoading || !pyodideReady
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-retro-orange text-black hover:bg-white'
              }
            `}
          >
            {isLoading ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play size={14} />
                Run Code
              </>
            )}
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="border-b border-retro-orange/30">
        <CodeMirror
          value={code}
          height="250px"
          theme="dark"
          extensions={[python()]}
          onChange={(value) => setCode(value)}
          className="text-sm"
          basicSetup={{
            lineNumbers: true,
            highlightActiveLineGutter: true,
            highlightActiveLine: true,
            foldGutter: true,
          }}
        />
      </div>

      {/* Output */}
      <div className="bg-[#1e1e1e] p-4">
        <div className="text-xs text-gray-500 mb-2 font-pixel">OUTPUT:</div>
        <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap">
          {output || 'No output yet. Run the code to see results.'}
        </pre>
      </div>
    </div>
  );
};
