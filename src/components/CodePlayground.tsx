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
  const [output, setOutput] = useState('Click "Run Code" to execute Python code in your browser.');
  const [isLoading, setIsLoading] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [mounted, setMounted] = useState(false);
  const pyodideRef = useRef<any>(null);
  const loadingRef = useRef(false);

  // Verificar que estamos en el cliente
  useEffect(() => {
    setMounted(true);
  }, []);

  // Cargar Pyodide solo cuando el usuario haga click por primera vez
  const loadPyodide = async () => {
    if (loadingRef.current || pyodideRef.current) return;
    
    loadingRef.current = true;
    setIsLoading(true);
    setOutput('Loading Python environment (first time only, ~15-20 seconds)...\n');
    
    try {
      // Verificar si ya existe el script
      if (!document.getElementById('pyodide-script')) {
        const script = document.createElement('script');
        script.id = 'pyodide-script';
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
        script.async = true;
        
        await new Promise<void>((resolve, reject) => {
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load Pyodide script'));
          document.head.appendChild(script);
        });
      }
      
      // @ts-ignore
      if (typeof window.loadPyodide === 'function') {
        // @ts-ignore
        const pyodide = await window.loadPyodide({
          indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/',
        });
        
        // Pre-instalar paquetes comunes de ML/CV disponibles en Pyodide
        setOutput(prev => prev + 'Installing packages (numpy, matplotlib, scikit-learn, pandas)...\n');
        await pyodide.loadPackage(['numpy', 'matplotlib', 'scikit-learn', 'pandas']);
        
        pyodideRef.current = pyodide;
        setPyodideReady(true);
        setOutput('✓ Python ready with ML libraries! Running your code...\n');
        return true;
      }
    } catch (error: any) {
      console.error('Error loading Pyodide:', error);
      setOutput(`Error loading Python: ${error.message}\nPlease refresh the page and try again.`);
      setIsLoading(false);
      loadingRef.current = false;
      return false;
    }
    
    return true;
  };

  const runCode = async () => {
    // Cargar Pyodide si no está listo
    if (!pyodideRef.current) {
      const loaded = await loadPyodide();
      if (!loaded) return;
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
    setOutput(pyodideReady 
      ? '✓ Code reset to original example. Click "Run Code" to execute.\n'
      : 'Click "Run Code" to execute Python code in your browser.'
    );
  };

  if (!mounted) {
    return (
      <div className="bg-retro-black border-2 border-retro-orange/30 rounded-lg overflow-hidden p-8 text-center">
        <Loader2 className="animate-spin mx-auto mb-2 text-retro-orange" size={24} />
        <p className="text-gray-400 text-sm">Loading editor...</p>
      </div>
    );
  }

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
            disabled={isLoading}
            className={`
              px-4 py-2 rounded flex items-center gap-2 text-xs font-semibold transition-all
              ${isLoading
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
