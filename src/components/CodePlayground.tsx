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
  const [output, setOutput] = useState('Haz clic en "Ejecutar Código" para ejecutar Python en tu navegador.');
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
    setOutput('Cargando entorno Python (solo la primera vez, ~15-20 segundos)...\n');
    
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
        setOutput(prev => prev + 'Instalando paquetes (numpy, matplotlib, scikit-learn, pandas)...\n');
        await pyodide.loadPackage(['numpy', 'matplotlib', 'scikit-learn', 'pandas']);
        
        pyodideRef.current = pyodide;
        setPyodideReady(true);
        setOutput('✓ Python listo con librerías ML! Ejecutando tu código...\n');
        return true;
      }
    } catch (error: any) {
      console.error('Error loading Pyodide:', error);
      setOutput(`Error cargando Python: ${error.message}\nPor favor recarga la página e intenta de nuevo.`);
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
    setOutput('Ejecutando...\n');

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
      
      setOutput(stdout || '✓ Código ejecutado exitosamente (sin salida)');
    } catch (error: any) {
      setOutput(`Error:\n${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetCode = () => {
    setCode(initialCode);
    setOutput(pyodideReady 
      ? '✓ Código reseteado al ejemplo original. Haz clic en "Ejecutar Código" para ejecutar.\n'
      : 'Haz clic en "Ejecutar Código" para ejecutar Python en tu navegador.'
    );
  };

  if (!mounted) {
    return (
      <div className="bg-retro-black border-2 border-retro-orange/30 rounded-lg overflow-hidden p-8 text-center">
        <Loader2 className="animate-spin mx-auto mb-2 text-retro-orange" size={24} />
        <p className="text-gray-400 text-sm">Cargando editor...</p>
      </div>
    );
  }

  return (
    <div className="bg-retro-black border-2 border-retro-orange/30 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-retro-gray/30 px-3 md:px-4 py-3 border-b border-retro-orange/30 flex items-center justify-between gap-2">
        <span className="text-retro-orange font-pixel text-[10px] md:text-xs flex-1 truncate">{title}</span>
        <div className="flex gap-2 flex-shrink-0">
          <button
            onClick={resetCode}
            className="p-3 md:p-2 hover:bg-retro-orange/20 rounded transition-colors text-gray-400 hover:text-white min-w-[44px] min-h-[44px] flex items-center justify-center"
            title="Reset code"
          >
            <RotateCcw size={16} />
          </button>
          <button
            onClick={runCode}
            disabled={isLoading}
            className={`
              px-4 md:px-4 py-3 md:py-2 rounded flex items-center gap-2 text-xs font-semibold transition-all min-h-[44px]
              ${isLoading
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-retro-orange text-black hover:bg-white'
              }
            `}
          >
            {isLoading ? (
              <>
                <Loader2 size={14} className="animate-spin" />
                <span className="hidden md:inline">Ejecutando...</span>
              </>
            ) : (
              <>
                <Play size={14} />
                <span className="hidden md:inline">Ejecutar Código</span>
              </>
            )}
          </button>
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

      {/* Output */}
      <div className="bg-[#1e1e1e] p-3 md:p-4">
        <div className="text-[10px] md:text-xs text-gray-500 mb-2 font-pixel">SALIDA:</div>
        <pre className="text-green-400 font-mono text-xs md:text-sm whitespace-pre-wrap break-words">
          {output || 'Sin salida aún. Ejecuta el código para ver los resultados.'}
        </pre>
      </div>
    </div>
  );
};
