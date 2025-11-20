import React from 'react';
import RobustDetector from './components/Detector';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-4 sm:p-8 font-sans selection:bg-indigo-500 selection:text-white">
      <header className="max-w-5xl mx-auto mb-12 text-center pt-8">
         <h1 className="text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-indigo-400 pb-2">
            ObjectSense AI
         </h1>
         <p className="text-lg text-slate-400 mt-4 max-w-2xl mx-auto leading-relaxed">
            Experience state-of-the-art object detection running entirely locally in your browser using Hugging Face Transformers and WebAssembly. No server uploads, 100% private.
         </p>
      </header>

      <main>
        <RobustDetector />
      </main>
      
      <footer className="mt-20 text-center text-slate-500 text-sm pb-8">
        <p>Powered by <a href="https://huggingface.co/Xenova/detr-resnet-50" className="text-indigo-400 hover:text-indigo-300 underline decoration-slate-700 underline-offset-4">Xenova/detr-resnet-50</a> & React</p>
      </footer>
    </div>
  );
};

export default App;