import React, { useState, useEffect, useRef, useCallback } from 'react';
import { pipeline } from '@huggingface/transformers';
import { DetectionResult, AppStatus, PipelineType } from '../types';
import { Spinner } from './Spinner';

export const Detector: React.FC = () => {
  const [status, setStatus] = useState<AppStatus>(AppStatus.LOADING_MODEL);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<DetectionResult[]>([]);
  const [progress, setProgress] = useState<string>('');
  
  // Refs to manage lifecycle and singleton pipeline
  const detectorRef = useRef<PipelineType | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize the model
  useEffect(() => {
    const loadModel = async () => {
      try {
        // @ts-ignore - The progress callback type definition can be tricky to align with strict TS
        detectorRef.current = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
          progress_callback: (data: any) => {
            if (data.status === 'progress') {
               const p = data.progress ? Math.round(data.progress) : 0;
               setProgress(`Downloading ${data.file}: ${p}%`);
            } else if (data.status === 'ready') {
               setProgress('Initializing...');
            }
          }
        });
        setStatus(AppStatus.READY);
      } catch (error) {
        console.error("Failed to load model:", error);
        setStatus(AppStatus.ERROR);
      }
    };

    loadModel();
  }, []);

  // Run detection when image changes
  const runDetection = useCallback(async (src: string) => {
    if (!detectorRef.current) return;

    setStatus(AppStatus.ANALYZING);
    setDetections([]);
    
    try {
      // The pipeline accepts the image URL directly
      const results = await detectorRef.current(src, { threshold: 0.5, percentage: true });
      setDetections(results);
      setStatus(AppStatus.READY);
    } catch (err) {
      console.error("Detection error:", err);
      setStatus(AppStatus.ERROR);
    }
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const src = evt.target?.result as string;
      setImageSrc(src);
      runDetection(src);
    };
    reader.readAsDataURL(file);
  };

  const handleSampleImage = () => {
     const sampleUrl = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";
     setImageSrc(sampleUrl);
     runDetection(sampleUrl);
  };

  // Calculate styles for bounding boxes
  // Since DETR via transformers.js often returns normalized coordinates or pixels depending on config.
  // The default pipeline call usually handles normalization if we ask it to, or we handle it here.
  // 'percentage: true' in options returns { box: { xmax, xmin, ymax, ymin } } as fractions 0-1? 
  // Wait, transformers.js documentation says 'percentage: true' returns detections with bounding box coordinates as percentages.
  // Let's verify logic:
  // If percentage=true: xmin, ymin, etc are 0 to 1.
  // Style: left: xmin * 100%, top: ymin * 100%, width: (xmax-xmin)*100%, height: (ymax-ymin)*100%.
  
  const renderBox = (det: DetectionResult, index: number) => {
    const { box, label, score } = det;
    // Assuming percentage: true was passed to the pipeline
    const left = `${box.xmin * 100}%`;
    const top = `${box.ymin * 100}%`;
    const width = `${(box.xmax - box.xmin) * 100}%`;
    const height = `${(box.ymax - box.ymin) * 100}%`;
    
    // Generate a consistent color based on label
    const colorHue = label.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % 360;
    const color = `hsl(${colorHue}, 80%, 60%)`;

    return (
      <div
        key={index}
        className="absolute border-2 flex flex-col"
        style={{ left, top, width, height, borderColor: color, boxShadow: `0 0 4px ${color}` }}
      >
        <span 
          className="absolute -top-7 left-0 text-xs font-bold px-2 py-1 rounded text-white shadow-sm whitespace-nowrap"
          style={{ backgroundColor: color }}
        >
          {label} ({Math.round(score * 100)}%)
        </span>
        <div className="w-full h-full opacity-10" style={{ backgroundColor: color }}></div>
      </div>
    );
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-slate-800/50 rounded-2xl shadow-xl border border-slate-700 backdrop-blur-sm">
      
      {/* Header / Controls */}
      <div className="flex flex-col md:flex-row items-center justify-between mb-6 gap-4">
        <div>
           <h2 className="text-2xl font-bold text-white tracking-tight">Detr-ResNet-50</h2>
           <p className="text-slate-400 text-sm">Client-side object detection powered by Transformers.js</p>
        </div>

        <div className="flex items-center gap-3">
            <button
              onClick={handleSampleImage}
              disabled={status === AppStatus.LOADING_MODEL || status === AppStatus.ANALYZING}
              className="px-4 py-2 rounded-lg bg-slate-700 text-slate-200 text-sm font-medium hover:bg-slate-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Try Sample
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={status === AppStatus.LOADING_MODEL || status === AppStatus.ANALYZING}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-bold hover:bg-indigo-500 transition-all shadow-lg shadow-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
               {status === AppStatus.ANALYZING ? <Spinner /> : (
                 <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
               )}
               Upload Image
            </button>
            <input 
              ref={fileInputRef} 
              type="file" 
              className="hidden" 
              accept="image/*" 
              onChange={handleFileUpload} 
            />
        </div>
      </div>

      {/* Main Display Area */}
      <div className="relative w-full min-h-[400px] bg-slate-900/80 rounded-xl overflow-hidden border border-slate-700 flex items-center justify-center group">
        
        {status === AppStatus.LOADING_MODEL && (
          <div className="flex flex-col items-center gap-4 p-6 text-center z-10">
            <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
            <div>
              <p className="text-lg font-medium text-white">Loading Model...</p>
              <p className="text-sm text-slate-400 mt-1 max-w-xs">{progress || "Fetching parameters (~160MB)..."}</p>
            </div>
          </div>
        )}

        {status === AppStatus.ERROR && (
           <div className="text-red-400 flex flex-col items-center gap-2">
             <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
             <span>Something went wrong initializing the model.</span>
           </div>
        )}

        {!imageSrc && status === AppStatus.READY && (
          <div className="text-slate-500 flex flex-col items-center">
            <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            <p>Upload an image or try sample to start detecting objects</p>
          </div>
        )}

        {imageSrc && (
            <div className="relative w-full h-full flex justify-center">
                 <img 
                   ref={imageRef}
                   src={imageSrc} 
                   alt="Analyzed content" 
                   className={`max-w-full max-h-[70vh] object-contain shadow-2xl transition-opacity duration-300 ${status === AppStatus.ANALYZING ? 'opacity-50 blur-sm' : 'opacity-100'}`}
                 />
                 
                 {/* Overlay Container - Must match image dimensions perfectly to align boxes 
                     In this implementation, we rely on percentages relative to the wrapping div 
                     which collapses to image size because of flex center, but let's ensure absolute positioning works contextually.
                     Actually, a common trick is to put the boxes inside a div that has the exact aspect ratio or fits the image.
                     However, simpler for responsive: absolute full width/height over the image container.
                 */}
                 
                 <div className="absolute inset-0 pointer-events-none">
                   {/* We need a container that matches the image exact rendered size for % boxes to work relative to image, not the container div which might be wider */}
                   {/* For simplicity in this demo, assuming image is centered and object-contain. 
                       For precise boxes on 'object-contain' images, we would need JS to measure rendered size vs natural size. 
                       However, if we force the wrapper to wrap the image exactly, it's easier.
                   */}
                 </div>

                 {/* Render Bounding Boxes inside a container that strictly follows image flow? 
                     No, standard HTML img flow is hard to overlay exactly without a wrapper.
                     Let's try a wrapper method for the image.
                  */}
            </div>
        )}
        
        {/* Precise Overlay Wrapper: Re-implementing image rendering to ensure boxes overlay correctly */}
        {imageSrc && status !== AppStatus.LOADING_MODEL && (
             <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="relative inline-block">
                   {/* Hidden ghost image to set container size if we wanted, but simpler is to map boxes to the displayed image if we could.
                       Actually, to be robust: 
                   */}
                   <img src={imageSrc} className="invisible max-w-full max-h-[70vh]" alt="ghost" />
                   
                   {/* The actual boxes, positioned absolutely over the centered content area. 
                       Wait, the previous img was already rendered.
                       Let's fix the layout structure below in the actual return.
                   */}
                </div>
             </div>
        )}
      </div>
      
      {/* Clean Implementation for Overlay */}
      {imageSrc && (
        <div className="mt-0 hidden">
           {/* This is a hacky thought process trace. I will fix the main render block below. */}
        </div>
      )}
      
    </div>
  );
};

// Re-writing the return of Detector to be robust about overlay positioning
const RobustDetector: React.FC = () => {
    // ... (reuse logic from above, just fixing the render part)
    const [status, setStatus] = useState<AppStatus>(AppStatus.LOADING_MODEL);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [detections, setDetections] = useState<DetectionResult[]>([]);
    const [progress, setProgress] = useState<string>('');
    const detectorRef = useRef<PipelineType | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const loadModel = async () => {
          try {
            // @ts-ignore
            detectorRef.current = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
              progress_callback: (data: any) => {
                if (data.status === 'progress') {
                    const p = data.progress ? Math.round(data.progress) : 0;
                    setProgress(`Loading Model: ${p}%`);
                } else if (data.status === 'ready') {
                    setProgress('Initializing...');
                }
              }
            });
            setStatus(AppStatus.READY);
          } catch (error) {
            console.error(error);
            setStatus(AppStatus.ERROR);
          }
        };
        loadModel();
    }, []);

    const runDetection = useCallback(async (src: string) => {
        if (!detectorRef.current) return;
        setStatus(AppStatus.ANALYZING);
        setDetections([]);
        try {
            const results = await detectorRef.current(src, { threshold: 0.5, percentage: true });
            setDetections(results);
            setStatus(AppStatus.READY);
        } catch (err) {
            setStatus(AppStatus.ERROR);
        }
    }, []);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            const src = evt.target?.result as string;
            setImageSrc(src);
            runDetection(src);
        };
        reader.readAsDataURL(file);
    };

    const handleSampleImage = () => {
         const sampleUrl = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";
         setImageSrc(sampleUrl);
         runDetection(sampleUrl);
    };

    return (
        <div className="w-full max-w-5xl mx-auto p-6 bg-slate-800/50 rounded-3xl shadow-2xl border border-slate-700/50 backdrop-blur-xl">
             <div className="flex flex-col md:flex-row items-center justify-between mb-8 gap-6 border-b border-slate-700/50 pb-6">
                <div>
                   <div className="flex items-center gap-3 mb-1">
                     <div className="w-2 h-8 bg-indigo-500 rounded-full"></div>
                     <h2 className="text-3xl font-bold text-white tracking-tight">ObjectSense AI</h2>
                   </div>
                   <p className="text-slate-400 text-sm pl-5">Powered by DETR ResNet-50 & WebAssembly</p>
                </div>
        
                <div className="flex items-center gap-3">
                    <button
                      onClick={handleSampleImage}
                      disabled={status === AppStatus.LOADING_MODEL || status === AppStatus.ANALYZING}
                      className="px-5 py-2.5 rounded-xl bg-slate-700/50 text-slate-200 text-sm font-medium hover:bg-slate-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed border border-slate-600"
                    >
                      Use Sample
                    </button>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={status === AppStatus.LOADING_MODEL || status === AppStatus.ANALYZING}
                      className="group flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 text-white text-sm font-bold hover:bg-indigo-500 transition-all shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 disabled:opacity-50 disabled:cursor-not-allowed active:scale-95"
                    >
                       {status === AppStatus.ANALYZING ? <Spinner /> : (
                         <svg className="w-5 h-5 group-hover:-translate-y-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                       )}
                       <span>Upload Image</span>
                    </button>
                    <input 
                      ref={fileInputRef} 
                      type="file" 
                      className="hidden" 
                      accept="image/*" 
                      onChange={handleFileUpload} 
                    />
                </div>
              </div>

              <div className="relative min-h-[500px] bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center">
                 
                 {status === AppStatus.LOADING_MODEL && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm">
                        <div className="relative">
                             <div className="w-16 h-16 border-4 border-slate-700 rounded-full"></div>
                             <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin absolute top-0 left-0"></div>
                        </div>
                        <p className="mt-6 text-lg font-medium text-white animate-pulse">Initializing Neural Network</p>
                        <p className="text-sm text-slate-500 mt-2">{progress || "Loading model parameters..."}</p>
                    </div>
                 )}

                 {!imageSrc && status === AppStatus.READY && (
                    <div className="text-center p-10">
                        <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 shadow-inner">
                            <svg className="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path></svg>
                        </div>
                        <h3 className="text-xl font-semibold text-slate-200">Ready to Detect</h3>
                        <p className="text-slate-500 mt-2 max-w-sm mx-auto">Upload a photo or use our sample image to see the browser-based Transformer model in action.</p>
                    </div>
                 )}

                 {imageSrc && (
                    <div className="relative inline-block max-w-full">
                        <img src={imageSrc} alt="Target" className="max-w-full max-h-[70vh] block rounded-lg" />
                        
                        {/* Bounding Boxes */}
                        {detections.map((det, i) => {
                            const { box, label, score } = det;
                            // Using percentage:true allows us to use percentages directly
                            return (
                                <div
                                    key={i}
                                    className="absolute border-2 box-border group cursor-help"
                                    style={{
                                        left: `${box.xmin * 100}%`,
                                        top: `${box.ymin * 100}%`,
                                        width: `${(box.xmax - box.xmin) * 100}%`,
                                        height: `${(box.ymax - box.ymin) * 100}%`,
                                        borderColor: `hsl(${(label.length * 50) % 360}, 80%, 60%)`,
                                        backgroundColor: `hsla(${(label.length * 50) % 360}, 80%, 60%, 0.1)`,
                                    }}
                                >
                                    <div 
                                        className="absolute -top-8 left-[-2px] px-3 py-1 rounded-md text-xs font-bold text-white shadow-lg whitespace-nowrap transition-all z-10"
                                        style={{
                                            backgroundColor: `hsl(${(label.length * 50) % 360}, 80%, 60%)`
                                        }}
                                    >
                                        {label} <span className="opacity-80 font-normal">{(score * 100).toFixed(1)}%</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                 )}
              </div>
              
              {/* Results Summary */}
              {detections.length > 0 && (
                  <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                     {detections.map((det, i) => (
                         <div key={i} className="bg-slate-700/30 border border-slate-700 rounded-lg p-3 flex items-center justify-between">
                            <span className="text-slate-200 font-medium capitalize">{det.label}</span>
                            <span className="text-xs text-indigo-400 font-mono font-bold">{(det.score * 100).toFixed(0)}%</span>
                         </div>
                     ))}
                  </div>
              )}
        </div>
    );
}

export default RobustDetector;