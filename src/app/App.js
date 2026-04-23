import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { filterByConfidence } from './tensorflowUtils';

function App() {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [allPredictions, setAllPredictions] = useState([]); // Store raw data
  const [displayedPredictions, setDisplayedPredictions] = useState([]); // Filtered data
  const [isLoading, setIsLoading] = useState(false);
  const [confidence, setConfidence] = useState(0.5);
  const [model, setModel] = useState(null);
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const fileInputRef = useRef(null);

  // 1. Load Model on Startup
  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
        console.log('✅ COCO-SSD model loaded');
      } catch (error) {
        console.error('❌ Failed to load model:', error);
      }
    };
    loadModel();
  }, []);

  // 2. Handle Filtering (When confidence slider moves)
  useEffect(() => {
    const filtered = filterByConfidence(allPredictions, confidence);
    setDisplayedPredictions(filtered);
  }, [allPredictions, confidence]);

  // 3. Draw to Canvas (When image or predictions change)
  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !uploadedImage) return;

    const canvas = canvasRef.current;
    const img = imageRef.current;
    const ctx = canvas.getContext('2d');

    const draw = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      displayedPredictions.forEach((pred) => {
        const [x, y, w, h] = pred.bbox;
        const color = '#4f46e5';

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        const label = `${pred.class} ${(pred.score * 100).toFixed(0)}%`;
        ctx.font = 'bold 16px sans-serif';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = color;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 5, y - 7);
      });
    };

    if (img.complete) {
        draw();
    } else {
        img.onload = draw;
    }
  }, [uploadedImage, displayedPredictions]);

  const handleFileSelect = (file) => {
    if (!file?.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target.result);
      setAllPredictions([]);
    };
    reader.readAsDataURL(file);
  };

  const handleDetect = async () => {
    if (!model || !imageRef.current) return;
    setIsLoading(true);
    
    try {
      const rawPredictions = await model.detect(imageRef.current);
      setAllPredictions(rawPredictions);
    } catch (error) {
      console.error('Detection error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>📸 Annotate</h1>
          <p>AI Object Detection</p>
        </div>
      </header>

      <main className="main-content">
        <div className="panel left-panel">
          {!uploadedImage ? (
            <div className="upload-area" onClick={() => fileInputRef.current.click()}>
              <div className="upload-content">
                <h2>Upload Image</h2>
                <p>Click to browse</p>
              </div>
              <input 
                ref={fileInputRef} 
                type="file" 
                hidden 
                onChange={(e) => handleFileSelect(e.target.files[0])} 
              />
            </div>
          ) : (
            <div className="canvas-container">
              <img ref={imageRef} src={uploadedImage} alt="hidden" style={{ display: 'none' }} crossOrigin="anonymous" />
              <canvas ref={canvasRef} className="canvas" />
              
              <div className="button-group">
                <button className="btn btn-primary" onClick={handleDetect} disabled={isLoading || !model}>
                  {isLoading ? '⏳ Analyzing...' : '🔍 Detect Objects'}
                </button>
                <button className="btn btn-secondary" onClick={() => setUploadedImage(null)}>
                  ➕ New Image
                </button>
              </div>
            </div>
          )}
        </div>

        {uploadedImage && (
          <div className="panel right-panel">
            <h3>Settings</h3>
            <div className="control-group">
              <label>Threshold: {(confidence * 100).toFixed(0)}%</label>
              <input 
                type="range" min="0" max="1" step="0.05" 
                value={confidence} 
                onChange={(e) => setConfidence(parseFloat(e.target.value))} 
              />
            </div>

            <h3>Results</h3>
            <div className="stat">
                <strong>{displayedPredictions.length}</strong> Objects Found
            </div>

            <div className="predictions-list">
              {displayedPredictions.map((pred, idx) => (
                <div key={idx} className="prediction-item">
                  <span>{pred.class}</span>
                  <span>{(pred.score * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;