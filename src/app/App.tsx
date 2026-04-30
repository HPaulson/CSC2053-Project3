"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import "./App.css";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-webgl";

// Color palette for different object classes
const CLASS_COLORS: { [key: string]: string } = {
  person: "#ef4444",
  bicycle: "#f97316",
  car: "#eab308",
  motorcycle: "#84cc16",
  airplane: "#22c55e",
  bus: "#14b8a6",
  train: "#06b6d4",
  truck: "#3b82f6",
  boat: "#8b5cf6",
  cat: "#ec4899",
  dog: "#f43f5e",
  horse: "#fb923c",
  sheep: "#a3e635",
  cow: "#34d399",
  elephant: "#38bdf8",
  bear: "#818cf8",
  zebra: "#f472b6",
  giraffe: "#fb7185",
  bird: "#4ade80",
  backpack: "#facc15",
  umbrella: "#60a5fa",
  handbag: "#c084fc",
  tie: "#f87171",
  suitcase: "#34d399",
  bottle: "#94a3b8",
  cup: "#fbbf24",
  fork: "#a78bfa",
  knife: "#f472b6",
  spoon: "#86efac",
  bowl: "#67e8f9",
  banana: "#fde047",
  apple: "#4ade80",
  sandwich: "#fdba74",
  orange: "#fb923c",
  broccoli: "#86efac",
  chair: "#a3a3a3",
  couch: "#c4b5fd",
  bed: "#fca5a5",
  laptop: "#6ee7b7",
  mouse: "#bef264",
  keyboard: "#7dd3fc",
  phone: "#f9a8d4",
  book: "#fde68a",
  clock: "#a5b4fc",
  vase: "#fda4af",
  scissors: "#d9f99d",
  toothbrush: "#e9d5ff",
};

function getClassColor(className: string): string {
  return CLASS_COLORS[className.toLowerCase()] || "#6366f1";
}

function App() {
  const [imageUrl, setImageUrl] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const [objectsList, setObjectsList] = useState<cocoSsd.DetectedObject[]>([]);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [darkMode, setDarkMode] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load model
  useEffect(() => {
    let isMounted = true;
    async function loadTensorFlowModel() {
      try {
        await import("@tensorflow/tfjs");
        const loadedModel = await cocoSsd.load();
        if (isMounted) setModel(loadedModel);
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    }
    loadTensorFlowModel();
    return () => { isMounted = false; };
  }, []);

  // Dark mode class on body
  useEffect(() => {
    document.body.classList.toggle("dark-mode", darkMode);
  }, [darkMode]);

  // Detect objects when image is ready
  useEffect(() => {
    if (!imageUrl || !imageLoaded || !model || !imageRef.current) return;
    const detectObjects = async () => {
      try {
        const predictions = await model.detect(imageRef.current!);
        setObjectsList(predictions);
      } catch (error) {
        console.error("Failed to detect objects:", error);
      }
    };
    void detectObjects();
  }, [imageUrl, imageLoaded, model]);

  // Draw boxes on canvas
  useEffect(() => {
    if (!canvasRef.current || !imageRef.current) return;
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const context = canvas.getContext("2d");
    if (!context) return;

    const drawImage = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      context.drawImage(image, 0, 0);

      for (let i = 0; i < objectsList.length; i++) {
        const obj = objectsList[i];
        if (obj.score < minConfidence) continue;

        const [startX, startY, boxWidth, boxHeight] = obj.bbox;
        const color = getClassColor(obj.class);

        // Draw box
        context.strokeStyle = color;
        context.lineWidth = 3;
        context.strokeRect(startX, startY, boxWidth, boxHeight);

        // Draw label background
        const confidencePercent = Math.round(obj.score * 100);
        const labelText = `${obj.class} (${confidencePercent}%)`;
        context.font = "bold 16px Arial";
        const textWidth = context.measureText(labelText).width;
        context.fillStyle = color;
        context.fillRect(startX, startY - 26, textWidth + 10, 26);

        // Draw label text
        context.fillStyle = "#ffffff";
        context.fillText(labelText, startX + 5, startY - 7);
      }
    };

    if (image.complete) drawImage();
    else image.onload = drawImage;
  }, [imageUrl, objectsList, minConfidence]);

  // Paste from clipboard (Ctrl+V)
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) handleFile(file);
          break;
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const url = event.target?.result as string;
      setImageUrl(url);
      setImageLoaded(false);
      setObjectsList([]);
    };
    reader.readAsDataURL(file);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  // Drag and drop handlers
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, []);

  // Download annotated image
  const handleDownload = () => {
    if (!canvasRef.current) return;
    const link = document.createElement("a");
    link.download = "detected-objects.png";
    link.href = canvasRef.current.toDataURL();
    link.click();
  };

  // Count filtered objects and breakdown by class
  let countFiltered = 0;
  const classCounts: { [key: string]: number } = {};
  for (let i = 0; i < objectsList.length; i++) {
    if (objectsList[i].score >= minConfidence) {
      countFiltered++;
      const cls = objectsList[i].class;
      classCounts[cls] = (classCounts[cls] || 0) + 1;
    }
  }

  return (

    /* Header of Whole Page */
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>TensorFlow Object Detection</h1>
        </div>
        <button
          className="btn btn-secondary dark-toggle"
          onClick={() => setDarkMode(!darkMode)}
        >
          {darkMode ? "☀ Light Mode" : "☾ Dark Mode"}
        </button>
      </header>

      <main className="main-content">

        <div className="panel left-panel">
          {imageUrl === "" ? (
            <div
              className={`upload-area ${isDragging ? "dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-content">
                <h2>Pick an Image</h2>
                <p>Click, drag & drop, or paste (Ctrl+V)</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  style={{ display: "none" }}
                />
              </div>
            </div>
          ) : (
            <div className="canvas-container">

              <img
                ref={imageRef}
                src={imageUrl}
                alt="uploaded"
                style={{ display: "none" }}
                onLoad={() => setImageLoaded(true)}
              />
              <canvas ref={canvasRef} className="canvas" />
              <div className="button-group">
                <button
                  onClick={() => {
                    // This button clears the UI
                    setImageUrl("");
                    setObjectsList([]);
                  }}
                  className="btn btn-secondary"
                >
                  Clear
                </button>
                <button onClick={handleDownload} className="btn btn-primary">
                  Download Image
                </button>
              </div>
            </div>
          )}
        </div>
         
         
        {imageUrl !== "" && (
          <div className="panel right-panel">
            
            <div className="control-group">
              <label>Confidence: {Math.round(minConfidence * 100)}%</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                className="slider"
              />
            </div>

            
            <div className="stat">
              <strong>{countFiltered}</strong> Objects Found
              {Object.keys(classCounts).length > 0 && (
                <div className="class-counts">
                  {Object.entries(classCounts).map(([cls, count]) => (
                    <div key={cls} className="class-count-item">
                      <span
                        className="class-dot"
                        style={{ background: getClassColor(cls) }}
                      />
                      <span>{cls}</span>
                      <span className="class-count-num">{count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            
            <div className="predictions-list">
              {objectsList.map((obj, index) => {
                // This makes sure list matches whats on the picture
                if (obj.score < minConfidence) {
                  return null;
                }

                const score = Math.round(obj.score * 100);
                const color = getClassColor(obj.class);
                return (
                  <div
                    key={index}
                    className="prediction-item"
                    style={{ borderLeft: `4px solid ${color}` }}
                  >
                    <span>{obj.class}</span>
                    <span>{score}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
