"use client";

import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-webgl";

function App() {
  const [imageUrl, setImageUrl] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const [objectsList, setObjectsList] = useState<cocoSsd.DetectedObject[]>([]);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [minConfidence, setMinConfidence] = useState(0.5);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    let isMounted = true;

    async function loadTensorFlowModel() {
      try {
        await import("@tensorflow/tfjs");

        const loadedModel = await cocoSsd.load();
        if (isMounted) {
          setModel(loadedModel);
        }
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    }

    loadTensorFlowModel();

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!imageUrl || !imageLoaded || !model || !imageRef.current) {
      return;
    }

    const image = imageRef.current;

    const detectObjects = async () => {
      try {
        const predictions = await model.detect(image);
        setObjectsList(predictions);
      } catch (error) {
        console.error("Failed to detect objects:", error);
      }
    };

    void detectObjects();
  }, [imageUrl, imageLoaded, model]);

  useEffect(() => {
    if (!canvasRef.current || !imageRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const image = imageRef.current;
    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    const drawImage = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      context.drawImage(image, 0, 0);

      for (let i = 0; i < objectsList.length; i++) {
        const obj = objectsList[i];

        if (obj.score < minConfidence) {
          continue;
        }

        const [startX, startY, boxWidth, boxHeight] = obj.bbox;

        context.strokeStyle = "blue";
        context.lineWidth = 2;
        context.strokeRect(startX, startY, boxWidth, boxHeight);

        const confidencePercent = Math.round(obj.score * 100);
        const labelText = obj.class + " (" + confidencePercent + "%)";

        context.fillStyle = "blue";
        context.font = "16px Arial";
        context.fillText(labelText, startX + 5, startY - 5);
      }
    };

    if (image.complete) {
      drawImage();
    } else {
      image.onload = drawImage;
    }
  }, [imageUrl, objectsList, minConfidence]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("input");
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const url = event.target?.result as string;
      setImageUrl(url);
      setImageLoaded(false);
      setObjectsList([]);
    };
    reader.readAsDataURL(file);
  };

  let countFiltered = 0;
  for (let i = 0; i < objectsList.length; i++) {
    if (objectsList[i].score >= minConfidence) {
      countFiltered = countFiltered + 1;
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>TensorFlow Object Detection</h1>
        </div>
      </header>

      <main className="main-content">
        <div className="panel left-panel">
          {imageUrl === "" ? (
            <div className="upload-area">
              <div className="upload-content">
                <h2>Pick an Image</h2>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                />
              </div>
            </div>
          ) : (
            <div className="canvas-container">
              {/* Need default img for tensorflow, next/image throws src errs*/}
              {/* eslint-disable-next-line @next/next/no-img-element */}
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
                    setImageUrl("");
                    setObjectsList([]);
                  }}
                  className="btn btn-secondary"
                >
                  Clear
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
            </div>

            <div className="predictions-list">
              {objectsList.map((obj, index) => {
                if (obj.score < minConfidence) {
                  return null;
                }

                const score = Math.round(obj.score * 100);
                return (
                  <div key={index} className="prediction-item">
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
