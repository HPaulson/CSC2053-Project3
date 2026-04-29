"use client"; 

import React, { useState, useRef, useEffect } from "react";
import "./App.css";
//Importing the COCO-SSD model for object detection
import * as cocoSsd from "@tensorflow-models/coco-ssd";
//Importing the WebGL backend for TensorFlow
import "@tensorflow/tfjs-backend-webgl";

function App() {
  //Declaring states
  const [imageUrl, setImageUrl] = useState(""); // This holds URL for the image
  const [imageLoaded, setImageLoaded] = useState(false); //Says if image is ready or not
  const [objectsList, setObjectsList] = useState<cocoSsd.DetectedObject[]>([]); // This stores an array of the objects in the image given
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null); //This holds the loaded instance
  const [minConfidence, setMinConfidence] = useState(0.5); //This filters the really low confidence predictions

  //These reference DOM so we can manipulate and add things, like the boxes
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  //This initializes the model so we can use it
  useEffect(() => {
    let isMounted = true;

    async function loadTensorFlowModel() {
      try {
        
        await import("@tensorflow/tfjs");
        const loadedModel = await cocoSsd.load();
        
        //Only update the state if the component hasn't been unmounted, or else it would have nothing to update
        if (isMounted) {
          setModel(loadedModel);
        }
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    }

    loadTensorFlowModel();

    //Cleanup
    return () => {
      isMounted = false;
    };
  }, []);

  // Runs if source, status or model changes. Need these to change or else it's the same thing
  useEffect(() => {
    if (!imageUrl || !imageLoaded || !model || !imageRef.current) {
      return;
    }

    const image = imageRef.current;

    const detectObjects = async () => {
      try {
        //Detect Objects
        const predictions = await model.detect(image);
        setObjectsList(predictions);
      } catch (error) {
        console.error("Failed to detect objects:", error);
      }
    };

    void detectObjects();
  }, [imageUrl, imageLoaded, model]);

  //This draws the boxes around each object
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
      //Set dimensions to match that of the image, so it knows where objects are
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      
      //Makes the uploaded image the background, so boxes will lay on top of given image
      context.drawImage(image, 0, 0);

      //Loop through each detection and draw a box around it 
      for (let i = 0; i < objectsList.length; i++) {
        const obj = objectsList[i];

        //Skip the detected objects that fall below the wanted confidence
        if (obj.score < minConfidence) {
          continue;
        }

        //Get the box coordinates
        const [startX, startY, boxWidth, boxHeight] = obj.bbox;

        //Draw the boxes, setting color and width using the coordinates from the box object
        context.strokeStyle = "blue";
        context.lineWidth = 2;
        context.strokeRect(startX, startY, boxWidth, boxHeight);

        //Draw the label on the box
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

  //Converts to URL for DOM
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("input");
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const url = event.target?.result as string;
      setImageUrl(url); //Triggers detection
      setImageLoaded(false); //Reset for next image
      setObjectsList([]); //Clear
    };
    reader.readAsDataURL(file);
  };

 
  //Calculates # of objects that meet confidence
  let countFiltered = 0;
  for (let i = 0; i < objectsList.length; i++) {
    if (objectsList[i].score >= minConfidence) {
      countFiltered = countFiltered + 1;
    }
  }


  return (

    /* Header of Whole Page */
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1>TensorFlow Object Detection</h1>
        </div>
      </header>

      <main className="main-content">
        
        <div className="panel left-panel">
          {imageUrl === "" ? (
            //Beginning prompting user to pick an image
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
            //After upload
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
              </div>
            </div>
          )}
        </div>
         
         /* This side bar only comes up once an image is put in */
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

            // this shows how many objects are above the current set confidence threshold
            <div className="stat">
              <strong>{countFiltered}</strong> Objects Found
            </div>

            
            <div className="predictions-list">
              {objectsList.map((obj, index) => {
                // This makes sure list matches whats on the picture
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