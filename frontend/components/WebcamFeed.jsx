import { useEffect, useMemo, useRef, useState } from "react";

const FRAME_WIDTH = 640;
const FRAME_HEIGHT = 480;

function getBoxStyle(box) {
  if (!box) {
    return {};
  }

  return {
    left: `${(box.x / FRAME_WIDTH) * 100}%`,
    top: `${(box.y / FRAME_HEIGHT) * 100}%`,
    width: `${(box.w / FRAME_WIDTH) * 100}%`,
    height: `${(box.h / FRAME_HEIGHT) * 100}%`
  };
}

export default function WebcamFeed({ onFrame, prediction, connected }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => {
    let stream;

    const setupCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
            facingMode: "user"
          },
          audio: false
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        setCameraError("Camera access failed. Please allow webcam permissions.");
      }
    };

    setupCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (!connected || !onFrame) {
      return;
    }

    const interval = window.setInterval(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        return;
      }

      canvas.width = FRAME_WIDTH;
      canvas.height = FRAME_HEIGHT;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);

      const encoded = canvas.toDataURL("image/jpeg", 0.72);
      onFrame(encoded);
    }, 90);

    return () => window.clearInterval(interval);
  }, [connected, onFrame]);

  const eyeBoxes = useMemo(
    () => prediction?.overlays?.eye_boxes ?? [],
    [prediction]
  );
  const faceBox = prediction?.overlays?.face_box;

  return (
    <div className="panel shadow-panel relative overflow-hidden rounded-2xl p-3">
      <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-cyan-200/20 bg-slate-950/70">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="h-full w-full object-cover"
        />
        <canvas ref={canvasRef} className="hidden" />

        <div className="pointer-events-none absolute inset-0">
          {faceBox ? (
            <div
              className="absolute border-2 border-cyan-300/70"
              style={getBoxStyle(faceBox)}
            />
          ) : null}

          {eyeBoxes.map((box, idx) => (
            <div
              key={`${box.x}-${box.y}-${idx}`}
              className="absolute border border-amber-300/80"
              style={getBoxStyle(box)}
            />
          ))}
        </div>

        <div className="absolute left-3 top-3 rounded-md bg-black/45 px-3 py-1 text-xs tracking-wide text-cyan-100">
          {connected ? "Live stream active" : "Connecting to backend"}
        </div>
      </div>

      {cameraError ? (
        <p className="mt-3 rounded-md border border-rose-400/40 bg-rose-500/10 p-2 text-sm text-rose-200">
          {cameraError}
        </p>
      ) : null}
    </div>
  );
}
