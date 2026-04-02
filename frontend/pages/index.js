import { motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";

import AlertSystem from "../components/AlertSystem";
import ConfidenceBar from "../components/ConfidenceBar";
import StatusIndicator from "../components/StatusIndicator";
import WebcamFeed from "../components/WebcamFeed";
import { DrowsyGuardSocket } from "../services/websocket";

const INITIAL_PREDICTION = {
  label: "Awake",
  confidence_pct: 0,
  attention_score: 100,
  blink_rate_per_min: 0,
  ear: null,
  eye_probabilities: {
    Open: 0,
    Closed: 0,
    Half: 0
  },
  temporal_probabilities: {
    Awake: 0,
    Drowsy: 0,
    Distracted: 0,
    Microsleep: 0
  },
  fatigue_trend: [],
  alarm: false
};

function FatigueTrendGraph({ values }) {
  const width = 420;
  const height = 130;
  const clipped = (values ?? []).slice(-120);

  if (clipped.length < 2) {
    return (
      <div className="grid h-[130px] place-content-center rounded-xl border border-cyan-200/20 bg-slate-950/50 text-sm text-cyan-100/70">
        Waiting for live fatigue trend data
      </div>
    );
  }

  const points = clipped
    .map((value, index) => {
      const x = (index / (clipped.length - 1)) * width;
      const normalized = Math.max(0, Math.min(100, Number(value) || 0));
      const y = height - (normalized / 100) * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="rounded-xl border border-cyan-200/20 bg-slate-950/50 p-2">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-[130px] w-full">
        <defs>
          <linearGradient id="fatigueGradient" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#5ca1ff" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#ff8f43" stopOpacity="0.95" />
          </linearGradient>
        </defs>
        <polyline fill="none" stroke="url(#fatigueGradient)" strokeWidth="3" points={points} />
      </svg>
    </div>
  );
}

export default function HomePage() {
  const [prediction, setPrediction] = useState(INITIAL_PREDICTION);
  const [connection, setConnection] = useState("connecting");
  const [sessionId, setSessionId] = useState("");
  const [socketClient, setSocketClient] = useState(null);

  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/live-detection";

  useEffect(() => {
    const client = new DrowsyGuardSocket(wsUrl, {
      maxFps: 10,
      onOpen: () => setConnection("connected"),
      onClose: () => setConnection("disconnected"),
      onError: () => setConnection("error"),
      onMessage: (payload) => {
        if (payload?.type === "session") {
          setSessionId(payload.session_id || "");
          return;
        }

        if (payload?.label) {
          setPrediction((current) => ({
            ...current,
            ...payload
          }));
        }
      }
    });

    client.connect();
    setSocketClient(client);

    return () => {
      client.close();
    };
  }, [wsUrl]);

  const handleFrame = useCallback(
    (imagePayload) => {
      if (!socketClient) {
        return;
      }
      socketClient.sendFrame(imagePayload);
    },
    [socketClient]
  );

  const connectionBadge = useMemo(() => {
    if (connection === "connected") {
      return "bg-emerald-500/20 text-emerald-200 border-emerald-300/40";
    }
    if (connection === "error") {
      return "bg-rose-500/20 text-rose-200 border-rose-300/40";
    }
    return "bg-amber-500/20 text-amber-100 border-amber-300/40";
  }, [connection]);

  const temporalProbabilities = prediction.temporal_probabilities || {};
  const eyeProbabilities = prediction.eye_probabilities || {};

  return (
    <main className="grid-background min-h-screen px-4 py-6 md:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.header
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          className="panel shadow-neon flex flex-col gap-4 rounded-2xl border p-5 md:flex-row md:items-center md:justify-between"
        >
          <div>
            <p className="font-display text-2xl tracking-wide md:text-3xl">DrowsyGuard AI</p>
            <p className="text-muted mt-1 text-sm md:text-base">
              Real-time two-stage drowsiness intelligence pipeline
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.2em]">
            <span className={`rounded-full border px-3 py-1 ${connectionBadge}`}>
              {connection}
            </span>
            {sessionId ? (
              <span className="rounded-full border border-cyan-200/30 bg-cyan-500/10 px-3 py-1 text-cyan-100">
                Session {sessionId.slice(0, 8)}
              </span>
            ) : null}
          </div>
        </motion.header>

        <div className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <motion.section initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}>
            <WebcamFeed
              onFrame={handleFrame}
              prediction={prediction}
              connected={connection === "connected"}
            />
          </motion.section>

          <motion.section
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-4"
          >
            <StatusIndicator
              label={prediction.label}
              attentionScore={prediction.attention_score}
            />

            <div className="panel space-y-4 rounded-2xl border p-4">
              <p className="font-display text-sm uppercase tracking-[0.22em] text-cyan-100/80">
                Temporal Confidence
              </p>
              <ConfidenceBar label="Overall" value={prediction.confidence_pct || 0} />
              {Object.entries(temporalProbabilities).map(([key, value]) => (
                <ConfidenceBar
                  key={key}
                  label={key}
                  value={(Number(value) || 0) * 100}
                  gradient="from-amber-300 via-orange-400 to-rose-500"
                />
              ))}
            </div>

            <div className="panel space-y-4 rounded-2xl border p-4">
              <p className="font-display text-sm uppercase tracking-[0.22em] text-cyan-100/80">
                Eye-State Probabilities
              </p>
              {Object.entries(eyeProbabilities).map(([key, value]) => (
                <ConfidenceBar
                  key={key}
                  label={key}
                  value={(Number(value) || 0) * 100}
                  gradient="from-cyan-300 via-sky-400 to-indigo-500"
                />
              ))}

              <div className="grid grid-cols-2 gap-3 pt-2 text-sm text-cyan-100/85">
                <div className="rounded-lg border border-cyan-200/20 bg-slate-900/60 p-2">
                  EAR: {prediction.ear == null ? "N/A" : prediction.ear}
                </div>
                <div className="rounded-lg border border-cyan-200/20 bg-slate-900/60 p-2">
                  Blink Rate: {prediction.blink_rate_per_min ?? 0}/min
                </div>
              </div>
            </div>

            <AlertSystem alarm={Boolean(prediction.alarm)} label={prediction.label} enableVoice />
          </motion.section>
        </div>

        <motion.section
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="panel rounded-2xl border p-4"
        >
          <div className="mb-3 flex items-center justify-between">
            <p className="font-display text-sm uppercase tracking-[0.24em] text-cyan-100/80">
              Fatigue Trend (Last 60 Seconds)
            </p>
            <span className="rounded-full border border-orange-300/35 bg-orange-500/10 px-3 py-1 text-xs text-orange-100">
              Current Fatigue: {Math.round(100 - (prediction.attention_score || 0))}
            </span>
          </div>
          <FatigueTrendGraph values={prediction.fatigue_trend || []} />
        </motion.section>
      </div>
    </main>
  );
}
