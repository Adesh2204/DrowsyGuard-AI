import { motion } from "framer-motion";

const STATUS_CONFIG = {
  Awake: {
    dot: "bg-emerald-400",
    panel: "border-emerald-300/35 bg-emerald-500/10",
    text: "text-emerald-200"
  },
  Drowsy: {
    dot: "bg-amber-400",
    panel: "border-amber-300/35 bg-amber-500/10",
    text: "text-amber-100"
  },
  Distracted: {
    dot: "bg-sky-400",
    panel: "border-sky-300/35 bg-sky-500/10",
    text: "text-sky-100"
  },
  Microsleep: {
    dot: "bg-rose-500",
    panel: "border-rose-400/45 bg-rose-500/15",
    text: "text-rose-100"
  }
};

export default function StatusIndicator({ label = "Awake", attentionScore = 100 }) {
  const config = STATUS_CONFIG[label] ?? STATUS_CONFIG.Awake;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`panel rounded-2xl border p-4 ${config.panel}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className={`h-3 w-3 rounded-full ${config.dot}`} />
          <p className="font-display text-lg">{label}</p>
        </div>
        <span className="text-xs uppercase tracking-[0.24em] text-cyan-100/70">
          Driver State
        </span>
      </div>

      <p className={`mt-3 text-sm ${config.text}`}>
        Attention Score: <strong>{Math.round(attentionScore)}</strong>
      </p>
    </motion.div>
  );
}
