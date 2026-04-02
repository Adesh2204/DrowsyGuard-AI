import { motion } from "framer-motion";

function clamp(value) {
  return Math.max(0, Math.min(100, Number(value) || 0));
}

export default function ConfidenceBar({
  label,
  value,
  gradient = "from-cyan-400 via-sky-400 to-blue-500"
}) {
  const safeValue = clamp(value);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-cyan-100/70">
        <span>{label}</span>
        <span>{safeValue.toFixed(1)}%</span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-white/10">
        <motion.div
          className={`h-full rounded-full bg-gradient-to-r ${gradient}`}
          initial={{ width: 0 }}
          animate={{ width: `${safeValue}%` }}
          transition={{ duration: 0.35, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
