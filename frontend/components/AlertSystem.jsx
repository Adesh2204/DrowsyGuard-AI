import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef } from "react";

function beep() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) {
    return;
  }

  const context = new AudioCtx();
  const oscillator = context.createOscillator();
  const gain = context.createGain();

  oscillator.type = "square";
  oscillator.frequency.value = 880;
  gain.gain.setValueAtTime(0.0001, context.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.4, context.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.45);

  oscillator.connect(gain);
  gain.connect(context.destination);

  oscillator.start();
  oscillator.stop(context.currentTime + 0.45);

  oscillator.onended = () => {
    context.close();
  };
}

export default function AlertSystem({ alarm, label, enableVoice = true }) {
  const lastAlertAt = useRef(0);

  useEffect(() => {
    if (!alarm) {
      return;
    }

    const now = Date.now();
    if (now - lastAlertAt.current < 2500) {
      return;
    }
    lastAlertAt.current = now;

    beep();

    if (enableVoice && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(
        label === "Microsleep" ? "Microsleep detected. Wake up now." : "Drowsiness detected. Stay alert."
      );
      utterance.rate = 1.05;
      utterance.pitch = 0.95;
      window.speechSynthesis.speak(utterance);
    }
  }, [alarm, label, enableVoice]);

  return (
    <AnimatePresence>
      {alarm ? (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          className="rounded-xl border border-rose-400/50 bg-rose-600/20 p-3 text-sm text-rose-100"
        >
          <p className="font-semibold uppercase tracking-[0.18em]">Alert Triggered</p>
          <p className="mt-1">{label} state detected. Immediate driver attention required.</p>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
