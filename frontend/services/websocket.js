const DEFAULT_MAX_FPS = 10;
const BASE_RECONNECT_DELAY_MS = 1200;

export class DrowsyGuardSocket {
  constructor(url, handlers = {}) {
    this.url = url;
    this.ws = null;
    this.sessionId = "";
    this.maxFps = handlers.maxFps ?? DEFAULT_MAX_FPS;
    this.onOpen = handlers.onOpen ?? (() => {});
    this.onClose = handlers.onClose ?? (() => {});
    this.onMessage = handlers.onMessage ?? (() => {});
    this.onError = handlers.onError ?? (() => {});

    this.lastFrameSentAt = 0;
    this.manualClose = false;
    this.reconnectAttempts = 0;
  }

  connect(sessionId = "") {
    this.manualClose = false;
    this.sessionId = sessionId;

    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    this.ws = new WebSocket(`${this.url}${query}`);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onOpen();
    };

    this.ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        this.onMessage(payload);
      } catch {
        this.onMessage({ type: "raw", payload: event.data });
      }
    };

    this.ws.onerror = (event) => {
      this.onError(event);
    };

    this.ws.onclose = () => {
      this.onClose();
      if (!this.manualClose) {
        this._scheduleReconnect();
      }
    };
  }

  _scheduleReconnect() {
    this.reconnectAttempts += 1;
    const delay = Math.min(BASE_RECONNECT_DELAY_MS * this.reconnectAttempts, 6000);
    window.setTimeout(() => {
      if (!this.manualClose) {
        this.connect(this.sessionId);
      }
    }, delay);
  }

  sendFrame(image) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const now = performance.now();
    const minInterval = 1000 / this.maxFps;
    if (now - this.lastFrameSentAt < minInterval) {
      return;
    }

    this.lastFrameSentAt = now;
    this.ws.send(JSON.stringify({ image }));
  }

  close() {
    this.manualClose = true;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
