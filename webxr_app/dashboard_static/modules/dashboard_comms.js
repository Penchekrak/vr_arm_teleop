export class DashboardComms {
  constructor(path = '/ws') {
    const scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    this._url = scheme + location.host + path;
    this._ws = null;
    this._closed = false;
    this._reconnectDelay = 500;
    this._reconnectMax = 5000;
    this.onJson = null;
    this.onBinary = null;
    this.onConnectionState = null;
    this._open();
  }

  _state(state) {
    if (this.onConnectionState) this.onConnectionState(state);
  }

  _open() {
    if (this._closed) return;
    this._state('connecting');
    const ws = new WebSocket(this._url);
    ws.binaryType = 'arraybuffer';

    ws.addEventListener('open', () => {
      this._reconnectDelay = 500;
      this._state('connected');
    });

    ws.addEventListener('message', (ev) => {
      if (typeof ev.data === 'string') {
        if (!this.onJson) return;
        try {
          this.onJson(JSON.parse(ev.data));
        } catch (err) {
          console.error('dashboard: bad JSON', err, ev.data);
        }
      } else if (this.onBinary) {
        this.onBinary(ev.data);
      }
    });

    ws.addEventListener('close', () => {
      this._ws = null;
      if (this._closed) {
        this._state('closed');
        return;
      }
      this._state('disconnected');
      setTimeout(() => this._open(), this._reconnectDelay);
      this._reconnectDelay = Math.min(this._reconnectMax, this._reconnectDelay * 2);
    });

    ws.addEventListener('error', () => {
      this._state('error');
    });

    this._ws = ws;
  }

  close() {
    this._closed = true;
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }

  async postJson(path) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Accept': 'application/json' },
    });
    const text = await res.text();
    let body = {};
    if (text) {
      try {
        body = JSON.parse(text);
      } catch (err) {
        throw new Error(`bad JSON response from ${path}: ${text}`);
      }
    }
    if (!res.ok) {
      throw new Error(body.reason || body.error || `${res.status} ${res.statusText}`);
    }
    return body;
  }
}
