// Bootstrap only: the synthesis worker is Rust compiled to WebAssembly.
import init, { start_worker } from './pkg/inhouse_tts_web.js';
await init();
start_worker();
