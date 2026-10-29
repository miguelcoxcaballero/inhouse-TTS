// Bootstrap only: the whole application is Rust compiled to WebAssembly.
import init, { start_ui } from './pkg/inhouse_tts_web.js';
await init();
start_ui();
