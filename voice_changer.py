"""Local RVC microphone -> virtual cable. No server, upload or recording."""
from __future__ import annotations
import os
import sys
import json
import time
import queue
import threading
import traceback
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import deque

BASE = Path(__file__).resolve().parent
def runtime_config():
    path = BASE / 'runtime.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text('utf-8-sig'))


runtime = runtime_config()
engine_path = Path(os.environ.get('RVC_HOME') or runtime.get('engine_dir', 'engine/Applio')).expanduser()
REPO = (BASE / engine_path).resolve() if not engine_path.is_absolute() else engine_path.resolve()
CONFIG = BASE / 'settings.json'
SR = 48000


def models():
    directory = BASE / 'models'
    indexes = list(directory.glob('*.index'))
    found = {}
    for model in sorted(directory.glob('*.pth')):
        if model.stem.startswith(('G_', 'D_', 'f0G', 'f0D')):
            continue
        index = model.with_suffix('.index')
        if not index.is_file() and len(indexes) == 1:
            index = indexes[0]
        found[model.stem] = (str(model), str(index) if index.is_file() else '')
    return found


class PlaybackBuffer:
    """Sample FIFO, bounded even if capture/playback clocks drift."""
    def __init__(self, max_samples, start_samples=0):
        self.parts = deque()
        self.size = 0
        self.limit = max_samples
        self.lock = threading.Lock()
        self.dropped = 0
        self.start_samples = start_samples
        self.ready = False

    def put(self, samples):
        with self.lock:
            samples = samples[-self.limit:]
            while self.parts and self.size + len(samples) > self.limit:
                self.size -= len(self.parts.popleft())
                self.dropped += 1
            self.parts.append(samples.copy())
            self.size += len(samples)

    def fill(self, out):
        out.fill(0)
        pos = 0
        with self.lock:
            if not self.ready:
                if self.size < self.start_samples:
                    return 0
                self.ready = True
            while self.parts and pos < len(out):
                part = self.parts[0]
                n = min(len(part), len(out) - pos)
                out[pos:pos+n, :] = part[:n, None]
                pos += n
                self.size -= n
                if n == len(part):
                    self.parts.popleft()
                else:
                    self.parts[0] = part[n:]
            if pos < len(out):
                self.ready = False
        return pos


def load_engine(model, index, block, threshold):
    if not (REPO / 'rvc/realtime/core.py').is_file():
        raise FileNotFoundError('请先在 runtime.json 的 engine_dir 中配置兼容的 Applio 引擎目录。')
    # The existing engine resolves its installed assets relative to cwd.
    required = [REPO / 'rvc/models/embedders/contentvec/pytorch_model.bin',
                REPO / 'rvc/models/embedders/contentvec/config.json',
                REPO / 'rvc/models/predictors/rmvpe.pt', Path(model)]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'缺少本地模型文件：{path}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['APPLIO_DISABLE_CUDA_GRAPHS'] = '1'
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import torch
    torch.set_num_threads(4)
    from rvc.realtime.core import VoiceChanger
    return VoiceChanger(block_frame=block, cross_fade_overlap_size=.04,
                        extra_convert_size=.5, model_path=model, index_path=index,
                        f0_method='rmvpe', embedder_model='contentvec',
                        silent_threshold=threshold, vad_enabled=False,
                        device='cuda:0' if torch.cuda.is_available() else 'cpu')


class AudioSession:
    def __init__(self, config, events):
        self.config = config
        self.events = events
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.stats = {'level': 0., 'ms': 0., 'dropped': 0, 'underruns': 0, 'processed': 0}
        self.failure = None

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def run(self):
        import numpy as np
        import sounddevice as sd
        cfg = self.config
        block = cfg['block_ms'] * 48
        incoming = queue.Queue(maxsize=2)
        playback = PlaybackBuffer(block * 3, start_samples=block * 2)
        streams = []
        engine = None
        last_input = time.monotonic()
        peak_input = 0.0

        def capture(data, frames, timing, status):
            nonlocal last_input, peak_input
            if self.stop_event.is_set():
                return
            last_input = time.monotonic()
            if status:
                self.stats['dropped'] += 1
            mono = data.mean(axis=1).copy()
            peak_input = max(peak_input, float(np.max(np.abs(mono))))
            try:
                incoming.put_nowait(mono)
            except queue.Full:
                try:
                    incoming.get_nowait()
                except queue.Empty:
                    pass
                try:
                    incoming.put_nowait(mono)
                except queue.Full:
                    pass
                self.stats['dropped'] += 1

        def output(data, frames, timing, status):
            if self.stop_event.is_set():
                data.fill(0)
                return
            if playback.fill(data) < frames:
                self.stats['underruns'] += 1

        try:
            self.events.put(('status', '正在加载音色并预热计算设备…'))
            engine = load_engine(cfg['model'], cfg['index'] if cfg['index_rate'] else '',
                                 block, cfg['threshold'])
            if self.stop_event.is_set():
                return
            warmup_ms = engine.warmup(index_rate=cfg['index_rate'])
            if self.stop_event.is_set():
                return
            # Refresh only before any stream opens. Device indices can change
            # during a slow first model load or after USB reconnection.
            sd._terminate()
            sd._initialize()
            current_devices = sd.query_devices()
            current_apis = sd.query_hostapis()
            for kind in ('input', 'output'):
                expected = cfg.get(kind + '_name')
                if expected:
                    matches = [d['index'] for d in current_devices
                               if f"{d['name']}  [{current_apis[d['hostapi']]['name']}]" == expected
                               and d['max_' + kind + '_channels'] > 0]
                    if len(matches) != 1:
                        raise RuntimeError(f'所选设备已断开或存在重名：{expected}。请刷新设备后重试。')
                    cfg[kind + '_id'] = matches[0]
            logging.info('Opening input=%s output=%s', sd.query_devices(cfg['input_id']), sd.query_devices(cfg['output_id']))
            sd.check_input_settings(device=cfg['input_id'], samplerate=SR, channels=1, dtype='float32')
            channels = min(2, sd.query_devices(cfg['output_id'])['max_output_channels'])
            sd.check_output_settings(device=cfg['output_id'], samplerate=SR, channels=channels, dtype='float32')
            output_stream = sd.OutputStream(device=cfg['output_id'], samplerate=SR,
                                           channels=channels, dtype='float32', blocksize=0,
                                           latency='low', callback=output)
            streams.append(output_stream)
            input_stream = sd.InputStream(device=cfg['input_id'], samplerate=SR,
                                         channels=1, dtype='float32', blocksize=block,
                                         latency='low', callback=capture)
            streams.append(input_stream)
            last_input = time.monotonic()
            output_stream.start()
            input_stream.start()
            self.events.put(('running', f'变声中 · {engine.device} · 预热耗时 {warmup_ms:.0f} ms/块'))
            while not self.stop_event.is_set():
                try:
                    audio = incoming.get(timeout=.15)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    if time.monotonic() - last_input > 3 or not input_stream.active or not output_stream.active:
                        raise RuntimeError('音频设备已断开或停止响应，请重新选择设备。')
                    continue
                converted, volume, perf = engine.on_request(
                    audio, f0_up_key=cfg['pitch'], index_rate=cfg['index_rate'],
                    protect=.33, volume_envelope=.8)
                if not np.isfinite(converted).all():
                    raise RuntimeError('模型输出异常，请停止后重新加载。')
                converted = np.clip(converted * cfg['gain'], -.98, .98).astype('float32')
                if self.stop_event.is_set():
                    break
                playback.put(converted)
                self.stats.update(level=volume, ms=perf[1], processed=self.stats['processed']+1,
                                  buffer_drops=playback.dropped, peak_input=peak_input)
        except Exception as exc:
            self.failure = str(exc)
            logging.exception('Audio session failed')
            self.events.put(('error', str(exc)))
        finally:
            self.stop_event.set()
            for stream in reversed(streams):
                try:
                    stream.abort()
                    stream.close()
                except Exception:
                    logging.exception('Closing audio stream')
            del engine
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.events.put(('stopped', None))


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.events = queue.Queue()
        self.session = None
        self.closing = False
        self.saved = {}
        try:
            self.saved = json.loads(CONFIG.read_text('utf-8'))
        except (OSError, ValueError):
            pass
        self.model_map = models()
        self.inputs, self.outputs = {}, {}
        root.title('本地 AI 变声器 · v0.1.0-beta.1')
        root.geometry('820x720')
        root.minsize(760, 690)
        root.configure(bg='#f3f5fa')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('Microsoft YaHei UI', 10))
        style.configure('TFrame', background='#f3f5fa')
        style.configure('TLabel', background='#f3f5fa', foreground='#243148')
        style.configure('Title.TLabel', font=('Microsoft YaHei UI', 22, 'bold'))
        style.configure('Hint.TLabel', foreground='#63718a')
        style.configure('TButton', padding=(14, 9))
        style.configure('Go.TButton', background='#2563eb', foreground='white')
        body = ttk.Frame(root, padding=26)
        body.pack(fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        ttk.Label(body, text='本地 AI 变声器', style='Title.TLabel').grid(row=0, column=0, columnspan=3, sticky='w')
        ttk.Label(body, text='麦克风 → RVC 音色转换 → 虚拟声卡 → 游戏 / 聊天软件',
                  style='Hint.TLabel').grid(row=1, column=0, columnspan=3, sticky='w', pady=(6, 24))
        self.controls = []
        self.model = tk.StringVar()
        self.input = tk.StringVar()
        self.output = tk.StringVar()
        for row, label, var in [(2, '音色模型', self.model), (3, '输入麦克风', self.input), (4, '变声输出', self.output)]:
            ttk.Label(body, text=label).grid(row=row, column=0, sticky='w', padx=(0, 18), pady=8)
            combo = ttk.Combobox(body, textvariable=var, state='readonly', width=51)
            combo.grid(row=row, column=1, columnspan=2, sticky='ew', pady=8)
            setattr(self, ['model_box','input_box','output_box'][row-2], combo)
            self.controls.append(combo)
        self.model_box['values'] = list(self.model_map)
        name = self.saved.get('model_name')
        self.model.set(name if name in self.model_map else next(iter(self.model_map), '未找到本地音色模型'))
        self.cable = tk.StringVar(value='正在检查音频设备…')
        ttk.Label(body, textvariable=self.cable, style='Hint.TLabel', wraplength=720).grid(row=5, column=0, columnspan=3, sticky='w', pady=(2, 12))
        options = ttk.Frame(body)
        options.grid(row=6, column=0, columnspan=3, sticky='ew', pady=8)
        self.pitch = tk.IntVar(value=self.saved.get('pitch', 0))
        self.block = tk.StringVar(value=str(self.saved.get('block_ms', 200)))
        self.retrieval = tk.StringVar(value=str(self.saved.get('index_rate', 0.0)))
        self.gain = tk.StringVar(value=str(self.saved.get('gain', 1.0)))
        self.threshold = tk.StringVar(value=str(self.saved.get('threshold', -60)))
        for col, (label, variable, values) in enumerate([
            ('音调 / 半音', self.pitch, list(range(-12, 13))),
            ('缓冲 / ms', self.block, [100, 150, 200, 300, 400]),
            ('音色检索', self.retrieval, [0.0, 0.1, 0.25, 0.5]),
            ('输出音量', self.gain, [0.5, 0.75, 1.0, 1.25, 1.5]),
            ('静音门限 / dB', self.threshold, [-70, -60, -55, -50, -40]),
        ]):
            options.columnconfigure(col, weight=1)
            ttk.Label(options, text=label).grid(row=0, column=col, sticky='w', pady=6)
            box = ttk.Combobox(options, textvariable=variable, values=values, width=11, state='readonly')
            box.grid(row=1, column=col, sticky='ew', padx=(0, 10))
            self.controls.append(box)
        ttk.Separator(body).grid(row=7, column=0, columnspan=3, sticky='ew', pady=20)
        self.status = tk.StringVar(value='待机 · 点击「开始变声」后才启用麦克风')
        ttk.Label(body, textvariable=self.status, wraplength=720).grid(row=8, column=0, columnspan=3, sticky='w')
        self.meter = ttk.Progressbar(body, maximum=100)
        self.meter.grid(row=9, column=0, columnspan=3, sticky='ew', pady=(12, 6))
        self.metrics = tk.StringVar(value='输入电平 —    推理耗时 —')
        ttk.Label(body, textvariable=self.metrics, style='Hint.TLabel').grid(row=10, column=0, columnspan=3, sticky='w')
        buttons = ttk.Frame(body)
        buttons.grid(row=11, column=0, columnspan=3, sticky='ew', pady=20)
        self.start_button = ttk.Button(buttons, text='开始变声', style='Go.TButton', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(buttons, text='停止', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=10)
        self.refresh_button = ttk.Button(buttons, text='刷新设备', command=self.refresh)
        self.refresh_button.pack(side='right')
        ttk.Label(body, text='接入方法：本程序输出选 CABLE Input；游戏、QQ、微信等的麦克风选 CABLE Output。\n调节音色、设备或缓冲后，停止并重新开始即可生效。试听时请佩戴耳机。',
                  style='Hint.TLabel', wraplength=720, justify='left').grid(row=12, column=0, columnspan=3, sticky='w', pady=8)
        links = ttk.Frame(body)
        links.grid(row=13, column=0, columnspan=3, sticky='w', pady=10)
        ttk.Button(links, text='安装虚拟声卡', command=self.install_driver).pack(side='left')
        ttk.Button(links, text='使用说明', command=lambda: os.startfile(str(BASE/'使用说明.md'))).pack(side='left', padx=10)
        ttk.Button(links, text='打开日志', command=lambda: os.startfile(str(BASE/'voice-changer.log'))).pack(side='left')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.refresh)
        root.after(150, self.poll)

    def refresh(self):
        if self.session:
            return
        import sounddevice as sd
        try:
            sd._terminate()
            sd._initialize()
            apis = sd.query_hostapis()
            devices = sd.query_devices()
            self.inputs, self.outputs = {}, {}
            for device in devices:
                api = apis[device['hostapi']]['name']
                if api not in ('Windows WASAPI', 'MME'):
                    continue
                label = f"{device['name']}  [{api}]"
                if device['max_input_channels']:
                    self.inputs[label] = device['index']
                if device['max_output_channels']:
                    self.outputs[label] = device['index']
            self.input_box['values'] = list(self.inputs)
            self.output_box['values'] = list(self.outputs)
            old_input = self.input.get() or self.saved.get('input_name')
            old_output = self.output.get() or self.saved.get('output_name')
            def best(mapping, keyword):
                return next((n for n in mapping if keyword.lower() in n.lower() and 'WASAPI' in n), '')
            physical_input = best(self.inputs, 'USB') or next((n for n in self.inputs if 'WASAPI' in n and 'cable' not in n.lower()), '')
            self.input.set(old_input if old_input in self.inputs else physical_input)
            cable = best(self.outputs, 'CABLE Input')
            self.output.set(old_output if old_output in self.outputs else cable)
            self.cable.set('已检测到 VB-CABLE · 聊天软件的麦克风请选择 CABLE Output' if cable else
                           '尚未检测到 VB-CABLE。请先安装虚拟声卡；安装后若未出现设备，请重启电脑。')
        except Exception as exc:
            self.status.set(f'无法读取音频设备：{exc}')
            logging.exception('Enumerating devices')

    def start(self):
        from tkinter import messagebox
        if self.session:
            return
        if self.model.get() not in self.model_map or self.input.get() not in self.inputs or self.output.get() not in self.outputs:
            messagebox.showinfo('选择设备', '请选择有效的音色、输入麦克风和变声输出。未安装虚拟声卡时，可手动选择耳机试听。')
            return
        if 'cable' not in self.output.get().lower():
            if not messagebox.askokcancel('耳机试听', '输出将播放到所选设备。请佩戴耳机，避免扬声器回授啸叫。继续试听？'):
                return
        model, index = self.model_map[self.model.get()]
        cfg = dict(model=model, index=index, input_id=self.inputs[self.input.get()], output_id=self.outputs[self.output.get()],
                   input_name=self.input.get(), output_name=self.output.get(),
                   pitch=int(self.pitch.get()), block_ms=int(self.block.get()), index_rate=float(self.retrieval.get()),
                   gain=float(self.gain.get()), threshold=int(self.threshold.get()))
        saved = {k: v for k, v in cfg.items() if k not in ('input_id', 'output_id', 'model', 'index')}
        saved.update(model_name=self.model.get(), input_name=self.input.get(), output_name=self.output.get())
        try:
            temp = CONFIG.with_suffix('.tmp')
            temp.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding='utf-8')
            temp.replace(CONFIG)
        except OSError:
            logging.exception('Saving settings')
        for widget in self.controls:
            widget.configure(state='disabled')
        self.start_button.configure(state='disabled')
        self.refresh_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.session = AudioSession(cfg, self.events)
        self.session.start()

    def stop(self):
        if self.session:
            self.session.stop()
            self.status.set('正在停止并释放设备…')
            self.stop_button.configure(state='disabled')

    def poll(self):
        import math
        from tkinter import messagebox
        while True:
            try:
                event, value = self.events.get_nowait()
            except queue.Empty:
                break
            if event in ('status', 'running'):
                if not self.closing:
                    self.status.set(value)
            elif event == 'error':
                self.status.set('启动或运行失败 · 查看日志了解详情')
                if not self.closing:
                    messagebox.showerror('变声器', value)
            elif event == 'stopped':
                failed = self.session and self.session.failure
                self.session = None
                if self.closing:
                    self.root.destroy()
                    return
                for widget in self.controls:
                    widget.configure(state='readonly')
                self.start_button.configure(state='normal')
                self.refresh_button.configure(state='normal')
                self.stop_button.configure(state='disabled')
                if not failed:
                    self.status.set('已停止 · 麦克风已释放')
                self.meter['value'] = 0
        if self.session:
            stats = self.session.stats.copy()
            db = 20 * math.log10(max(stats['level'], 1e-6))
            self.meter['value'] = max(0, min(100, (db + 60) / 60 * 100))
            slow = ' · 推理过慢，请增加缓冲' if stats['ms'] > self.session.config['block_ms'] else ''
            self.metrics.set(f"输入 {db:.0f} dB    推理 {stats['ms']:.0f} ms/块    丢块 {stats['dropped'] + stats.get('buffer_drops',0)}{slow}")
        self.root.after(150, self.poll)

    def install_driver(self):
        from tkinter import messagebox
        if self.session:
            messagebox.showinfo('先停止变声', '请停止变声后再安装驱动。')
            return
        script = BASE / '安装虚拟声卡.ps1'
        import subprocess
        subprocess.Popen(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)],
                         creationflags=subprocess.CREATE_NO_WINDOW)

    def close(self):
        self.closing = True
        if self.session:
            self.stop()
        else:
            self.root.destroy()


def main():
    handler = RotatingFileHandler(BASE/'voice-changer.log', maxBytes=2_000_000, backupCount=2, encoding='utf-8')
    logging.basicConfig(handlers=[handler], level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # pythonw has no console, but installed engine prints diagnostics.
    log_stream = open(BASE/'engine.log', 'w', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = log_stream
    import tkinter as tk
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
