import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal
import threading
import queue
import gc

class RecordingManager(QObject):
    status_update = pyqtSignal(str)
    saving_started = pyqtSignal(int)
    saving_progress = pyqtSignal(int)
    saving_finished = pyqtSignal()

    def __init__(self, project_root):
        super().__init__()
        self.project_root = project_root
        self.is_recording = False
        self.rec_dir = None
        self.rec_out_color = None
        self.rec_file_pen = None
        self.rec_frame_idx = 0
        self.rec_start_time = 0
        self._lock = threading.Lock()
        self._last_debug_emit_ts = 0.0
        self._write_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=3600)
        self._writer_thread: threading.Thread = None
        self._writer_stop = threading.Event()
        self._saving_mode = False
        self._saving_done = 0
        self._bundle_total = 0
        self._R_w = None
        self._origin_w = None
        
    def start(self, cam, desk_plane_depth):
        with self._lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.rec_dir = os.path.join(self.project_root, "recordings", f"rec_{timestamp}")
            print(self.rec_dir)
            os.makedirs(self.rec_dir, exist_ok=True)
            os.makedirs(os.path.join(self.rec_dir, "color"), exist_ok=True)
            os.makedirs(os.path.join(self.rec_dir, "ir"), exist_ok=True)
            meta = cam.get_calibration_data()
            meta["desk_plane"] = desk_plane_depth.tolist() if desk_plane_depth is not None else None
            with open(os.path.join(self.rec_dir, "meta.json"), 'w') as f:
                json.dump(meta, f, indent=2)
            self.rec_out_color = None
            self.rec_file_pen = open(os.path.join(self.rec_dir, "pen_data.jsonl"), 'w')
            self.rec_frame_idx = 0
            now = time.time()
            self.rec_start_time = now
            self._last_debug_emit_ts = now
            self._writer_stop.clear()
            self._saving_mode = False
            self._saving_done = 0
            self._bundle_total = 0
            self.is_recording = True
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
            self.status_update.emit(f"Recording started: {os.path.basename(self.rec_dir)}")

    def update_world_transform(self, R_w, origin_w):
        with self._lock:
            self._R_w = R_w
            self._origin_w = origin_w

    def stop(self):
        with self._lock:
            if not self.is_recording:
                return
            self.is_recording = False
            total = self._write_queue.qsize()
            self._saving_mode = True
            self._saving_done = 0
            self.saving_started.emit(total)
            self._writer_stop.set()
            threading.Thread(target=self._wait_for_writer, daemon=True).start()

    def record_frame(self, frame, tracker_result):
        with self._lock:
            if not self.is_recording:
                return
            c_img = None
            ir_img = None
            if frame.color is not None:
                c_img = frame.color
            if frame.ir is not None:
                ir_img = frame.ir
            pen_entry = {
                "frame_id": self.rec_frame_idx,
                "timestamp": frame.timestamp,
                "tip_pos_cam": tracker_result.tip_pos_cam.tolist() if tracker_result.tip_pos_cam is not None else None,
                "tail_pos_cam": tracker_result.tail_pos_cam.tolist() if getattr(tracker_result, "tail_pos_cam", None) is not None else None,
                "direction": tracker_result.direction.tolist() if tracker_result.direction is not None else None,
                "tip_pos_world": None,
                "tail_pos_world": None
            }
            if self._R_w is not None and self._origin_w is not None and tracker_result.tip_pos_cam is not None and getattr(tracker_result, "tail_pos_cam", None) is not None:
                tip_w = self._R_w @ (tracker_result.tip_pos_cam - self._origin_w)
                tail_w = self._R_w @ (tracker_result.tail_pos_cam - self._origin_w)
                pen_entry["tip_pos_world"] = tip_w.tolist()
                pen_entry["tail_pos_world"] = tail_w.tolist()
            if not self._write_queue.full():
                self._write_queue.put(("bundle", int(self.rec_frame_idx), pen_entry, c_img, ir_img))
                self.rec_frame_idx += 1
                self._bundle_total += 1
            now = time.time()
            if now - self._last_debug_emit_ts >= 1.0 and self.rec_start_time > 0:
                elapsed = now - self.rec_start_time
                if elapsed > 0:
                    fps = self.rec_frame_idx / elapsed
                    self.status_update.emit(f"Recording progress: frames={self.rec_frame_idx}, fps={fps:.2f}")
                self._last_debug_emit_ts = now

    def _writer_loop(self):
        color_dir = os.path.join(self.rec_dir, "color")
        ir_dir = os.path.join(self.rec_dir, "ir")
        while True:
            if self._writer_stop.is_set():
                while not self._write_queue.empty():
                    task = self._write_queue.get_nowait()
                    kind = task[0]
                    if kind == "bundle":
                        idx = int(task[1])
                        pen_entry = task[2]
                        c_img = task[3]
                        ir_img = task[4]
                        if c_img is not None:
                            if len(c_img.shape) == 3 and c_img.shape[2] == 4:
                                c_img = cv2.cvtColor(c_img, cv2.COLOR_BGRA2BGR)
                            c_img = np.ascontiguousarray(c_img)
                            cv2.imwrite(os.path.join(color_dir, f"{idx:06d}.png"), c_img)
                            del c_img
                            pass
                        if ir_img is not None:
                            ir_img = np.ascontiguousarray(ir_img)
                            cv2.imwrite(os.path.join(ir_dir, f"{idx:06d}.png"), ir_img)
                            del ir_img
                            pass
                        if self.rec_file_pen is None:
                            self.rec_file_pen = open(os.path.join(self.rec_dir, "pen_data.jsonl"), 'a')
                        self.rec_file_pen.write(json.dumps(pen_entry) + "\n")
                        if self._saving_mode:
                            self._saving_done += 1
                            self.saving_progress.emit(self._saving_done)
                if self._write_queue.empty():
                    break
            try:
                #task = self._write_queue.get(timeout=0.1)
                sleep(0.1)
                continue
            except queue.Empty:
                continue
            kind = task[0]
            if kind == "bundle":
                idx = int(task[1])
                pen_entry = task[2]
                c_img = task[3]
                ir_img = task[4]
                if c_img is not None:
                    if len(c_img.shape) == 3 and c_img.shape[2] == 4:
                        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGRA2BGR)
                    c_img = np.ascontiguousarray(c_img)
                    cv2.imwrite(os.path.join(color_dir, f"{idx:06d}.png"), c_img)
                    del c_img
                if ir_img is not None:
                    ir_img = np.ascontiguousarray(ir_img)
                    cv2.imwrite(os.path.join(ir_dir, f"{idx:06d}.png"), ir_img)
                    del ir_img
                if self.rec_file_pen is None:
                    self.rec_file_pen = open(os.path.join(self.rec_dir, "pen_data.jsonl"), 'a')
                self.rec_file_pen.write(json.dumps(pen_entry) + "\n")
                if self._saving_mode:
                    self._saving_done += 1
                    self.saving_progress.emit(self._saving_done)
        if self.rec_file_pen:
            self.rec_file_pen.close()
            self.rec_file_pen = None
        gc.collect()

    def _wait_for_writer(self):
        if self._writer_thread is not None:
            self._writer_thread.join()
        self.saving_finished.emit()
        self.status_update.emit(f"Recording saved: {os.path.basename(self.rec_dir)}")
        self.rec_dir = None
