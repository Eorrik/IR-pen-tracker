import cv2
import numpy as np
from collections import deque

class OrthoVisualizer:
    def __init__(self, width=400, height=400, x_range=(-0.2, 0.2), y_range=(-0.05, 0.15), z_range=(-0.2, 0.2)):
        self.w = width
        self.h = height
        # World Frame Ranges
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        self.canvas_top = np.zeros((height//2, width, 3), dtype=np.uint8)
        self.canvas_side = np.zeros((height//2, width, 3), dtype=np.uint8)
        
        self.history = deque(maxlen=100) # Store (x, y, z) in World Frame
        
        # Pen State
        self.current_tip = None
        self.current_tail = None
        self.current_mark1 = None
        self.current_mark2 = None
        self.current_hover_height = None

    def update(self, x, y, z, tail=None, hover_height=None, mark1=None, mark2=None):
        self.history.append((x, y, z))
        self.current_tip = (x, y, z)
        self.current_tail = tail
        self.current_hover_height = hover_height
        if mark1 is not None:
            self.current_mark1 = mark1
        if mark2 is not None:
            self.current_mark2 = mark2

    def _draw_grid_and_points(self, canvas, h_val_getter, v_val_getter, h_range, v_range, label, extra_draw_func=None, anchor_bottom=False):
        """
        Generic drawer that maintains aspect ratio.
        h_range: (min, max) for horizontal axis
        v_range: (min, max) for vertical axis
        """
        cw, ch = canvas.shape[1], canvas.shape[0]
        canvas.fill(30)
        
        # Calculate isotropic scale
        h_len = h_range[1] - h_range[0]
        v_len = v_range[1] - v_range[0]
        
        if h_len <= 0 or v_len <= 0:
            return

        # Determine scale (pixels per meter) to fit both dimensions
        scale_h = cw / h_len
        scale_v = ch / v_len
        scale = min(scale_h, scale_v) * 0.9 # 90% fill
        
        # Center offsets
        cx = cw / 2
        cy = ch / 2
        
        h_mid = (h_range[0] + h_range[1]) / 2
        v_mid = (v_range[0] + v_range[1]) / 2
        
        def to_uv(h_val, v_val):
            u = int(cx + (h_val - h_mid) * scale)
            if anchor_bottom:
                v = int(ch - (v_val - v_range[0]) * scale)
            else:
                v = int(cy - (v_val - v_mid) * scale)
            return u, v
            
        # Draw Grid (0.1m steps)
        # Vertical lines (Horizontal axis)
        start_h = np.floor(h_range[0] * 10) / 10.0
        for x in np.arange(start_h, h_range[1], 0.1):
            u, _ = to_uv(x, v_mid)
            if 0 <= u < cw:
                cv2.line(canvas, (u, 0), (u, ch), (50, 50, 50), 1)
                # Label axis if it's 0 or 0.5 or 1.0
                if abs(x % 0.5) < 1e-5 or abs(x) < 1e-5:
                     cv2.putText(canvas, f"{x:.1f}", (u+2, ch-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Horizontal lines (Vertical axis)
        start_v = np.floor(v_range[0] * 10) / 10.0
        for y in np.arange(start_v, v_range[1], 0.1):
            _, v = to_uv(h_mid, y)
            if 0 <= v < ch:
                cv2.line(canvas, (0, v), (cw, v), (50, 50, 50), 1)
                if abs(y % 0.5) < 1e-5 or abs(y) < 1e-5:
                     cv2.putText(canvas, f"{y:.1f}", (5, v-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Draw Points (Trace)
        for i, p3d in enumerate(self.history):
            h_val = h_val_getter(p3d)
            v_val = v_val_getter(p3d)
            
            u, v = to_uv(h_val, v_val)
            
            if 0 <= u < cw and 0 <= v < ch:
                # Color by Z (always use Z for color to show depth)
                z_val = p3d[2]
                norm = np.clip((z_val - self.z_range[0]) / (self.z_range[1] - self.z_range[0]), 0, 1)
                hue = int((1.0 - norm) * 120) 
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                c = (int(color[0]), int(color[1]), int(color[2]))
                
                radius = 2 if i < len(self.history)-1 else 4
                cv2.circle(canvas, (u, v), radius, c, -1)

        # Draw Current Pen Body
        if self.current_tip is not None and self.current_tail is not None:
             tip_h = h_val_getter(self.current_tip)
             tip_v = v_val_getter(self.current_tip)
             tail_h = h_val_getter(self.current_tail)
             tail_v = v_val_getter(self.current_tail)
             
             u_tip, v_tip = to_uv(tip_h, tip_v)
             u_tail, v_tail = to_uv(tail_h, tail_v)
             
             cv2.line(canvas, (u_tip, v_tip), (u_tail, v_tail), (255, 255, 0), 2)
             cv2.circle(canvas, (u_tip, v_tip), 5, (255, 0, 255), -1)  # predicted tip

        # Draw current markers (mark1: tip marker, mark2: tail marker)
        if self.current_mark1 is not None:
            m1_h = h_val_getter(self.current_mark1)
            m1_v = v_val_getter(self.current_mark1)
            u1, v1 = to_uv(m1_h, m1_v)
            cv2.circle(canvas, (u1, v1), 5, (0, 0, 255), -1)
            cv2.putText(canvas, "mark1", (u1+6, v1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.current_mark2 is not None:
            m2_h = h_val_getter(self.current_mark2)
            m2_v = v_val_getter(self.current_mark2)
            u2, v2 = to_uv(m2_h, m2_v)
            cv2.circle(canvas, (u2, v2), 5, (255, 0, 0), -1)
            cv2.putText(canvas, "mark2", (u2+6, v2-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # Allow extra drawing (e.g. Desk Plane)
        if extra_draw_func:
            extra_draw_func(canvas, to_uv)

        # Label
        cv2.putText(canvas, f"{label} (Scale: {scale:.1f} px/m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw(self):
        # 1. Top View (World XY)
        # H: X, V: Y
        self._draw_grid_and_points(
            self.canvas_top,
            lambda p: p[0], # X
            lambda p: p[1], # Y
            self.x_range,
            self.y_range,
            "Top View (World XY)"
        )
        y0 = 40
        dy = 22
        if self.current_mark1 is not None:
            x, y, z = self.current_mark1
            cv2.putText(self.canvas_top, f"mark1: x={x:.3f} y={y:.3f} z={z:.3f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y0 += dy
        if self.current_mark2 is not None:
            x, y, z = self.current_mark2
            cv2.putText(self.canvas_top, f"mark2: x={x:.3f} y={y:.3f} z={z:.3f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y0 += dy
        if self.current_tip is not None:
            x, y, z = self.current_tip
            cv2.putText(self.canvas_top, f"pred:  x={x:.3f} y={y:.3f} z={z:.3f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 2. Side View (World XZ)
        # H: X, V: Z
        def draw_board_line(canvas, to_uv):
            # Draw line at Z=0 (board plane)
            u1, v1 = to_uv(self.x_range[0], 0.0)
            u2, v2 = to_uv(self.x_range[1], 0.0)
            cv2.line(canvas, (u1, v1), (u2, v2), (0, 0, 255), 2)
            cv2.putText(canvas, "Board (Z=0)", (u1+10, v1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        self._draw_grid_and_points(
            self.canvas_side,
            lambda p: p[0], # X
            lambda p: p[2], # Z
            self.x_range,
            self.z_range,
            "Side View (World XZ)",
            extra_draw_func=draw_board_line
            , anchor_bottom=True
        )
        y1 = 40
        if self.current_mark1 is not None:
            x, y, z = self.current_mark1
            cv2.putText(self.canvas_side, f"mark1: x={x:.3f} y={y:.3f} z={z:.3f}", (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y1 += dy
        if self.current_mark2 is not None:
            x, y, z = self.current_mark2
            cv2.putText(self.canvas_side, f"mark2: x={x:.3f} y={y:.3f} z={z:.3f}", (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y1 += dy
        if self.current_tip is not None:
            x, y, z = self.current_tip
            cv2.putText(self.canvas_side, f"pred:  x={x:.3f} y={y:.3f} z={z:.3f}", (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        return np.vstack((self.canvas_top, self.canvas_side))
