"""
================================================================================
AVCI DRONE - KAMIKAZE VISION SYSTEM
GNSS Yaklaşma + Kamera Tespit/Takip + Dalış/Çarpışma
================================================================================
MİMARİ:
  FAZA 1 - GNSS YAKLAŞMA   : drone_kamikaze_controller.py mantığı (GNSS ile)
  FAZ  2 - VİZYON GEÇİŞ   : Hedef kamerada onaylanınca devret
  FAZ  3 - VİZYON TAKİP   : YOLO + Kalman + PID ile kilitlen
  FAZ  4 - DALIŞI KILITLE  : Hedef merkezde → tam gaz dalış → ÇARPIŞMA

KURULUM:
  pip install mss pywin32 ultralytics opencv-python numpy --break-system-packages

KULLANIM:
  # Sim penceresini bul:
  python kamikaze_vision.py --list-windows

  # Çalıştır:
  python kamikaze_vision.py --window-name "Unreal" --weights talon_v4.pt --show

PARAMETRELER:
  --window-name   : Sim pencere başlığının bir parçası. "Unreal", "AvciDrone" vb.
  --weights       : Model dosyası (varsayılan: talon_v4.pt)
  --show          : Kamera görüntüsünü ayrı pencerede göster (tespit + KF overlay)
  --conf          : YOLO confidence eşiği (varsayılan 0.35)
  --vision-range  : Kaç cm'de GNSS → Vision geçişi (varsayılan 4000 = 40m)
  --use-obs       : OBS Virtual Camera kullan (ekran yakalama yerine)
  --camera-index  : OBS modunda kamera index
================================================================================
"""

import math
import time
import logging
import threading
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

import cv2
import numpy as np
import drone_sdk as drone
from drone_kamikaze_controller import DroneKamikazeController


# ===========================================================================
# KAPANMA HIZI İZLEYİCİ (closing-speed watchdog)
# ===========================================================================
class ClosingRateMonitor:
    """
    SORUN TEŞHİSİ:
    DroneKamikazeController hedefin HIZINA eşitleniyor (match_target_thr),
    mesafeyi AGRESİF KAPATMIYOR. speed_gap telafisi sadece drone hedeften
    yavaşsa devreye giriyor; drone zaten hedef hızındaysa ekstra itki YOK.
    Sonuç: drone hedefin arkasında sabit mesafede "paralel kovalama" yapıyor,
    mesafe vision_range eşiğine hiç düşmüyor, vizyon fazına asla geçilmiyor.

    ÇÖZÜM:
    Mesafenin gerçekten kapanıp kapanmadığını ölçüyoruz (dD/dt). Kapanma
    hızı yeterli değilse (mesafe sabit/açılıyor) throttle'a EK İTKİ
    enjekte ediyoruz — closing_boost. Bu, GNSS kontrolcüsünün iç mantığını
    bozmadan (DroneKamikazeController değişmedi) dışarıdan telafi eder.
    """

    def __init__(self, window_sec: float = 1.0, min_closing_speed: float = 300.0):
        # min_closing_speed: cm/s. Bu hızın altında kapanıyorsa "yavaş" sayılır.
        self.window_sec = window_sec
        self.min_closing_speed = min_closing_speed
        self._history = []   # [(t, dist_cm), ...]
        self.closing_rate = 0.0   # cm/s, pozitif = kapanıyor

    def update(self, dist_cm: float) -> float:
        now = time.time()
        self._history.append((now, dist_cm))
        self._history = [(t, d) for (t, d) in self._history if now - t <= self.window_sec]

        if len(self._history) >= 2:
            t0, d0 = self._history[0]
            t1, d1 = self._history[-1]
            dt = t1 - t0
            if dt > 0.1:
                self.closing_rate = (d0 - d1) / dt   # pozitif = mesafe azalıyor
        return self.closing_rate

    def needs_boost(self) -> bool:
        if len(self._history) < 5:
            return False
        return self.closing_rate < self.min_closing_speed

    def boost_amount(self) -> float:
        """closing_rate ne kadar düşükse o kadar fazla throttle ekle."""
        deficit = max(0.0, self.min_closing_speed - self.closing_rate)
        return min(0.35, deficit / 1000.0)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("kamikaze_vision")


# ===========================================================================
# FAZA ENUMERATİON
# ===========================================================================
class AttackPhase(Enum):
    GNSS_APPROACH  = "GNSS_APPROACH"   # GNSS ile yaklaş
    CAMERA_ARM     = "CAMERA_ARM"      # Kamera açıldı, YOLO hedef arıyor
    VISION_TRACK   = "VISION_TRACK"    # Kamera PID takip (GNSS KAPALI)
    DIVE_LOCK      = "DIVE_LOCK"       # Merkezde → dalış kilidi
    IMPACT         = "IMPACT"          # Çarpışma


# ===========================================================================
# EKRAN YAKALAMA
# ===========================================================================
class ScreenCapture:
    def __init__(self, window_name: str = "", target_w: int = 1280, target_h: int = 720):
        self.window_name = window_name
        self.target_w = target_w
        self.target_h = target_h
        self._hwnd = None
        self._mss = None
        self._monitor = None
        self._use_winapi = False

    def start(self):
        try:
            import mss as mss_module
            self._mss = mss_module.mss()
        except ImportError:
            raise RuntimeError("mss paketi eksik: pip install mss --break-system-packages")

        if self.window_name:
            self._hwnd = self._find_window(self.window_name)
            if self._hwnd:
                logger.info("Sim penceresi bulundu: '%s'", self.window_name)
                self._use_winapi = True
            else:
                logger.warning("Pencere bulunamadı: '%s'. Tam monitör yakalanacak.", self.window_name)

        if not self._use_winapi:
            self._monitor = self._mss.monitors[1]
            logger.info("Tam monitör yakalanıyor: %dx%d",
                        self._monitor["width"], self._monitor["height"])

    def _find_window(self, partial_title: str):
        try:
            import win32gui
            found = []
            def cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    if partial_title.lower() in win32gui.GetWindowText(hwnd).lower():
                        found.append(hwnd)
            win32gui.EnumWindows(cb, None)
            if not found:
                return None
            def area(h):
                try:
                    r = win32gui.GetWindowRect(h)
                    return (r[2]-r[0]) * (r[3]-r[1])
                except Exception:
                    return 0
            return max(found, key=area)
        except ImportError:
            logger.warning("pywin32 eksik: pip install pywin32 --break-system-packages")
            return None

    def grab(self) -> Optional[np.ndarray]:
        try:
            if self._use_winapi and self._hwnd:
                import win32gui, win32ui
                from ctypes import windll
                r = win32gui.GetWindowRect(self._hwnd)
                x, y, x2, y2 = r
                w, h = x2-x, y2-y
                if w <= 0 or h <= 0:
                    return None
                hwndDC = win32gui.GetWindowDC(self._hwnd)
                mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                bmp    = win32ui.CreateBitmap()
                bmp.CreateCompatibleBitmap(mfcDC, w, h)
                saveDC.SelectObject(bmp)
                ok = windll.user32.PrintWindow(self._hwnd, saveDC.GetSafeHdc(), 2)
                info = bmp.GetInfo()
                raw  = bmp.GetBitmapBits(True)
                win32gui.DeleteObject(bmp.GetHandle())
                saveDC.DeleteDC(); mfcDC.DeleteDC()
                win32gui.ReleaseDC(self._hwnd, hwndDC)
                if ok == 0:
                    return self._mss_region(x, y, w, h)
                img = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (info["bmHeight"], info["bmWidth"], 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                sct = self._mss.grab(self._monitor)
                img = cv2.cvtColor(np.array(sct), cv2.COLOR_BGRA2BGR)

            if img.shape[1] != self.target_w or img.shape[0] != self.target_h:
                img = cv2.resize(img, (self.target_w, self.target_h))
            return img
        except Exception as e:
            logger.debug("Grab hata: %s", e)
            return None

    def _mss_region(self, x, y, w, h):
        try:
            sct = self._mss.grab({"left": x, "top": y, "width": w, "height": h})
            img = cv2.cvtColor(np.array(sct), cv2.COLOR_BGRA2BGR)
            if img.shape[1] != self.target_w or img.shape[0] != self.target_h:
                img = cv2.resize(img, (self.target_w, self.target_h))
            return img
        except Exception:
            return None

    def stop(self):
        if self._mss:
            self._mss.close()


# ===========================================================================
# KALMAN FİLTRESİ (2D bbox merkez + hız)
# ===========================================================================
class BBoxKalman:
    def __init__(self, dt=1/30.0, accel_std=0.5, r_meas=1e-2):
        # State transition (F) ve Observation (H)
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
        
        # Kullanıcının önerdiği: İvme tabanlı (Kinematic) Q Matrisi
        dt2 = dt**2; dt3 = dt**3; dt4 = dt**4
        q_pos = (dt4 / 4) * accel_std
        q_cov = (dt3 / 2) * accel_std
        q_vel = dt2 * accel_std
        
        self.Q = np.array([
            [q_pos, 0,     q_cov, 0    ],
            [0,     q_pos, 0,     q_cov],
            [q_cov, 0,     q_vel, 0    ],
            [0,     q_cov, 0,     q_vel]
        ], dtype=np.float64)
        
        # Measurement noise (R)
        self.R = np.eye(2, dtype=np.float64) * r_meas
        
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.1
        self._init = False
        self._miss = 0

    def reset(self):
        self.x[:] = 0; self.P = np.eye(4)*0.1
        self._init = False; self._miss = 0

    def init(self, cx, cy):
        self.x = np.array([cx, cy, 0., 0.])
        self.P = np.eye(4) * 0.05
        self._init = True; self._miss = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, cx, cy):
        z = np.array([cx, cy])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self._miss = 0

    def on_miss(self, max_miss=12):
        self._miss += 1
        return self._miss >= max_miss

    @property
    def cx(self): return float(self.x[0])
    @property
    def cy(self): return float(self.x[1])
    @property
    def vcx(self): return float(self.x[2])
    @property
    def vcy(self): return float(self.x[3])
    @property
    def initialized(self): return self._init


def _norm_center(xyxy, W, H):
    x1,y1,x2,y2 = xyxy
    return ((x1+x2)/2 - W/2)/(W/2), ((y1+y2)/2 - H/2)/(H/2)


# ===========================================================================
# VİZYON THREAD (YOLO + Kalman, arka planda 30 Hz)
# ===========================================================================
@dataclass
class VisionResult:
    found: bool = False
    confirmed: bool = False
    confidence: float = 0.0
    bbox_xyxy: Optional[tuple] = None   # ham YOLO kutusu (piksel)
    kf_cx: float = 0.0                  # Kalman smooth merkez [-1,1]
    kf_cy: float = 0.0
    kf_vcx: float = 0.0                 # Kalman hız tahmini
    kf_vcy: float = 0.0
    kf_valid: bool = False              # Kalman başlatılmış mı
    frame: Optional[np.ndarray] = None  # Son ham kare (overlay için)


class VisionThread:
    """YOLO + Kalman filtresi, ayrı thread'de 30 Hz çalışır."""

    def __init__(self, weights: str, window_name: str, conf: float = 0.35,
                 confirm_streak: int = 4, lost_streak: int = 8,
                 use_obs: bool = False, camera_index: int = 1,
                 show_window: bool = False,
                 confirm_window: int = 10, confirm_ratio: float = 0.4):
        self.weights = weights
        self.conf = conf
        self.confirm_streak = confirm_streak
        self.lost_streak = lost_streak
        self.show_window = show_window

        # --- DEBUG NOTU (sliding-window confirm) ---
        # Eski mantık: ARDIŞIK 4 kare gerekiyordu (_hit += 1, found=False
        # gelince anında _hit=0). Gerçek log: hit=1/4 dakikalarca sabit
        # kaldı - yani found True/False/True/False şeklinde titriyordu,
        # asla 4 ardışık True üst üste gelmedi. Tek bir kayıp kare bütün
        # streak'i sıfırlıyordu.
        # Yeni mantık: son `confirm_window` karenin en az `confirm_ratio`
        # oranı kadarı (örn. 10 karenin %40'ı = 4 tanesi) found=True ise
        # onaylanır - ardışıklık şartı yok, titrek tespitlere dayanıklı.
        self.confirm_window = confirm_window
        self.confirm_ratio = confirm_ratio
        self._found_history = []  # son N karenin found bool listesi

        # Capture
        if use_obs:
            self._cap_mode = "obs"
            self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                raise RuntimeError(f"OBS kamera açılamadı (index={camera_index})")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            self._cap_mode = "screen"
            self._sc = ScreenCapture(window_name=window_name, target_w=1280, target_h=720)
            self._sc.start()

        # YOLO
        from ultralytics import YOLO
        logger.info("YOLO yükleniyor: %s (conf=%.2f)", weights, conf)
        self._model = YOLO(weights)
        logger.info("Model sınıfları: %s", self._model.names)

        # Kalman + confirmer
        self._kf = BBoxKalman(dt=1/30.0, accel_std=0.5, r_meas=1e-2)
        self._hit = 0
        self._miss_conf = 0
        self._confirmed = False

        # Shared state
        self._lock = threading.Lock()
        self._result = VisionResult()
        self._running = False
        self._thread = None
        
        # CSRT Tracker state
        self._csrt_tracker = None
        self._csrt_tracking = False
        self._csrt_fails = 0
        self._csrt_since_yolo = 999

    def _grab(self):
        if self._cap_mode == "obs":
            ret, f = self._cap.read()
            return f if ret else None
        return self._sc.grab()

    def _osd_rectangles(self, w: int, h: int):
        return [
            (0, 0, int(w*0.25), int(h*0.25)),
            (int(w*0.76), 0, w, int(h*0.25)),
            (0, int(h*0.70), int(w*0.30), h),
            (int(w*0.70), int(h*0.58), w, h),
            (int(w*0.24), int(h*0.50), int(w*0.82), int(h*0.86)),
        ]

    @staticmethod
    def _intersection_ratio(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1))
        return inter / area

    def _reject_bbox(self, xyxy, w, h):
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            return True
        bw, bh = x2 - x1, y2 - y1
        area_px = bw * bh
        area_ratio = area_px / max(1.0, float(w * h))
        aspect = bw / max(1.0, bh)
        cx = (x1 + x2) * 0.5 / max(1, w)
        cy = (y1 + y2) * 0.5 / max(1, h)

        if area_px < 55 or area_ratio > 0.08:
            return True
        if aspect < 0.15 or aspect > 6.5:
            return True
        if any(self._intersection_ratio((x1, y1, x2, y2), rect) >= 0.10
               for rect in self._osd_rectangles(w, h)):
            return True
        # Yanlis kilitlerin buyuk kismi ufuk alti zemin/OSD lekeleri.
        if cy > 0.55 and area_ratio < 0.015:
            return True
        if cx < 0.05 or cx > 0.95 or cy < 0.03 or cy > 0.92:
            return True
        return False

    def _apply_osd_mask(self, frame):
        """
        OSD yazılarını ve drone burnunu siyah kutularla gizleyerek
        YOLO ve CSRT'nin yanlış yerlere kilitlenmesini engeller.
        """
        masked = frame.copy()
        h, w = masked.shape[:2]
        for x1, y1, x2, y2 in self._osd_rectangles(w, h):
            cv2.rectangle(masked, (x1, y1), (x2, y2), (0,0,0), -1)
        
        # 1. Sol Üst (İrtifa / Batarya vb. veriler)
        cv2.rectangle(masked, (0, 0), (int(w*0.18), int(h*0.20)), (0,0,0), -1)
        # 2. Sağ Üst (Hız / Süre)
        cv2.rectangle(masked, (int(w*0.80), 0), (w, int(h*0.18)), (0,0,0), -1)
        # 3. Sol Alt (Akım vb.)
        cv2.rectangle(masked, (0, int(h*0.75)), (int(w*0.20), h), (0,0,0), -1)
        # 4. Sağ Alt (Uçuş Modu)
        cv2.rectangle(masked, (int(w*0.75), int(h*0.80)), (w, h), (0,0,0), -1)
        # 5. Ekranın tam alt ortasındaki dronun burnu
        cv2.rectangle(masked, (int(w*0.35), int(h*0.85)), (int(w*0.65), h), (0,0,0), -1)
        # 6. Ekranın ortasındaki "ARMED TRIGGER: NOT READY" yazıları
        cv2.rectangle(masked, (int(w*0.35), int(h*0.55)), (int(w*0.65), int(h*0.75)), (0,0,0), -1)
        
        return masked

    def _detect(self, frame):
        h, w = frame.shape[:2]
        # Bytetrack kullanımı ile hedefleri izle (persist=True)
        res = self._model.predict(frame, imgsz=640, conf=self.conf, iou=0.45, verbose=False)
        if not res or len(res[0].boxes) == 0:
            return False, 0.0, None, (h, w)
        boxes = res[0].boxes
        confs = boxes.conf.tolist()
        xyxys = boxes.xyxy.tolist()
        clss = boxes.cls.tolist() if boxes.cls is not None else [None] * len(confs)
        names = getattr(self._model, "names", {}) or {}
        candidates = []
        for i, (conf, raw_xyxy) in enumerate(zip(confs, xyxys)):
            cls_id = int(clss[i]) if clss[i] is not None else None
            cls_name = str(names.get(cls_id, "")).lower()
            if len(names) > 1 and not any(k in cls_name for k in ("talon", "drone", "uav", "iha", "aircraft")):
                continue
            xyxy = tuple(float(v) for v in raw_xyxy)
            if self._reject_bbox(xyxy, w, h):
                continue
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) * 0.5 / max(1, w)
            cy = (y1 + y2) * 0.5 / max(1, h)
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
            center_bonus = 0.12 * (1.0 - min(1.0, abs(cx - 0.5) / 0.5))
            sky_bonus = 0.10 if cy < 0.55 else 0.0
            size_bonus = min(0.08, area_ratio * 20.0)
            candidates.append((conf + center_bonus + sky_bonus + size_bonus, conf, xyxy))
        if not candidates:
            return False, 0.0, None, (h, w)
        _, conf, xyxy = max(candidates, key=lambda item: item[0])
        return True, conf, xyxy, (h, w)

    def _draw(self, frame, found, xyxy, conf, kf_cx, kf_cy, confirmed):
        h, w = frame.shape[:2]
        # Merkez çizgileri
        cv2.line(frame, (w//2, 0), (w//2, h), (70,70,70), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (70,70,70), 1)

        # -----------------------------------------------------
        # Kullanıcının önerdiği Legend ve Renk Kodlaması
        # True (Yeşil), Predict (Mavi), Update (Kırmızı)
        # -----------------------------------------------------
        cv2.circle(frame, (20, 20), 6, (0, 255, 0), -1)
        cv2.putText(frame, "True (YOLO)", (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.circle(frame, (20, 50), 6, (255, 0, 0), -1)
        cv2.putText(frame, "Predict (Kalman)", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Kalman Update (Predict + Hız) -> Mavi Göstergesi
        if self._kf.initialized:
            kx = int((kf_cx+1)/2*w)
            ky = int((kf_cy+1)/2*h)
            cv2.circle(frame, (kx,ky), 8, (255,0,0), 3)  # Predict: Mavi
            
            # Hız vektörü oku
            vx_px = int(self._kf.vcx * w * 0.5)
            vy_px = int(self._kf.vcy * h * 0.5)
            cv2.arrowedLine(frame, (kx,ky), (kx+vx_px, ky+vy_px),
                            (255,0,0), 2, tipLength=0.3)

        # YOLO True / Update bbox (Yeşil)
        if found and xyxy is not None:
            x1,y1,x2,y2 = (int(v) for v in xyxy)
            color = (0,255,0) if confirmed else (0,220,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"talon {conf:.2f}" + (" ✓" if confirmed else "")
            cv2.putText(frame, label, (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            # True YOLO merkez (Kırmızı nokta) -> Gerçek verinin Update olduğunu belirtmek için
            cx_px = (x1+x2)//2; cy_px = (y1+y2)//2
            cv2.circle(frame, (cx_px,cy_px), 6, (0,0,255), -1) # Update: Kırmızı
        else:
            cv2.putText(frame, "TESPIT YOK", (20,85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # Durum satırı
        ratio_now = (sum(self._found_history)/len(self._found_history)
                    if self._found_history else 0.0)
        status = ("VİZYON: ONAYLANDI" if confirmed
                  else f"ARAMA | ratio={ratio_now:.0%} ({sum(self._found_history)}/{len(self._found_history)}) lost={self._miss_conf}")
        cv2.putText(frame, status, (20, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0,255,0) if confirmed else (0,165,255), 2)
        # Conf threshold
        cv2.putText(frame, f"conf_thr={self.conf:.2f}", (20,h-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,0), 1)
        return frame

    def _loop(self):
        dt = 1/30.0
        grab_fails = 0
        while self._running:
            t0 = time.time()
            frame = self._grab()
            if frame is None:
                grab_fails += 1
                if grab_fails % 30 == 1:
                    logger.warning("Kare alınamadı (%d). Pencere adı doğru mu?", grab_fails)
                time.sleep(0.1)
                continue
            grab_fails = 0

            # --- OSD MASKELEME ---
            # AI'nin yazılara kilitlenmemesi için maskelenmiş bir kopya çıkartılır
            masked_frame = self._apply_osd_mask(frame)

            found, conf, xyxy, (H,W) = self._detect(masked_frame)
            yolo_found = found

            # --- BBOX + CONF FİLTRESİ ---
            # 400 metreden yakalayabilmek için alan limitini çok düşürdük
            if found and xyxy is not None:
                bbox_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if conf < max(0.40, self.conf) or bbox_area < 55 or self._reject_bbox(xyxy, W, H):
                    found = False  # Güvenilir değil, filtrelendi
                    yolo_found = False

            # --- CSRT HİBRİD TAKİP (YOLO + CSRT) ---
            if found and xyxy is not None:
                # YOLO bulduysa CSRT Tracker'ı YOLO kutusuyla GÜNCELLE (Re-init)
                # Bu sayede YOLO'nun gördüğü her an CSRT'nin kaymasını (drift) sıfırlamış oluruz.
                try:
                    self._csrt_tracker = cv2.TrackerCSRT_create()
                    bw, bh = max(2, int(xyxy[2]-xyxy[0])), max(2, int(xyxy[3]-xyxy[1]))
                    bbox_cv = (int(xyxy[0]), int(xyxy[1]), bw, bh)
                    self._csrt_tracker.init(masked_frame, bbox_cv)
                    self._csrt_tracking = True
                    self._csrt_fails = 0
                    self._csrt_since_yolo = 0
                except Exception as e:
                    # OpenCV contrib kurulu değilse hata yutulur
                    pass
            elif self._csrt_tracking and self._csrt_tracker is not None:
                # YOLO kare atladıysa (bulamadıysa) CSRT devralır
                self._csrt_since_yolo += 1
                success, bbox = self._csrt_tracker.update(masked_frame)
                if success:
                    found = True
                    xyxy = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
                    if self._csrt_since_yolo > 10 or self._reject_bbox(xyxy, W, H):
                        found = False
                        xyxy = None
                        self._csrt_tracking = False
                        self._csrt_tracker = None
                    else:
                        conf = 0.30  # CSRT tahmini; YOLO gibi onay guveni sayilmaz.
                else:
                    self._csrt_fails += 1
                    if self._csrt_fails > 15:  # Yarım saniye üst üste bulamazsa CSRT iptal
                        self._csrt_tracking = False
                        self._csrt_tracker = None

            # Kalman güncelle
            if found:
                cx, cy = _norm_center(xyxy, W, H)
                if not self._kf.initialized:
                    self._kf.init(cx, cy)
                else:
                    self._kf.predict()
                    self._kf.update(cx, cy)
                kf_valid = True
            else:
                if self._kf.initialized:
                    if self._kf.on_miss(12):
                        self._kf.reset()
                        kf_valid = False
                    else:
                        self._kf.predict()
                        kf_valid = True
                else:
                    kf_valid = False

            # CSRT sadece kisa sureli takip destegi verir; yeni kilit/onay uretmez.
            confirm_hit = yolo_found or (self._confirmed and found and self._csrt_since_yolo <= 3)

            # Confirmer - sliding window (ardışık değil, pencere içi oran)
            self._found_history.append(confirm_hit)
            if len(self._found_history) > self.confirm_window:
                self._found_history.pop(0)

            if confirm_hit:
                self._hit += 1; self._miss_conf = 0
            else:
                self._miss_conf += 1; self._hit = 0

            # Pencere doluysa oran kontrolü yap
            if len(self._found_history) >= self.confirm_window:
                hit_ratio = sum(self._found_history) / len(self._found_history)
                if hit_ratio >= self.confirm_ratio:
                    self._confirmed = True
                elif self._miss_conf >= self.lost_streak:
                    self._confirmed = False
            elif self._miss_conf >= self.lost_streak:
                self._confirmed = False

            result = VisionResult(
                found=found,
                confirmed=self._confirmed,
                confidence=conf,
                bbox_xyxy=xyxy,
                kf_cx=self._kf.cx if kf_valid else 0.0,
                kf_cy=self._kf.cy if kf_valid else 0.0,
                kf_vcx=self._kf.vcx if kf_valid else 0.0,
                kf_vcy=self._kf.vcy if kf_valid else 0.0,
                kf_valid=kf_valid,
                frame=frame.copy(),
            )
            with self._lock:
                self._result = result

            # Debug pencere
            if self.show_window:
                disp = self._draw(frame.copy(), found, xyxy, conf,
                                  self._kf.cx, self._kf.cy, self._confirmed)
                try:
                    cv2.imshow("Avci Drone - Kamera Tespiti", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._running = False
                        break
                except cv2.error:
                    logger.error("[HATA] OpenCV GUI desteklenmiyor. Lutfen 'pip install opencv-python' kurun! Goruntu kapatildi, otonom vurus devam ediyor.")
                    self.show_window = False

            time.sleep(max(0.0, dt - (time.time()-t0)))

        try:
            if self.show_window:
                cv2.destroyAllWindows()
        except cv2.error:
            pass

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("VisionThread başladı.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap_mode == "obs":
            self._cap.release()
        else:
            self._sc.stop()

    def get(self) -> VisionResult:
        with self._lock:
            return self._result


# ===========================================================================
# VİZYON PID + DALIŞI KİLİTLEME KONTROLCÜSÜ
# ===========================================================================
class VisionDiveController:
    """
    Kamera tespitinden yaw/throttle/pitch komutu üretir.
    Hedef merkezde ve yakınken → TAM GAZ DALIŞI başlatır.
    """

    # PID gainleri - Sabit Kanat (Fixed-Wing) Dinamiklerine Uygun
    KP_YAW    = 0.35;  KD_YAW    = 0.10;  KFF_YAW    = 0.05
    KP_PITCH  = 0.35;  KD_PITCH  = 0.10;  KFF_PITCH  = 0.05  # Y ekseni hedefini vurmak için Pitch PID

    # Dalış parametreleri
    CENTER_TOL      = 0.15   
    DIVE_LOCK_COUNT = 4      
    BASE_THROTTLE   = 0.80   # Sabit kanat düşmesin diye her zaman motor itkisi %80
    DIVE_THROTTLE   = 1.00   # Son dalışta tam güç füze etkisi

    def __init__(self):
        self._yaw_int  = 0.0; self._yaw_prev  = 0.0; self._yaw_t  = None
        self._pitch_int  = 0.0; self._pitch_prev  = 0.0; self._pitch_t  = None
        self._center_count = 0
        self._dive_locked  = False

    def reset(self):
        self._yaw_int  = 0.0; self._yaw_prev  = 0.0; self._yaw_t  = None
        self._pitch_int  = 0.0; self._pitch_prev  = 0.0; self._pitch_t  = None
        self._center_count = 0
        self._dive_locked  = False

    def _pid(self, err, integral, prev, kp, kd, ff_val, kff, t, max_i=0.5):
        now = time.time()
        dt  = (now - t) if t else 0.02
        dt  = max(dt, 0.005)
        integral = max(-max_i, min(max_i, integral + err * dt))
        deriv = (err - prev) / dt
        out   = kp * err + kd * deriv + kff * ff_val
        return out, integral, err, now

    def compute(self, r: VisionResult) -> dict:
        if not r.kf_valid:
            return {"thr": 0.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0,
                    "dive": False, "center_count": 0}

        # --- KALMAN PREDİCTİON ---
        # Hedefin 150ms sonraki tahmini konumuna uç
        # Hareketli hedefte vuruş isabetini ciddi şekilde artırır
        cx  = r.kf_cx + r.kf_vcx * 0.15
        cy  = r.kf_cy + r.kf_vcy * 0.15
        vcx = r.kf_vcx; vcy = r.kf_vcy

        # --- YAW (Sağ-Sol) ---
        yaw_out, self._yaw_int, self._yaw_prev, self._yaw_t = self._pid(
            cx, self._yaw_int, self._yaw_prev,
            self.KP_YAW, self.KD_YAW, vcx, self.KFF_YAW, self._yaw_t)
        yaw_cmd = max(-0.8, min(0.8, yaw_out))

        # --- PITCH (Yukarı-Aşağı: Sabit Kanat için Elevator Kontrolü) ---
        # cy < 0 ise hedef yukarıda, burnu kaldırmak (PITCH < 0) gerekebilir (flight controller'ın PITCH yönüne bağlı)
        # Genelde kamikazelerde hedef ekranda aşağıdaysa (cy > 0), dronun burnunu ezip dalması (PITCH > 0) gerekir.
        pitch_out, self._pitch_int, self._pitch_prev, self._pitch_t = self._pid(
            cy, self._pitch_int, self._pitch_prev,
            self.KP_PITCH, self.KD_PITCH, vcy, self.KFF_PITCH, self._pitch_t)
        # Elevator aşırı tepki verip takla (stall) atmaması için pitch_cmd %40 ile sınırlandı
        pitch_cmd = max(-0.4, min(0.4, pitch_out))

        # --- THROTTLE (İtici Güç) ---
        thr_cmd = self.BASE_THROTTLE

        # --- MERKEZLEMEyi SAYMA (dalış kilidi için) ---
        centered = (abs(cx) < self.CENTER_TOL) and (abs(cy) < self.CENTER_TOL)
        if centered:
            self._center_count += 1
        else:
            self._center_count = max(0, self._center_count - 1)

        # --- DALIŞI KİLİTLE ---
        if self._center_count >= self.DIVE_LOCK_COUNT:
            self._dive_locked = True

        if self._dive_locked:
            return {
                "thr": self.DIVE_THROTTLE,
                "pitch": pitch_cmd,
                "roll": 0.0,
                "yaw": yaw_cmd,
                "dive": True,
                "center_count": self._center_count,
            }

        return {
            "thr": thr_cmd,
            "pitch": pitch_cmd,
            "roll": 0.0,
            "yaw": yaw_cmd,
            "dive": False,
            "center_count": self._center_count,
        }


# ===========================================================================
# ANA GÖREV DÖNGÜSÜ
# ===========================================================================
class KamikazeVisionMission:
    """
    GNSS → Vision geçişli tam kamikaze görev döngüsü.
    """

    def __init__(self, args):
        self.args = args

        # GNSS kontrolcüsü (mevcut çalışan kod)
        self.gnss_ctrl = DroneKamikazeController()

        # Vizyon thread — LAZY START: başlangıçta oluşturulur ama başlatılmaz
        # YOLO yalnızca 20m'ye girildiğinde çalışacak → GPU tasarrufu
        self.vision = VisionThread(
            weights=args.weights,
            window_name=args.window_name,
            conf=args.conf,
            confirm_streak=args.confirm_streak,
            lost_streak=args.lost_streak,
            use_obs=args.use_obs,
            camera_index=args.camera_index,
            show_window=args.show,
            confirm_window=args.confirm_window,
            confirm_ratio=args.confirm_ratio,
        )

        # Vizyon PID + dalış kilidi
        self.vision_ctrl = VisionDiveController()

        # Kapanma hızı izleyici
        self.closing_monitor = ClosingRateMonitor(
            window_sec=1.0, min_closing_speed=args.min_closing_speed)

        # Durum
        self.phase = AttackPhase.GNSS_APPROACH
        self._last_log = 0.0
        self._last_locked_yaw = 0.0
        self.vision_started = False                    # YOLO henüz çalışmıyor
        self.camera_arm_distance = args.camera_arm_distance  # 20m varsayılan
        self._last_log = 0.0
        self._mission_start_t = None
        self._initial_dist = None

    def start(self):
        logger.info("SDK bağlanıyor...")
        if not drone.connect():
            raise RuntimeError("Sim bağlantısı başarısız. 'Play' modunda mı?")
        logger.info("Bağlandı.")

        # NOT: Vision thread burada başlatılmıyor!
        # 20m'ye girildiğinde step() içinde lazy-start edilecek.
        logger.info("YOLO kapalı — 20m'ye girildiğinde otomatik başlayacak.")

        self._mission_start_t = time.time()
        self._initial_dist = self._gnss_distance()
        logger.info("GÖREV BAŞLADI. Başlangıç mesafesi=%.0fcm (%.1fm). İlk faz: %s",
                    self._initial_dist, self._initial_dist/100, self.phase.value)

    def stop(self):
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, arm=False)

        # --- GÖREV ÖZET RAPORU (test mühendisliği için) ---
        if self._mission_start_t is not None:
            elapsed = time.time() - self._mission_start_t
            final_dist = self._gnss_distance()
            logger.info("=" * 60)
            logger.info("GÖREV ÖZETİ")
            logger.info("  Süre               : %.1f sn", elapsed)
            logger.info("  Başlangıç mesafesi : %.0fcm (%.1fm)",
                        self._initial_dist, self._initial_dist/100)
            logger.info("  Bitiş mesafesi     : %.0fcm (%.1fm)",
                        final_dist, final_dist/100)
            logger.info("  Net kapanma        : %.0fcm",
                        self._initial_dist - final_dist)
            logger.info("  Son ortalama kapanma hızı: %.0f cm/s",
                        self.closing_monitor.closing_rate)
            logger.info("  Son faz            : %s", self.phase.value)
            if self.phase == AttackPhase.GNSS_APPROACH:
                logger.info("  !! UYARI: Görev GNSS fazında bitti, vizyona hiç geçilmedi.")
                logger.info("     --camera-arm-distance değerini artırmayı veya")
                logger.info("     --min-closing-speed değerini düşürmeyi deneyin.")
            elif self.phase == AttackPhase.CAMERA_ARM:
                logger.info("  !! UYARI: Görev CAMERA_ARM fazında bitti, YOLO hedef bulamadı.")
                logger.info("     Hedef kamerada görünüyor mu? conf/bbox filtresi çok sıkı mı?")
            logger.info("=" * 60)

        # Vision thread'i sadece başlatılmışsa durdur
        if self.vision_started:
            self.vision.stop()
        drone.disconnect()
        logger.info("Görev durduruldu.")

    def _gnss_distance(self) -> float:
        tel = drone.get_telemetry()
        dp  = tel["drone"]["position"]
        tp  = tel["target"]["position"]
        return math.sqrt((tp[0]-dp[0])**2 + (tp[1]-dp[1])**2 + (tp[2]-dp[2])**2)

    def _maybe_transition(self, dist_cm: float, vis: VisionResult):
        """
        İKİ AŞAMALI FAZ GEÇİŞLERİ:
        GNSS_APPROACH → CAMERA_ARM → VISION_TRACK → DIVE_LOCK → IMPACT

        Kritik kurallar:
        - CAMERA_ARM'a geçiş: mesafe ≤ 20m (YOLO açılır)
        - VISION_TRACK'a geçiş: ÜÇLÜ ŞART (confirmed + conf≥0.55 + bbox≥12k)
        - VISION_TRACK'teyken GNSS TAMAMEN KAPALI — PID kavgası yok
        - VISION_TRACK'ten geri düşüş CAMERA_ARM'a olur, GNSS'e değil
        """
        current = self.phase

        if current == AttackPhase.GNSS_APPROACH:
            # 20m'ye girince kamerayı aç
            if dist_cm <= self.camera_arm_distance:
                logger.info(">>> GECIS: GNSS_APPROACH -> CAMERA_ARM (dist=%.0fcm, <=%.0fm)",
                            dist_cm, self.camera_arm_distance/100)
                self.phase = AttackPhase.CAMERA_ARM
                self.vision_ctrl.reset()

        elif current == AttackPhase.CAMERA_ARM:
            # ÜÇLÜ GEÇİŞ ŞARTI: confirmed + yeterli confidence + yeterli bbox
            if vis.confirmed:
                logger.info(">>> GECIS: CAMERA_ARM -> VISION_TRACK "
                            "(hedef onaylandi! conf=%.2f)", vis.confidence)
                self.phase = AttackPhase.VISION_TRACK
                self.vision_ctrl.reset()
            elif dist_cm > self.camera_arm_distance * 1.25:
                # Uzaklaştıysa GNSS'e geri dön, kamerayı kapat
                logger.info(">>> GECIS: CAMERA_ARM -> GNSS_APPROACH "
                            "(hedef uzaklasti dist=%.0fcm)", dist_cm)
                self.phase = AttackPhase.GNSS_APPROACH

        elif current == AttackPhase.VISION_TRACK:
            if not vis.confirmed:
                # GNSS'e DEĞİL, CAMERA_ARM'a geri dön
                # Böylece kamera açık kalır, YOLO aramaya devam eder
                logger.info(">>> GECIS: VISION_TRACK -> CAMERA_ARM (takip kaybi)")
                self.phase = AttackPhase.CAMERA_ARM

    def step(self):
        dist_cm = self._gnss_distance()
        closing_rate = self.closing_monitor.update(dist_cm)

        # ===================================================================
        # LAZY YOLO START: 20m'ye girince kamerayı aç
        # dist_cm > 50 şartı: Simülasyon başlarken (henüz hedefe ışınlanmadan)
        # mesafe 0 ölçülürse kameranın hatalı açılmasını engeller.
        # ===================================================================
        if (not self.vision_started) and 50 < dist_cm <= self.camera_arm_distance:
            logger.info("=" * 50)
            logger.info(">>> %.0fm'ye girildi -- YOLO BASLATILIYOR!", dist_cm/100)
            logger.info("=" * 50)
            self.vision.start()
            time.sleep(0.5)  # Model warmup
            self.vision_started = True

        # Vision sonucu al (başlatılmamışsa boş)
        if self.vision_started:
            vis = self.vision.get()
        else:
            vis = VisionResult()

        self._maybe_transition(dist_cm, vis)

        phase = self.phase
        now   = time.time()

        if phase == AttackPhase.GNSS_APPROACH:
            # ===============================================================
            # FAZ 1: GNSS YAKLAŞMA (YOLO KAPALI)
            # ===============================================================
            tel  = drone.get_telemetry()
            ctrl = self.gnss_ctrl.calculate_control_inputs(tel)

            throttle_cmd = ctrl["throttle"]

            # Kapanma hızı telafisi
            boost_applied = 0.0
            if self.closing_monitor.needs_boost():
                boost_applied = self.closing_monitor.boost_amount()
                throttle_cmd = min(self.args.max_gnss_throttle,
                                   throttle_cmd + boost_applied)

            drone.set_control_surfaces(
                throttle_cmd, ctrl["pitch"], ctrl["roll"], ctrl["yaw"], True)

            if now - self._last_log > 0.4:
                self._last_log = now
                logger.info(
                    "[GNSS] dist=%.0fcm(%.1fm) closing=%.0fcm/s boost=%.2f "
                    "faz=%d thr=%.2f(+%.2f) pitch=%.2f yaw=%.2f",
                    dist_cm, dist_cm/100, closing_rate, boost_applied,
                    ctrl["phase"], throttle_cmd, boost_applied,
                    ctrl["pitch"], ctrl["yaw"])

        elif phase == AttackPhase.CAMERA_ARM:
            # ===============================================================
            # FAZ 2: CAMERA_ARM — YOLO ARIOR, GNSS HAFİFLETİLMİŞ
            # GNSS ile yaklaşmaya devam et ama throttle kısılmış
            # ===============================================================
            tel  = drone.get_telemetry()
            ctrl = self.gnss_ctrl.calculate_control_inputs(tel)

            # Throttle'ı %75'e düşür — kamerayı kaçırıp geçmesin
            throttle_cmd = ctrl["throttle"] * 0.75
            if self.closing_monitor.needs_boost():
                throttle_cmd = min(self.args.max_gnss_throttle,
                                   throttle_cmd + self.closing_monitor.boost_amount())

            drone.set_control_surfaces(
                throttle_cmd, ctrl["pitch"] * 0.65,
                ctrl["roll"], ctrl["yaw"], True)

            if now - self._last_log > 0.4:
                self._last_log = now
                hist = self.vision._found_history
                ratio = sum(hist)/len(hist) if hist else 0.0
                logger.info(
                    "[KAMERA_ARM] dist=%.0fcm(%.1fm) closing=%.0fcm/s "
                    "vis_found=%s ratio=%.0f%%(%d/%d) miss=%d conf=%.2f",
                    dist_cm, dist_cm/100, closing_rate,
                    vis.found, ratio*100, sum(hist), len(hist),
                    self.vision._miss_conf, vis.confidence)

        else:
            # ===============================================================
            # FAZ 3/4: VISION_TRACK veya DIVE_LOCK
            # GNSS TAMAMEN DEVRE DIŞI — sadece kamera PID
            # ===============================================================
            if self.phase == AttackPhase.DIVE_LOCK:
                # DALIŞ KİLİDİ: Vizyon çalışıyorsa PITCH ve YAW'ı hedefe göre güncelle, körse son komutu koru!
                if vis.kf_valid:
                    vcmd = self.vision_ctrl.compute(vis)
                    self._last_locked_yaw = vcmd["yaw"]
                    self._last_locked_pitch = vcmd["pitch"]
                    log_msg = f"[DALIŞ KİLİDİ - GÖREREK] kf=({vis.kf_cx:.2f},{vis.kf_cy:.2f}) yaw={self._last_locked_yaw:.2f} pitch={self._last_locked_pitch:.2f} FULL THR!"
                else:
                    log_msg = f"[KÖRLEME DALIŞ KİLİDİ] yaw={self._last_locked_yaw:.2f} pitch={self._last_locked_pitch:.2f} FULL THR (Hedef araniyor)!"

                drone.set_control_surfaces(
                    self.vision_ctrl.DIVE_THROTTLE, 
                    self._last_locked_pitch, 
                    0.0, 
                    self._last_locked_yaw, 
                    True)
                
                if now - self._last_log > 0.2:
                    self._last_log = now
                    logger.info(log_msg)
            else:
                vcmd = self.vision_ctrl.compute(vis)
                drone.set_control_surfaces(
                    vcmd["thr"], vcmd["pitch"], vcmd["roll"], vcmd["yaw"], True)

                if vcmd["dive"]:
                    self.phase = AttackPhase.DIVE_LOCK
                    self._last_locked_yaw = vcmd["yaw"]
                    self._last_locked_pitch = vcmd["pitch"]
                    if now - self._last_log > 0.2:
                        self._last_log = now
                        logger.info("[DALIŞ KİLİDİ BAŞLADI] kf=(%.2f,%.2f) yaw=%.2f pitch=%.2f FULL THR!",
                                    vis.kf_cx, vis.kf_cy, vcmd["yaw"], vcmd["pitch"])
                else:
                    if now - self._last_log > 0.3:
                        self._last_log = now
                        logger.info("[VİZYON TAKİP] kf=(%.2f,%.2f) vel=(%.3f,%.3f) "
                                    "center=%d thr=%.2f yaw=%.2f pitch=%.2f dist=%.0fm",
                                    vis.kf_cx, vis.kf_cy,
                                    vis.kf_vcx, vis.kf_vcy,
                                    vcmd["center_count"],
                                    vcmd["thr"], vcmd["yaw"], vcmd["pitch"],
                                    dist_cm/100)

    def run(self, hz: float = 50.0):
        self.start()
        dt = 1.0 / hz
        try:
            while True:
                t0 = time.time()
                self.step()
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            logger.info("Ctrl+C ile durduruldu.")
        finally:
            self.stop()


# ===========================================================================
# PENCERE LİSTELEME YARDIMCISI
# ===========================================================================
def list_windows():
    print("\nAçık pencereler (200x200'den büyük):")
    try:
        import win32gui
        def cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if t.strip():
                    r = win32gui.GetWindowRect(hwnd)
                    w, h = r[2]-r[0], r[3]-r[1]
                    if w > 200 and h > 200:
                        print(f"  [{w:5d}x{h:4d}]  {t}")
        win32gui.EnumWindows(cb, None)
    except ImportError:
        print("  pywin32 gerekli: pip install pywin32 --break-system-packages")
    print("\nKullanım örneği:")
    print("  py kamikaze_vision.py --window-name 'Unreal' --weights talon_v4.pt --show")


# ===========================================================================
# GİRİŞ NOKTASI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Avci Drone - Iki Asamali Vision System (GNSS -> Kamera -> Dalis)")
    parser.add_argument("--weights",      default="talon_v4.pt",
                        help="YOLO model dosyası")
    parser.add_argument("--window-name",  default="",
                        help="Sim pencere başlığının bir parçası. Ör: 'Unreal', 'AvciDrone'")
    parser.add_argument("--show",         action="store_true",
                        help="Kamera tespit penceresini göster (önerilir)")
    parser.add_argument("--conf",         type=float, default=0.35,
                        help="YOLO confidence eşiği (varsayılan 0.35). NOT: VisionThread "
                             "içinde conf<0.55 ve bbox<12000 olan tespitler ek olarak "
                             "filtrelenir.")
    parser.add_argument("--camera-arm-distance", type=float, default=2000.0,
                        help="Kac cm'de kamera acilir / YOLO baslar (varsayilan 2000=20m). "
                             "Bu mesafeye girildiginde GNSS->CAMERA_ARM gecisi yapilir "
                             "ve YOLO lazy-start edilir.")
    parser.add_argument("--vision-range", type=float, default=7000.0,
                        help="(KULLANILMIYOR - geriye dönük uyumluluk). "
                             "camera-arm-distance kullanın.")
    parser.add_argument("--min-closing-speed", type=float, default=300.0,
                        help="cm/s. GNSS fazında mesafe en az bu hızda kapanmıyorsa "
                             "throttle'a otomatik ek itki (boost) enjekte edilir "
                             "(varsayılan 300 cm/s = 3 m/s).")
    parser.add_argument("--max-gnss-throttle", type=float, default=1.0,
                        help="GNSS fazında boost dahil maksimum throttle (varsayılan 1.0)")
    parser.add_argument("--confirm-window", type=int, default=10,
                        help="Sliding-window confirm: kaç karelik pencere bakılsın "
                             "(varsayılan 10). Ardışıklık ŞARTI YOK.")
    parser.add_argument("--confirm-ratio", type=float, default=0.4,
                        help="Pencere içindeki found=True oranı bu eşiği geçerse "
                             "onaylanır (varsayılan 0.4 = %%40, yani 10 karenin 4'ü).")
    parser.add_argument("--confirm-streak", type=int, default=4,
                        help="(eski parametre, geriye dönük uyumluluk için tutuldu)")
    parser.add_argument("--lost-streak", type=int, default=8,
                        help="Ardışık kaç miss sonrası onay geri alınır (varsayılan 8)")
    parser.add_argument("--hz",           type=float, default=50.0,
                        help="Kontrol döngüsü Hz (varsayılan 50)")
    parser.add_argument("--use-obs",      action="store_true",
                        help="OBS Virtual Camera kullan")
    parser.add_argument("--camera-index", type=int, default=1,
                        help="OBS modu kamera index")
    parser.add_argument("--list-windows", action="store_true",
                        help="Açık pencereleri listele ve çık")
    args = parser.parse_args()

    if args.list_windows:
        list_windows()
        import sys; sys.exit(0)

    if not args.window_name and not args.use_obs:
        print("[UYARI] --window-name verilmedi. Tam monitör yakalanacak (yavaş).")
        print("        Önce: py kamikaze_vision.py --list-windows")
        print()

    # Kullanıcı pencereyi görmediğini belirttiği için görselleştirmeyi ZORUNLU açıyoruz
    args.show = True

    # Baslangiç bilgisi
    print("=" * 60)
    print("  AVCI DRONE -- IKI ASAMALI VIZYON SISTEMI")
    print(f"  YOLO Model    : {args.weights}")
    print(f"  Kamera Acilma : {args.camera_arm_distance/100:.0f}m")
    print(f"  Conf Filtre   : >=0.55 (VisionThread ici)")
    print(f"  Bbox Filtre   : >=12000px2 (VisionThread ici)")
    print(f"  Center Tol    : 0.12 (dalis merkezleme)")
    print(f"  Dive Lock     : 4 kare (80ms)")
    print(f"  Kalman Pred.  : 150ms ilerisi")
    print("=" * 60)

    mission = KamikazeVisionMission(args)
    mission.run(hz=args.hz)
