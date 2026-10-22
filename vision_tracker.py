"""
================================================================================
AVCI DRONE - GORSEL HEDEF TESPITI v2 (YOLO + SortCSRT Multi-Tracker)
================================================================================
v1'den FARKLAR:
  1. Tek CSRT tracker yerine SortCSRT (SORT-tarzı çoklu CSRT yönetimi)
     - IoU tabanlı association ile doğru eşleştirme
     - Çoklu tracker yönetimi ile kısa kayıplarda hedef yaşıyor
     - Her tracker'a ID, hit_streak, time_since_update
  2. _detect_dark_blob filtreleri sıkılaştırıldı (false positive azaltma)
  3. Route prediction bbox boyutu (area_ratio) tahmini eklendi
  4. confirm_streak_required: 5 → 4
  5. lost_streak_to_clear: 8 → 12

Bagimliliklar:
  pip install ultralytics opencv-python scipy --break-system-packages

Kullanim (standalone test):
  python vision_tracker.py --weights talon_v7.pt --camera-index 1 --show
================================================================================
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from sort_csrt import SortCSRT
from kalman_filter import KalmanFilter

logger = logging.getLogger("vision_tracker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(_h)


# =============================================================================
# AYARLAR
# =============================================================================
@dataclass
class VisionConfig:
    weights_path: str = "talon_v7.pt"
    camera_index: int = 1          # OBS Virtual Camera genelde 0'dan sonraki bir
                                    # index'tir (dahili webcam varsa 0 odur).
    capture_width: int = 1280
    capture_height: int = 720
    inference_size: int = 640      # YOLO'ya verilecek kare boyutu (kare, piksel)
    confidence_threshold: float = 0.55   # bu altindaki tespitler yok sayilir
    iou_threshold: float = 0.45
    device: Optional[str] = None    # None=Ultralytics otomatik, "cpu" veya CUDA icin "0"
    half: bool = False              # CUDA'da FP16 hizlandirma; CPU'da acmayin

    # --- Dogrulama (confirmation) ---
    confirm_streak_required: int = 2    # v4: hizli kilitlenme
    lost_streak_to_clear: int = 150      # v4: tracker yasarken kilit kopmasini engelle (uzun)

    # --- Performans ---
    inference_loop_hz: float = 30.0     # kamera 30fps varsayimi

    # --- YOLO + SortCSRT hibrit takip ---
    enable_csrt_tracking: bool = True
    detector_interval_frames: int = 8   # SortCSRT kilitliyken YOLO'yu her N karede bir tazele
    tracker_max_age_frames: int = 150    # v4: CSRT kutusunu daha uzun yasat (kopmayi engeller)
    tracker_min_hits: int = 3           # onay için gereken ardışık hit
    tracker_iou_threshold: float = 0.2  # association eşiği
    use_gray_for_csrt: bool = True      # CSRT'yi gri goruntuyle besle; OSD/renk degisimine daha sakin
    enable_click_roi: bool = True       # Pencerede hedefe tiklayarak CSRT ROI baslat
    manual_roi_size: int = 86           # Tiklanan nokta etrafindaki kare ROI boyutu
    gray_preview: bool = False          # Debug penceresinde gri goruntu goster
    route_prediction_sec: float = 0.15  # v3 gibi hedefin kisa sure sonraki rotasina uc
    max_route_predict_frames: int = 150  # v4: kisa kayiplarda rotayi uzun sure koru

    # --- Gorsel hata ayiklama penceresi ---
    show_window: bool = False    # True ise cv2.imshow ile tespit kutusu cizili goruntu acilir
    window_name: str = "Avci Drone - Tespit v2"


# =============================================================================
# TEK KARE TESPIT SONUCU
# =============================================================================
@dataclass
class DetectionResult:
    found: bool
    confidence: float = 0.0
    bbox_xyxy: Optional[tuple] = None       # (x1,y1,x2,y2) piksel, orijinal kare boyutunda
    bbox_center_norm: Optional[tuple] = None  # (cx,cy) normalize [-1,1] (0,0=goruntu merkezi)
    bbox_area_ratio: Optional[float] = None   # bbox alani / kare alani; kaba yakinlik sinyali
    frame_shape: Optional[tuple] = None     # (h,w)
    tracker_id: Optional[int] = None        # v2: SortCSRT tracker ID'si


@dataclass
class VisionResult:
    found: bool = False
    confirmed: bool = False
    confidence: float = 0.0
    bbox_xyxy: Optional[tuple] = None
    kf_cx: float = 0.0
    kf_cy: float = 0.0
    kf_vcx: float = 0.0
    kf_vcy: float = 0.0
    kf_valid: bool = False


def _to_normalized_center(bbox_xyxy, frame_w, frame_h):
    """Bbox merkezini goruntu merkezine gore [-1,1] araliginda normalize eder."""
    x1, y1, x2, y2 = bbox_xyxy
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0
    cx_norm = (cx_px - frame_w / 2.0) / (frame_w / 2.0)
    cy_norm = (cy_px - frame_h / 2.0) / (frame_h / 2.0)
    return (cx_norm, cy_norm)


def _draw_overlay(frame, detection, confirmed: bool):
    """Tespit kutusunu, guven skorunu, merkez noktasini ve onay durumunu
    goruntu uzerine cizer."""
    h, w = frame.shape[:2]
    # Goruntu merkezi referans cizgileri
    cv2.line(frame, (w // 2, 0), (w // 2, h), (90, 90, 90), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (90, 90, 90), 1)

    if detection.found and detection.bbox_xyxy is not None:
        x1, y1, x2, y2 = (int(v) for v in detection.bbox_xyxy)
        color = (0, 255, 0) if confirmed else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tid = detection.tracker_id if detection.tracker_id is not None else "?"
        label = f"hedef#{tid} {detection.confidence:.2f}" + (" [ONAYLI]" if confirmed else "")
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cx_px = int((x1 + x2) / 2)
        cy_px = int((y1 + y2) / 2)
        cv2.circle(frame, (cx_px, cy_px), 5, color, -1)
    else:
        cv2.putText(frame, "tespit yok", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    status_text = "DURUM: ONAYLI (VISUAL_TRACK)" if confirmed else "DURUM: ARANIYOR (SEARCH/GNSS)"
    status_color = (0, 255, 0) if confirmed else (0, 165, 255)
    cv2.putText(frame, status_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    return frame


# =============================================================================
# DOGRULAMA SAYACI (confirmation streak) - histerezis ile
# =============================================================================
class DetectionConfirmer:
    """Ardisik kare tespitlerini sayar; confirm/lost esiklerine gore
    'hedef onaylandi mi' durumunu histerezis ile tutar."""

    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg
        self._hit_streak = 0
        self._miss_streak = 0
        self._confirmed = False
        self.last_detection: Optional[DetectionResult] = None

    def update(self, detection: DetectionResult) -> bool:
        self.last_detection = detection
        if detection.found:
            self._hit_streak += 1
            self._miss_streak = 0
            if self._hit_streak >= self.cfg.confirm_streak_required:
                self._confirmed = True
        else:
            self._miss_streak += 1
            self._hit_streak = 0
            if self._miss_streak >= self.cfg.lost_streak_to_clear:
                self._confirmed = False
        return self._confirmed

    @property
    def confirmed(self) -> bool:
        return self._confirmed


# =============================================================================
# ANA SINIF: VISION TRACKER v2
# =============================================================================
class VisionTracker:
    """OBS Virtual Camera'dan goruntu okuyup arka planda surekli YOLO + SortCSRT
    tespiti calistiran thread tabanli sinif."""

    def __init__(self, config: VisionConfig = None):
        self.cfg = config or VisionConfig()
        self.confirmer = DetectionConfirmer(self.cfg)
        self._model = None
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._last_result = DetectionResult(found=False)
        self._last_display_frame = None
        self._frame_index = 0
        self._mouse_callback_ready = False
        self._pending_manual_roi = None
        self._last_frame_shape = None

        # v2: SortCSRT çoklu tracker yöneticisi
        self._sort_tracker = SortCSRT(
            max_age=self.cfg.tracker_max_age_frames,
            min_hits=self.cfg.tracker_min_hits,
            iou_threshold=self.cfg.tracker_iou_threshold,
            use_gray_for_csrt=self.cfg.use_gray_for_csrt,
        ) if self.cfg.enable_csrt_tracking else None

        # v4: Kalman Filter for stable bounding box center tracking
        dt = 1.0 / self.cfg.inference_loop_hz
        self._kalman = KalmanFilter(dt=dt, INIT_POS_STD=0.1, INIT_VEL_STD=0.1, ACCEL_STD=0.8, GPS_POS_STD=0.05)
        self._kalman_initialized = False
        
        # Route prediction state
        self._route_last_t = None
        self._route_miss_frames = 0
        self._route_last_area = None
        self._route_last_shape = None
        self._route_area_velocity = 0.0  # v2: bbox boyutu tahmini

    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or not self.cfg.enable_click_roi:
            return
        with self._lock:
            shape = self._last_frame_shape
            if shape is None:
                return
            h, w = shape[:2]
            size = int(max(24, self.cfg.manual_roi_size))
            half = size // 2
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(w - 1, x + half)
            y2 = min(h - 1, y + half)
            if x2 <= x1 or y2 <= y1:
                return
            self._pending_manual_roi = (float(x1), float(y1), float(x2), float(y2))
        logger.info("Manual ROI secildi: center=(%d,%d) bbox=(%d,%d,%d,%d)", x, y, x1, y1, x2, y2)

    # --- OSD korumasi ---
    def _osd_rectangles(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        return [
            # Alt merkez ARM/TRIGGER yazisi ve onun altindaki sahte kutu bolgesi.
            (int(0.24 * w), int(0.50 * h), int(0.82 * w), int(0.86 * h)),
            (0, int(0.70 * h), int(0.30 * w), h),
            (int(0.78 * w), 0, w, int(0.27 * h)),
            (int(0.20 * w), 0, int(0.80 * w), int(0.13 * h)),
            (int(0.72 * w), int(0.58 * h), w, int(0.82 * h)),
        ]

    @staticmethod
    def _intersection_ratio(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = max(1.0, max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1))
        return inter / area

    def _mask_osd_for_detection(self, frame):
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in self._osd_rectangles(w, h):
            frame[y1:y2, x1:x2] = 0
        return frame

    def _is_osd_false_positive(self, xyxy: tuple[float, float, float, float], w: int, h: int) -> bool:
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            return True
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = max(0.0, x2 - x1) / max(1, w)
        bh = max(0.0, y2 - y1) / max(1, h)
        box = (x1, y1, x2, y2)
        if any(self._intersection_ratio(box, rect) >= 0.12 for rect in self._osd_rectangles(w, h)):
            return True
        # Alt yarida cok kucuk/orta kutular genelde OSD, zemin lekesi veya imlec oluyor.
        if cy > 0.48 * h and max(bw, bh) >= 0.010:
            return True
        return False

    def _candidate_score(self, xyxy, conf: float, w: int, h: int) -> float:
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) * 0.5 / max(1, w)
        cy = (y1 + y2) * 0.5 / max(1, h)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, float(w * h))
        center_bonus = 0.18 * (1.0 - min(1.0, abs(cx - 0.5) / 0.5))
        sky_bonus = 0.14 if cy < 0.52 else 0.0
        size_bonus = min(0.12, area * 30.0)
        return conf + center_bonus + sky_bonus + size_bonus

    def _detect_dark_blob(self, frame) -> DetectionResult:
        """v2: Sıkılaştırılmış dark blob tespiti — false positive azaltma."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_mask = gray.copy()
        allowed = cv2.inRange(search_mask, 0, 255)
        allowed[:, :] = 0
        # v2: arama bölgesi daraltıldı (0.14-0.48 arası, v1'de 0.12-0.55)
        allowed[int(0.14 * h):int(0.48 * h), int(0.12 * w):int(0.90 * w)] = 255
        allowed[0:int(0.28 * h), 0:int(0.25 * w)] = 0
        allowed[0:int(0.30 * h), int(0.78 * w):w] = 0
        for x1, y1, x2, y2 in self._osd_rectangles(w, h):
            allowed[y1:y2, x1:x2] = 0

        candidates = []
        # v2: threshold aralığı daraltıldı (130-150, v1'de 130-160)
        for threshold in (130, 140, 150):
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            binary = cv2.bitwise_and(binary, allowed)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                area_px = bw * bh
                # v2: minimum alan yükseltildi (45→65), maximum düşürüldü (1400→1100)
                if area_px < 65 or area_px > 1100:
                    continue
                aspect = bw / max(1, bh)
                # v2: aspect ratio sıkılaştırıldı (1.8-9.5 → 2.0-8.0)
                if aspect < 2.0 or aspect > 8.0:
                    continue
                xyxy = (float(x), float(y), float(x + bw), float(y + bh))
                if self._is_osd_false_positive(xyxy, w, h):
                    continue
                cx = (x + bw * 0.5) / max(1, w)
                cy = (y + bh * 0.5) / max(1, h)
                if cx < 0.25 or cy < 0.14 or cy > 0.48:
                    continue
                area_ratio = area_px / max(1.0, float(w * h))
                if cy > 0.42 and cx < 0.62 and area_ratio < 0.00020:
                    continue
                if cy > 0.38 and cx > 0.88:
                    continue
                darkness = max(0.0, threshold - float(gray[y:y + bh, x:x + bw].mean()))
                score = area_px * 0.010 + aspect * 0.08 + darkness * 0.010 + (1.0 - abs(cx - 0.58)) * 0.20
                candidates.append((score, xyxy, area_px))

        if not candidates:
            return DetectionResult(found=False, frame_shape=(h, w))

        _, xyxy, _ = max(candidates, key=lambda item: item[0])
        x1, y1, x2, y2 = xyxy
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
        # v2: dark blob confidence düşürüldü (0.32 → 0.20) — daha düşük güven
        return DetectionResult(
            found=True,
            confidence=0.20,
            bbox_xyxy=xyxy,
            bbox_center_norm=_to_normalized_center(xyxy, w, h),
            bbox_area_ratio=area_ratio,
            frame_shape=(h, w),
        )

    # --- Kurulum ---
    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "ultralytics paketi kurulu degil. Kurulum: "
                "pip install ultralytics --break-system-packages"
            ) from e
        logger.info("YOLO modeli yukleniyor: %s", self.cfg.weights_path)
        self._model = YOLO(self.cfg.weights_path)
        logger.info("Model yuklendi. Siniflar: %s", self._model.names)

    def _open_camera(self):
        logger.info("OBS Virtual Camera aciliyor (index=%d, DirectShow)...",
                     self.cfg.camera_index)
        cap = cv2.VideoCapture(self.cfg.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"Kamera acilamadi (index={self.cfg.camera_index}). "
                f"OBS Virtual Camera 'Start Virtual Camera' ile baslatildi mi? "
                f"Index dogru mu (0,1,2... deneyin)?"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.capture_height)
        self._cap = cap
        logger.info("Kamera acildi: %dx%d", self.cfg.capture_width, self.cfg.capture_height)

    # --- Tek kare YOLO tespiti ---
    def _detect_yolo(self, frame) -> list[dict]:
        """YOLO ile tespit yap. Sonuçları dict listesi olarak döndür.

        Returns:
            list of {"bbox_xyxy": tuple, "confidence": float}
        """
        h, w = frame.shape[:2]
        predict_kwargs = {
            "imgsz": self.cfg.inference_size,
            "conf": self.cfg.confidence_threshold,
            "iou": self.cfg.iou_threshold,
            "verbose": False,
        }
        if self.cfg.device:
            predict_kwargs["device"] = self.cfg.device
        if self.cfg.half:
            predict_kwargs["half"] = True

        infer_frame = self._mask_osd_for_detection(frame.copy())
        results = self._model.predict(infer_frame, **predict_kwargs)

        if not results or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes
        confidences = boxes.conf.tolist()
        xyxys = boxes.xyxy.tolist()
        detections = []
        for conf, raw_xyxy in zip(confidences, xyxys):
            xyxy_i = tuple(float(v) for v in raw_xyxy)
            if self._is_osd_false_positive(xyxy_i, w, h):
                continue
            detections.append({
                "bbox_xyxy": xyxy_i,
                "confidence": float(conf),
            })

        return detections

    # --- Ana tespit döngüsü (SortCSRT entegrasyonu) ---
    def _detect_one_frame(self, frame) -> DetectionResult:
        """Her karede çağrılır. YOLO + SortCSRT ile tespit/takip yapar."""
        self._frame_index += 1
        h, w = frame.shape[:2]
        with self._lock:
            self._last_frame_shape = frame.shape
            pending_manual_roi = self._pending_manual_roi
            self._pending_manual_roi = None

        # YOLO çalıştırma kararı
        should_run_yolo = (
            self._sort_tracker is None
            or self._sort_tracker.active_tracker_count == 0
            or self._frame_index % max(1, self.cfg.detector_interval_frames) == 0
        )

        yolo_detections = []
        if should_run_yolo:
            yolo_detections = self._detect_yolo(frame)

        if pending_manual_roi is not None:
            yolo_detections.insert(0, {
                "bbox_xyxy": pending_manual_roi,
                "confidence": 0.88,
            })
            self._kalman_initialized = False
            self._route_miss_frames = 0

        # SortCSRT entegrasyonu
        if self._sort_tracker is not None:
            # SortCSRT.update: hem YOLO tespitleri hem CSRT tahminleri
            tracker_results = self._sort_tracker.update(frame, yolo_detections)
            best = self._sort_tracker.get_best_target(tracker_results)

            if best is not None and best.get("bbox_xyxy"):
                xyxy = best["bbox_xyxy"]
                x1, y1, x2, y2 = xyxy
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(w * h))
                detection = DetectionResult(
                    found=True,
                    confidence=best["confidence"],
                    bbox_xyxy=xyxy,
                    bbox_center_norm=_to_normalized_center(xyxy, w, h),
                    bbox_area_ratio=area_ratio,
                    frame_shape=(h, w),
                    tracker_id=best["id"],
                )
                return self._stabilize_route(detection)

        # SortCSRT yoksa veya tracker bulamadıysa: dark blob fallback IPTAL EDILDI (V2 Kamikaze Revizyonu)
        # Sadece YOLO'ya guvenilecek, sahte kilitlenmeleri (calilar/golgeler) onlemek amaciyla kaldirildi.
        # if should_run_yolo and not yolo_detections:
        #     dark_blob = self._detect_dark_blob(frame)
        #     if dark_blob.found:
        #         # Dark blob sonucunu da SortCSRT'ye besle (düşük conf ile)
        #         if self._sort_tracker is not None:
        #             self._sort_tracker.update(frame, [{
        #                 "bbox_xyxy": dark_blob.bbox_xyxy,
        #                 "confidence": dark_blob.confidence,
        #             }])
        #         return self._stabilize_route(dark_blob)

        return self._stabilize_route(DetectionResult(found=False, frame_shape=(h, w)))

    def _clamp_center(self, center: tuple[float, float]) -> tuple[float, float]:
        return (max(-1.0, min(1.0, center[0])), max(-1.0, min(1.0, center[1])))

    def _stabilize_route(self, detection: DetectionResult) -> DetectionResult:
        """Route prediction: Kalman Filter ile merkez tahmini + bbox boyutu tahmini."""
        now = time.time()
        dt = 1.0 / max(1.0, self.cfg.inference_loop_hz) if self._route_last_t is None else max(1e-3, min(0.12, now - self._route_last_t))
        self._route_last_t = now
        
        # update kalman dt
        self._kalman.dt = dt
        self._kalman.F[0, 2] = dt
        self._kalman.F[1, 3] = dt
        self._kalman.L[0, 0] = 0.5 * dt**2
        self._kalman.L[1, 1] = 0.5 * dt**2
        self._kalman.L[2, 0] = dt
        self._kalman.L[3, 1] = dt
        self._kalman.Q = np.dot(self._kalman.L, np.dot(self._kalman.q, self._kalman.L.T))

        if detection.found and detection.bbox_center_norm is not None:
            meas = detection.bbox_center_norm
            
            if not self._kalman_initialized:
                self._kalman.x = np.array([[meas[0]], [meas[1]], [0.0], [0.0]])
                self._kalman_initialized = True
                self._route_miss_frames = 0
            
            # Predict
            self._kalman.predict()
            
            # Update with measurement
            z = np.array([[meas[0]], [meas[1]]])
            cx, cy = self._kalman.update(z)
            
            self._route_miss_frames = 0
            
            # bbox boyutu tahmini
            if detection.bbox_area_ratio is not None:
                if self._route_last_area is not None:
                    area_velocity = (detection.bbox_area_ratio - self._route_last_area) / dt
                    self._route_area_velocity = 0.80 * self._route_area_velocity + 0.20 * area_velocity
                self._route_last_area = detection.bbox_area_ratio

            self._route_last_shape = detection.frame_shape
            
            # Lead point
            lead = self.cfg.route_prediction_sec
            guided_center = self._clamp_center((
                cx + float(self._kalman.x[2, 0]) * lead,
                cy + float(self._kalman.x[3, 0]) * lead,
            ))
            detection.bbox_center_norm = guided_center
            
            if self._route_last_area is not None:
                predicted_area = max(0.0, self._route_last_area + self._route_area_velocity * lead)
                detection.bbox_area_ratio = predicted_area

            return detection

        # Eger hedef kayipsa (bulunamadiysa) Kalman tahminiyle idare et
        if not self._kalman_initialized or self._route_miss_frames >= self.cfg.max_route_predict_frames:
            return detection

        self._route_miss_frames += 1
        
        # Sadece predict (olcum yok)
        cx, cy = self._kalman.predict()
        
        lead = self.cfg.route_prediction_sec
        guided_center = self._clamp_center((
            cx + float(self._kalman.x[2, 0]) * lead,
            cy + float(self._kalman.x[3, 0]) * lead,
        ))

        predicted_area = self._route_last_area
        if predicted_area is not None:
            predicted_area = max(0.0, predicted_area + self._route_area_velocity * dt)
            self._route_last_area = predicted_area

        return DetectionResult(
            found=True,
            confidence=0.22,
            bbox_center_norm=guided_center,
            bbox_area_ratio=predicted_area,
            frame_shape=self._route_last_shape or detection.frame_shape,
        )

    # --- Arka plan dongusu ---
    def _run_loop(self):
        dt_target = 1.0 / self.cfg.inference_loop_hz
        while self._running:
            try:
                loop_start = time.time()
                ok, frame = self._cap.read()
                if not ok:
                    logger.warning("Kameradan kare okunamadi, tekrar deneniyor...")
                    time.sleep(0.1)
                    continue

                detection = self._detect_one_frame(frame)
                confirmed = self.confirmer.update(detection)

                with self._lock:
                    self._last_result = detection

                if detection.found:
                    tid = detection.tracker_id if detection.tracker_id is not None else "?"
                    logger.info(
                        "Tespit: conf=%.2f trk#%s bbox_merkez_norm=(%.2f,%.2f) onayli=%s",
                        detection.confidence, tid,
                        detection.bbox_center_norm[0],
                        detection.bbox_center_norm[1], confirmed,
                    )

                # --- Gorsel hata ayiklama penceresi (Main Thread icin kaydet) ---
                if self.cfg.show_window:
                    display_src = frame
                    if self.cfg.gray_preview:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        display_src = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    display_frame = _draw_overlay(display_src.copy(), detection, confirmed)
                    with self._lock:
                        self._last_display_frame = display_frame

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt_target - elapsed))
            except Exception as e:
                import traceback
                logger.error("VisionTracker arka plan dongusunde hata: %s\n%s", e, traceback.format_exc())
                time.sleep(0.5)

    # --- Disa acik arayuz ---
    def start(self):
        """Modeli ve kamerayi yukler, arka plan tespit thread'ini baslatir."""
        self._load_model()
        self._open_camera()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("VisionTracker v2 baslatildi (%.0f Hz, SortCSRT: %s).",
                     self.cfg.inference_loop_hz,
                     "AKTIF" if self._sort_tracker else "KAPALI")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        logger.info("VisionTracker v2 durduruldu.")

    def update_window(self) -> bool:
        """Ana thread'den cagrilmalidir. Pencereyi gunceller ve 'q' tusunu kontrol eder.
        Return False if 'q' is pressed, else True."""
        if not self.cfg.show_window:
            return True
            
        with self._lock:
            frame = self._last_display_frame
            
        if frame is not None:
            cv2.imshow(self.cfg.window_name, frame)
            if self.cfg.enable_click_roi and not self._mouse_callback_ready:
                cv2.setMouseCallback(self.cfg.window_name, self._on_mouse)
                self._mouse_callback_ready = True
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("'q' tusuna basildi, kapatiliyor.")
                self._running = False
                return False
        return True

    def is_target_visible(self) -> bool:
        """guidance_controller.run_guidance_loop(vision_override_check=...)
        parametresine DIREKT verilecek fonksiyon. True -> GNSS yonelmesi
        kesilir (hedef kamerada onaylandi)."""
        return self.confirmer.confirmed

    def get_last_detection(self) -> DetectionResult:
        with self._lock:
            return self._last_result


class VisionThread:
    """angajman_mix.py uyumluluk katmani.

    Eski kamikaze_vision_v3.VisionThread arayuzunu VisionTracker/SortCSRT
    uzerine tasir. Mouse ile ROI secimi ve gri CSRT takibi bu katmanda aktiftir.
    """

    def __init__(self, weights: str, window_name: str = "", conf: float = 0.45,
                 lost_streak: int = 8, use_obs: bool = False, camera_index: int = 1,
                 show_window: bool = False, confirm_window: int = 10,
                 confirm_ratio: float = 0.4, enable_click_roi: bool = True,
                 gray_preview: bool = False, manual_roi_size: int = 86):
        self._found_history = []
        self._miss_conf = 0
        self._confirm_window = max(1, int(confirm_window))
        self._confirm_ratio = float(confirm_ratio)

        cfg = VisionConfig(
            weights_path=weights,
            camera_index=camera_index,
            confidence_threshold=conf,
            confirm_streak_required=max(2, int(round(self._confirm_window * self._confirm_ratio))),
            lost_streak_to_clear=max(int(lost_streak), 30),
            show_window=show_window,
            window_name=window_name or "Avci Drone - Kamera Tespiti",
            enable_click_roi=enable_click_roi,
            gray_preview=gray_preview,
            manual_roi_size=manual_roi_size,
        )
        self._tracker = VisionTracker(cfg)

    def start(self):
        self._tracker.start()

    def stop(self):
        self._tracker.stop()

    def _to_result(self, det: DetectionResult, confirmed: bool) -> VisionResult:
        cx, cy = (0.0, 0.0)
        if det.bbox_center_norm is not None:
            cx, cy = det.bbox_center_norm
        vx = float(self._tracker._kalman.x[2, 0]) if self._tracker._kalman_initialized else 0.0
        vy = float(self._tracker._kalman.x[3, 0]) if self._tracker._kalman_initialized else 0.0
        return VisionResult(
            found=det.found,
            confirmed=confirmed,
            confidence=det.confidence,
            bbox_xyxy=det.bbox_xyxy,
            kf_cx=float(cx),
            kf_cy=float(cy),
            kf_vcx=vx,
            kf_vcy=vy,
            kf_valid=det.found and det.bbox_center_norm is not None,
        )

    def get(self) -> VisionResult:
        self._tracker.update_window()
        det = self._tracker.get_last_detection()
        confirmed = self._tracker.is_target_visible()

        self._found_history.append(bool(det.found))
        if len(self._found_history) > self._confirm_window:
            self._found_history.pop(0)
        self._miss_conf = 0 if det.found else self._miss_conf + 1

        return self._to_result(det, confirmed)


# =============================================================================
# STANDALONE TEST (kamera + model gerektirir)
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vision tracker v2 standalone test")
    parser.add_argument("--weights", default="talon_v7.pt")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO inference boyutu. FPS dusukse 320 deneyin.")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Tespit guven esigi.")
    parser.add_argument("--device", default=None,
                        help='Ultralytics device: "cpu" veya CUDA icin "0".')
    parser.add_argument("--half", action="store_true",
                        help="CUDA'da FP16 kullan. CPU'da acmayin.")
    parser.add_argument("--show", action="store_true", help="OpenCV pencerede goster (debug)")
    args = parser.parse_args()

    cfg = VisionConfig(
        weights_path=args.weights,
        camera_index=args.camera_index,
        inference_size=args.imgsz,
        confidence_threshold=args.conf,
        device=args.device,
        half=args.half,
        show_window=args.show,
    )
    tracker = VisionTracker(cfg)
    tracker.start()

    logger.info("Cikmak icin Ctrl+C (ya da --show ile acilan pencerede 'q' tusu).")
    try:
        while tracker._running:
            time.sleep(0.02)
            if not tracker.update_window():
                break
            
            det = tracker.get_last_detection()
            visible = tracker.is_target_visible()
            logger.info("is_target_visible()=%s | son_tespit_found=%s conf=%.2f trk#%s",
                         visible, det.found, det.confidence,
                         det.tracker_id if det.tracker_id is not None else "?")
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()


if __name__ == "__main__":
    main()
