"""
================================================================================
AVCI DRONE - SORT-Tarzı Çoklu CSRT Tracker Yönetimi
================================================================================
ClassySORT (YOLO + SORT) referansından ilham alınarak yazılmıştır.

Orijinal SORT, Kalman Filter ile tahmin yapar ve IoU ile eşleştirme yapar.
Bu modül aynı mantığı CSRT tracker ile uygular:
  - Kalman predict() yerine CSRT.update(frame) ile tahmin
  - IoU matris hesaplama + Hungarian algorithm ile optimal eşleştirme
  - Çoklu hedef yönetimi: her tracker'a ID, hit_streak, time_since_update

Referanslar:
  - SORT: Alex Bewley (2016) - Simple Online and Realtime Tracker
  - ClassySORT: Jason Sohn (2020) - YOLO + vanilla SORT

Lisans: Bu dosya GPL v3 lisanslı SORT/ClassySORT kaynak kodlarından
        türetilmiş yapısal tasarım kalıpları içerir.
================================================================================
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger("sort_csrt")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(_h)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def linear_assignment(cost_matrix):
    """Hungarian algorithm ile minimum maliyetli eşleştirme.
    lap kütüphanesi varsa onu kullanır (daha hızlı), yoksa scipy fallback."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    İki bbox grubu arası IoU hesaplama.
    bb_test: (N, 4) — [x1,y1,x2,y2]
    bb_gt:   (M, 4) — [x1,y1,x2,y2]
    Return:  (N, M) IoU matrisi
    """
    bb_gt = np.expand_dims(bb_gt, 0)     # (1, M, 4)
    bb_test = np.expand_dims(bb_test, 1) # (N, 1, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - wh

    iou = wh / (union + 1e-6)
    return iou


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.2):
    """
    Tespitleri tracker'lara atar (IoU tabanlı, Hungarian algorithm).

    Args:
        detections: (N, 4) bbox'lar [x1,y1,x2,y2]
        trackers:   (M, 4) tahmin bbox'ları [x1,y1,x2,y2]
        iou_threshold: eşik altındaki eşleşmeler reddedilir

    Returns:
        matches:             (K, 2) — [det_idx, trk_idx] çiftleri
        unmatched_detections: eşleşmeyen tespit indeksleri
        unmatched_trackers:   eşleşmeyen tracker indeksleri
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers)),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Düşük IoU eşleşmeleri filtrele
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# =============================================================================
# BBOX FORMAT DÖNÜŞÜMLERİ
# =============================================================================

def xyxy_to_xywh(xyxy):
    """[x1,y1,x2,y2] → [x,y,w,h] (CSRT format)"""
    x1, y1, x2, y2 = xyxy
    return (int(x1), int(y1), int(max(1.0, x2 - x1)), int(max(1.0, y2 - y1)))


def xywh_to_xyxy(xywh):
    """[x,y,w,h] → [x1,y1,x2,y2]"""
    x, y, w, h = xywh
    return [float(x), float(y), float(x + w), float(y + h)]


# =============================================================================
# TEK BİR CSRT TRACKER NESNESİ
# =============================================================================

class CSRTTracker:
    """Tek bir hedef için CSRT tracker + metadata.

    SORT'un KalmanBoxTracker sınıfına karşılık gelir, ancak
    Kalman predict yerine CSRT.update(frame) kullanır.
    """
    _id_counter = 0

    def __init__(self, bbox_xyxy, frame, confidence=0.0, min_hits=3, max_age=30):
        """
        Args:
            bbox_xyxy: [x1,y1,x2,y2] — ilk tespit kutusu
            frame: BGR görüntü (CSRT init için)
            confidence: YOLO confidence skoru
            min_hits: onaylama için gereken minimum ardışık hit
            max_age: güncellemesiz yaşatma süresi (frame)
        """
        self.csrt = self._create_csrt()
        xywh = xyxy_to_xywh(bbox_xyxy)
        self.csrt.init(frame, xywh)

        self.bbox_xyxy = list(bbox_xyxy)  # mevcut bbox [x1,y1,x2,y2]
        self.confidence = confidence
        self.time_since_update = 0
        self.id = CSRTTracker._id_counter
        CSRTTracker._id_counter += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.min_hits = min_hits
        self.max_age = max_age
        self._prediction_valid = True

    @staticmethod
    def _create_csrt():
        """OpenCV sürümüne uygun CSRT tracker oluşturur."""
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        raise RuntimeError("OpenCV CSRT tracker bulunamadı. opencv-contrib-python gerekebilir.")

    def update_with_detection(self, bbox_xyxy, frame, confidence=0.0):
        """Eşleşen YOLO tespiti ile tracker'ı güncelle (re-init)."""
        self.csrt = self._create_csrt()
        xywh = xyxy_to_xywh(bbox_xyxy)
        self.csrt.init(frame, xywh)
        self.bbox_xyxy = list(bbox_xyxy)
        self.confidence = confidence
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self._prediction_valid = True

    def predict(self, frame):
        """CSRT ile bir frame ileri tahmin yap.

        Returns:
            [x1,y1,x2,y2] veya tahmin başarısızsa None
        """
        self.age += 1
        self.time_since_update += 1

        if not self._prediction_valid:
            return None

        ok, xywh = self.csrt.update(frame)
        if not ok:
            self._prediction_valid = False
            return None

        bbox = xywh_to_xyxy(xywh)
        # Sınır kontrolü
        h, w = frame.shape[:2]
        bbox[0] = max(0.0, min(float(w - 1), bbox[0]))
        bbox[1] = max(0.0, min(float(h - 1), bbox[1]))
        bbox[2] = max(0.0, min(float(w - 1), bbox[2]))
        bbox[3] = max(0.0, min(float(h - 1), bbox[3]))

        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            self._prediction_valid = False
            return None

        self.bbox_xyxy = bbox
        # Güncelleme yokken confidence zamanla düşer
        self.confidence = max(0.15, self.confidence - 0.012)
        return bbox

    def get_state(self):
        """Mevcut tracker durumunu döndürür.

        Returns:
            dict: bbox_xyxy, confidence, id, hit_streak, time_since_update, is_confirmed
        """
        # Confirmation mantığı: yeterli hit streak VE max age aşılmamış
        is_confirmed = (
            self.hit_streak >= self.min_hits
            or (self.time_since_update == 0 and self.hits >= self.min_hits)
        )
        # Max age aşıldıysa onayı kaldır
        if self.time_since_update > self.max_age:
            is_confirmed = False
            self.hit_streak = 0

        return {
            "bbox_xyxy": tuple(self.bbox_xyxy),
            "confidence": self.confidence,
            "id": self.id,
            "hit_streak": self.hit_streak,
            "time_since_update": self.time_since_update,
            "is_confirmed": is_confirmed,
            "age": self.age,
        }

    def is_dead(self):
        """Tracker'ın silinmesi gerekiyor mu?"""
        return self.time_since_update > self.max_age or not self._prediction_valid


# =============================================================================
# ANA SORT-CSRT YÖNETİCİSİ
# =============================================================================

class SortCSRT:
    """SORT tarzı çoklu CSRT tracker yöneticisi.

    Her frame'de:
      1. Mevcut tracker'ları predict et (CSRT.update)
      2. Yeni YOLO tespitleri ile IoU association yap
      3. Eşleşenleri güncelle, eşleşmeyenleri yeni tracker olarak aç
      4. Ölü tracker'ları sil
      5. En iyi hedefi döndür
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.2,
                 use_gray_for_csrt=True):
        """
        Args:
            max_age: YOLO yokken CSRT tahminini yaşatma süresi (frame)
            min_hits: onay için gereken ardışık hit sayısı
            iou_threshold: association eşiği
            use_gray_for_csrt: CSRT'ye gri görüntü besle (OSD'ye daha dayanıklı)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_gray_for_csrt = use_gray_for_csrt
        self.trackers: list[CSRTTracker] = []
        self.frame_count = 0

    def _csrt_frame(self, frame):
        """CSRT için görüntü hazırla (opsiyonel gri dönüşüm)."""
        if not self.use_gray_for_csrt:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def update(self, frame, detections):
        """Ana güncelleme döngüsü.

        Args:
            frame: BGR görüntü
            detections: list of dict, her biri:
                {"bbox_xyxy": (x1,y1,x2,y2), "confidence": float}
                veya boş liste (tespit yok)

        Returns:
            list of dict: aktif (confirmed) tracker'ların durumları
                [{"bbox_xyxy", "confidence", "id", "hit_streak", ...}, ...]
        """
        self.frame_count += 1
        csrt_frame = self._csrt_frame(frame)

        # 1. Mevcut tracker'ları predict et
        predicted_bboxes = []
        to_delete = []
        for i, trk in enumerate(self.trackers):
            pred = trk.predict(csrt_frame)
            if pred is None:
                to_delete.append(i)
            else:
                predicted_bboxes.append(pred)

        # Ölü tracker'ları sil (tersten sil ki index kayması olmasın)
        for i in reversed(to_delete):
            self.trackers.pop(i)

        # Kalan tracker'ların predict sonuçlarını topla
        if len(self.trackers) != len(predicted_bboxes):
            # Uyumsuzluk: predict sonrasında silinenler var, yeniden topla
            predicted_bboxes = [trk.bbox_xyxy for trk in self.trackers]

        # 2. YOLO tespitlerini numpy'a çevir
        det_bboxes = np.array([d["bbox_xyxy"] for d in detections]) if detections else np.empty((0, 4))
        det_confs = [d["confidence"] for d in detections] if detections else []
        trk_bboxes = np.array(predicted_bboxes) if predicted_bboxes else np.empty((0, 4))

        # 3. Association
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_bboxes, trk_bboxes, self.iou_threshold
        )

        # 4. Eşleşenleri güncelle
        for m in matched:
            det_idx, trk_idx = int(m[0]), int(m[1])
            self.trackers[trk_idx].update_with_detection(
                detections[det_idx]["bbox_xyxy"],
                csrt_frame,
                confidence=det_confs[det_idx],
            )

        # 5. Eşleşmeyen tespitler → yeni tracker
        for det_idx in unmatched_dets:
            trk = CSRTTracker(
                detections[int(det_idx)]["bbox_xyxy"],
                csrt_frame,
                confidence=det_confs[int(det_idx)],
                min_hits=self.min_hits,
                max_age=self.max_age,
            )
            self.trackers.append(trk)

        # 6. Eşleşmeyen tracker'lar: hit_streak sıfırla
        for trk_idx in unmatched_trks:
            idx = int(trk_idx)
            if idx < len(self.trackers):
                self.trackers[idx].hit_streak = 0

        # 7. Ölü tracker'ları temizle
        self.trackers = [trk for trk in self.trackers if not trk.is_dead()]

        # 8. Sonuçları topla
        results = []
        for trk in self.trackers:
            state = trk.get_state()
            results.append(state)

        return results

    def get_best_target(self, results=None):
        """En yüksek güvenli hedefi döndür.

        Öncelik sırası:
          1. is_confirmed == True olan tracker'lar
          2. İçlerinden en yüksek (hit_streak * confidence) skoru
          3. Confirmed yoksa en yüksek confidence'lı unconfirmed

        Returns:
            dict veya None
        """
        if results is None:
            results = [trk.get_state() for trk in self.trackers]

        if not results:
            return None

        confirmed = [r for r in results if r["is_confirmed"]]
        if confirmed:
            return max(confirmed, key=lambda r: r["hit_streak"] * r["confidence"])

        # Confirmed yoksa en yüksek confidence
        return max(results, key=lambda r: r["confidence"])

    def reset(self):
        """Tüm tracker'ları sıfırla."""
        self.trackers.clear()
        self.frame_count = 0

    @property
    def active_tracker_count(self):
        return len(self.trackers)
