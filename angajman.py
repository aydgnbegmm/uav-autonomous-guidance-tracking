"""
================================================================================
AVCI DRONE - GOREV STATE MACHINE v2 (TAKEOFF -> SEARCH, opsiyonel VISUAL_TRACK)
================================================================================
v1'den FARKLAR:
  1. _visual_lost_grace_sec: 1.20 → 2.0 saniye (CSRT max_age ile uyumlu)
  2. SEARCH → VISUAL_TRACK geçişinde PID state'lerini sıfırlamak yerine
     GNSS komutlarından seed etme (throttle, pitch, yaw)
  3. VISUAL_TRACK → SEARCH geçişinde son visual komutları ile
     GuidanceController'a seed_full_state çağrısı
  4. confirm_streak_required: 5 → 4, lost_streak_to_clear: 8 → 12
     (vision_tracker.py VisionConfig'te ayarlandı)
  5. SortCSRT multi-tracker entegrasyonu (vision_tracker.py v2)

State gecisleri:
       TAKEOFF  -->  SEARCH  [--visual-track aciksa <--> VISUAL_TRACK]
       (yer)    (kalkis     (hedef kamerada)
                 irtifasina
                 ulasinca)
================================================================================
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum

import drone_sdk as drone
from kamikaze_gnss_controller import KamikazeGNSSController

logger = logging.getLogger("mission_fsm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(_h)

CM_TO_M = 0.01


class MissionState(Enum):
    TAKEOFF      = "TAKEOFF"        # Yerden hedef irtifasina tirmanma
    SEARCH       = "SEARCH"         # GNSS yonelmesi aktif
    VISUAL_TRACK = "VISUAL_TRACK"   # Gorsel PID aktif


# =============================================================================
# TAKEOFF KONFIGURASYON
# =============================================================================
@dataclass
class TakeoffConfig:
    # Hedef irtifaya ne kadar yaklasinca "ulastim" sayilsin (metre)
    altitude_tolerance_m: float = 8.0

    # Throttle rampasi: her adimda ne kadar artir (50Hz dongu -> 0.01/adim = 0.5/sn)
    throttle_ramp_rate: float = 0.01

    # Maksimum takeoff throttle'i (hover + biraz fazlasi)
    throttle_max: float = 0.85

    # Hover throttle tahmini.
    # README: throttle=0.0 = hover. Eski deger 0.62 YANLIS'ti.
    hover_throttle: float = 0.0

    # Takeoff sirasinda irtifa hedefine gore P kazanci
    kp_altitude: float = 0.012

    # Takeoff'ta yaw ve pitch sifir (sadece yukari cik)
    lock_yaw_pitch: bool = True


# =============================================================================
# TAKEOFF KONTROLCUSU
# =============================================================================
class TakeoffController:
    """Drone'u yerden hedef irtifasina tirmandirir.
    Hedef irtifaya ulasinca 'done' flag'i True olur."""

    def __init__(self, config: TakeoffConfig = None):
        self.cfg = config or TakeoffConfig()
        self._current_throttle = 0.0
        self._done = False

    def reset(self):
        self._current_throttle = 0.0
        self._done = False

    @property
    def done(self):
        return self._done

    def step(self, current_altitude_m: float, target_altitude_m: float) -> dict:
        cfg = self.cfg
        alt_error = target_altitude_m - current_altitude_m

        # Hedefe ulastik mi?
        if alt_error <= cfg.altitude_tolerance_m and current_altitude_m >= (target_altitude_m * 0.85):
            self._done = True
            # Hover'a yumsat
            cmd_throttle = cfg.hover_throttle
        else:
            # P kontrol + hover offset
            p_term = cfg.kp_altitude * alt_error
            target_throttle = cfg.hover_throttle + p_term

            # Rampali artis (ani throttle spike'i onle)
            if target_throttle > self._current_throttle:
                self._current_throttle = min(
                    target_throttle,
                    self._current_throttle + cfg.throttle_ramp_rate,
                )
            else:
                # Iniste daha hizli tepki ver
                self._current_throttle = target_throttle

            cmd_throttle = max(0.0, min(cfg.throttle_max, self._current_throttle))

        return {
            "throttle": cmd_throttle,
            "pitch":    0.0,
            "roll":     0.0,
            "yaw":      0.0,
            "alt_error_m":      alt_error,
            "current_alt_m":    current_altitude_m,
            "target_alt_m":     target_altitude_m,
            "done":             self._done,
        }


# =============================================================================
# GORSEL PID KONFIGURASYON
# =============================================================================
@dataclass
class PidConfig:
    # yaw PID
    kp_yaw: float = 0.75
    ki_yaw: float = 0.02  # Hizalama icin kucuk integral
    kd_yaw: float = 0.10
    yaw_command_max: float = 0.80

    # pitch PID (kamikaze dive)
    kp_pitch: float = 0.65
    ki_pitch: float = 0.02  # Hizalama icin kucuk integral
    kd_pitch: float = 0.10
    pitch_command_max: float = 0.85

    kamikaze_throttle: float = -0.90  # Gerçek Kamikaze dalışı (Tam güç burun aşağı!)
    command_smoothing_alpha: float = 0.4


class SimplePid:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._integral = 0.0
        self._prev_error = None
        self._prev_t = None

    def update(self, error: float, t: float) -> float:
        dt = 0.0 if self._prev_t is None else max(1e-3, t - self._prev_t)
        self._integral += error * dt
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        self._prev_t = t
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._prev_t = None


class VisualTrackController:
    def __init__(self, config: PidConfig = None):
        self.cfg = config or PidConfig()
        self.yaw_pid = SimplePid(self.cfg.kp_yaw, self.cfg.ki_yaw, self.cfg.kd_yaw)
        self.pitch_pid = SimplePid(self.cfg.kp_pitch, self.cfg.ki_pitch, self.cfg.kd_pitch)
        self._prev_yaw_cmd = None
        self._prev_pitch_cmd = None
        self._prev_throttle_cmd = None

    def reset(self):
        self.yaw_pid.reset()
        self.pitch_pid.reset()
        self._prev_yaw_cmd = None
        self._prev_pitch_cmd = None
        self._prev_throttle_cmd = None

    def seed_from_gnss(self, throttle: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        """v2: SEARCH → VISUAL_TRACK geçişinde GNSS komutlarından seed et.
        Böylece PID state'leri sıfırlanmak yerine GNSS'in son değerlerinden
        yumuşak bir geçiş yapılır."""
        self._prev_throttle_cmd = throttle
        self._prev_pitch_cmd = pitch
        self._prev_yaw_cmd = yaw

    def _smooth(self, prev, value):
        if prev is None:
            return value
        alpha = max(0.0, min(1.0, self.cfg.command_smoothing_alpha))
        return prev + (value - prev) * alpha

    def compute_command(self, bbox_center_norm: tuple, t: float = None,
                        altitude_error_m: float = 0.0,
                        bbox_area_ratio: float = None) -> dict:
        cfg = self.cfg
        t = time.time() if t is None else t
        cx, cy = bbox_center_norm

        # YAW (Sağ/Sol) KONTROLÜ
        yaw_cmd = self.yaw_pid.update(cx, t)
        yaw_cmd = max(-cfg.yaw_command_max, min(cfg.yaw_command_max, yaw_cmd))
        yaw_cmd = self._smooth(self._prev_yaw_cmd, yaw_cmd)
        if abs(yaw_cmd) < 0.01:
            yaw_cmd = 0.0
        self._prev_yaw_cmd = yaw_cmd

        # Z-EKSENİ (THROTTLE) KONTROLÜ (Aşağı/Yukarı Dalış)
        # cy > 0 (hedef aşağıda) -> Aşağı inmek için negatif throttle
        # cy < 0 (hedef yukarıda) -> Yukarı çıkmak için pozitif throttle
        # pitch_pid'yi burada Z ekseni için kullanıyoruz.
        throttle_cmd = -self.pitch_pid.update(cy, t)
        throttle_cmd = max(-1.0, min(1.0, throttle_cmd))
        throttle_cmd = self._smooth(self._prev_throttle_cmd, throttle_cmd)
        if abs(throttle_cmd) < 0.01:
            throttle_cmd = 0.0
        self._prev_throttle_cmd = throttle_cmd

        # PITCH (İleri Hız) KONTROLÜ
        # Kamikaze dalışında her zaman hedefe doğru güçlü bir şekilde ileri uçmalıyız.
        pitch_cmd = cfg.pitch_command_max
        pitch_cmd = self._smooth(self._prev_pitch_cmd, pitch_cmd)
        self._prev_pitch_cmd = pitch_cmd

        centered = (abs(cx) < 0.15) and (abs(cy) < 0.15)

        return {
            "throttle": throttle_cmd,
            "pitch":    pitch_cmd,
            "roll":     0.0,
            "yaw":      yaw_cmd,
            "cx": cx, "cy": cy, "centered": centered,
            "bbox_area_ratio": bbox_area_ratio,
            "altitude_error_m": altitude_error_m,
        }


# =============================================================================
# GOREV STATE MACHINE v2
# =============================================================================
class MissionFSM:
    def __init__(self,
                 vision_cfg=None,
                 pid_cfg:       PidConfig        = None,
                 takeoff_cfg:   TakeoffConfig    = None,
                 enable_visual_tracking: bool = False):

        self.gnss_approach    = KamikazeGNSSController()
        self.visual_controller = VisualTrackController(pid_cfg)
        self.takeoff_ctrl     = TakeoffController(takeoff_cfg)
        self.enable_visual_tracking = enable_visual_tracking
        self.vision = self._create_vision_tracker(vision_cfg) if enable_visual_tracking else None

        self.state = MissionState.TAKEOFF
        self._last_log_t = 0.0
        self._visual_entry_altitude_m = None
        self._visual_lost_t = None
        # Kamikaze modunda hedef kaybedilse bile dalisa devam etmesi icin tolerans
        self._visual_lost_grace_sec = 8.0
        self._last_visual_cmd = None
        # v2: Son GNSS komutu kaydı (geçiş seed'leme için)
        self._last_gnss_cmd = None

        # Takeoff hedef irtifasi: SABIT deger.
        self._takeoff_target_alt_m = 45.0   # SABIT kalkis irtifasi (m)

        self._takeoff_target_set   = True   # GNSS'ten guncellenmesini engelle

    @staticmethod
    def _create_vision_tracker(vision_cfg=None):
        # GNSS-only varsayilan modda OpenCV/YOLO bagimliligi yuklenmesin.
        from vision_tracker import VisionTracker, VisionConfig

        return VisionTracker(vision_cfg or VisionConfig())

    def start(self):
        logger.info("drone_sdk baglantisi deneniyor (127.0.0.1:12345)...")
        if not drone.connect():
            raise RuntimeError("Baglanti basarisiz. Unreal sim 'Play' modunda mi?")
        
        if self.enable_visual_tracking and self.vision is not None:
            logger.info("Baglandi. Vision tracker v2 baslatiliyor...")
            self.vision.start()
        else:
            logger.info("Baglandi. GNSS-only kamikaze modu aktif; vision tracker kapali.")
        logger.info(
            "GOREV BASLADI: state=%s | takeoff hedef=%.1fm | visual=%s",
            self.state.value, self._takeoff_target_alt_m, self.enable_visual_tracking,
        )

    def stop(self):
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, arm=False)
        if self.enable_visual_tracking and self.vision is not None:
            self.vision.stop()
        drone.disconnect()
        logger.info("Gorev durduruldu.")

    # ------------------------------------------------------------------
    # STATE GECISLERI
    # ------------------------------------------------------------------
    def _maybe_transition(self, gnss_cmd: dict = None):
        """
        TAKEOFF  -> SEARCH      : takeoff_ctrl.done == True
        SEARCH   -> VISUAL_TRACK: hedef kamerada onaylandi
        VISUAL_TRACK -> SEARCH  : takip kaybedildi
        """
        target_visible = (
            self.enable_visual_tracking
            and self.vision is not None
            and self.vision.is_target_visible()
        )

        if self.state == MissionState.TAKEOFF:
            if self.takeoff_ctrl.done:
                logger.info(
                    ">>> GECIS: TAKEOFF -> SEARCH (irtifa hedefine ulasildi: %.1fm)",
                    drone.get_drone_location()[2] * CM_TO_M,
                )
                self.state = MissionState.SEARCH

        elif self.state == MissionState.SEARCH:
            # Hedef görünüyor mu ve mesafe yeterince yakın mı?
            # Uzaktan çizgi halinde göründüğünde kilitlenmesini engelle
            dist = getattr(self.gnss_approach, 'distance_to_target', float('inf')) / 100.0 # metre
            
            if target_visible and dist < 800.0:
                logger.info(f">>> GECIS: SEARCH -> VISUAL_TRACK (hedef kamerada onaylandi, dist={dist:.1f}m)")
                self.state = MissionState.VISUAL_TRACK

                # v2: PID state'lerini sıfırlamak yerine GNSS komutlarından seed et
                # Böylece throttle/pitch/yaw ani sıçrama yapmaz
                if self._last_gnss_cmd is not None:
                    self.visual_controller.seed_from_gnss(
                        throttle=self._last_gnss_cmd.get("throttle", 0.0),
                        pitch=self._last_gnss_cmd.get("pitch", 0.0),
                        yaw=self._last_gnss_cmd.get("yaw", 0.0),
                    )
                else:
                    self.visual_controller.reset()

                self._visual_entry_altitude_m = drone.get_drone_location()[2] * CM_TO_M
                self._visual_lost_t = None
                self._last_visual_cmd = None

        elif self.state == MissionState.VISUAL_TRACK:
            if target_visible:
                self._visual_lost_t = None
            else:
                now = time.time()
                if self._visual_lost_t is None:
                    self._visual_lost_t = now
                    logger.info(">>> VISUAL_TRACK: kisa takip kaybi, son rota ile tutunuluyor")
                elif now - self._visual_lost_t >= self._visual_lost_grace_sec:
                    logger.info(">>> GECIS: VISUAL_TRACK -> SEARCH (takip %.1fs kayip)", now - self._visual_lost_t)

                    self.state = MissionState.SEARCH
                    self._visual_entry_altitude_m = None
                    self._visual_lost_t = None
                    self._last_visual_cmd = None

    # ------------------------------------------------------------------
    # ANA ADIM
    # ------------------------------------------------------------------
    def step(self):
        self._maybe_transition()

        # ---- TAKEOFF ----
        if self.state == MissionState.TAKEOFF:
            current_alt_m = drone.get_drone_location()[2] * CM_TO_M
            
            if not getattr(self, '_initial_z_set', False):
                self._takeoff_target_alt_m = current_alt_m + 45.0
                self.gnss_approach.target_z = self._takeoff_target_alt_m * 100.0
                self._initial_z_set = True
                logger.info(f"Dinamik kalkis hedefi belirlendi: {self._takeoff_target_alt_m:.1f}m (Zemin: {current_alt_m:.1f}m)")

            cmd = self.takeoff_ctrl.step(current_alt_m, self._takeoff_target_alt_m)

            drone.set_control_surfaces(
                throttle=cmd["throttle"],
                pitch=cmd["pitch"],
                roll=cmd["roll"],
                yaw=cmd["yaw"],
                arm=True,
            )
            self._log_takeoff(cmd)
            return self.state, cmd

        # ---- SEARCH ----
        elif self.state == MissionState.SEARCH:
            tel = drone.get_telemetry()
            cmd = self.gnss_approach.calculate_control_inputs(tel)
            
            drone.set_control_surfaces(
                throttle=cmd["throttle"],
                pitch=cmd["pitch"],
                roll=cmd["roll"],
                yaw=cmd["yaw"],
                arm=True,
            )
            
            self._last_gnss_cmd = dict(cmd)
            
            now = time.time()
            if now - self._last_log_t > 0.5:
                logger.info(f"[GNSS_APPROACH] {cmd.get('status','')} dist={cmd.get('distance', 0.0)/100:.1f}m thr={cmd['throttle']:.2f} pitch={cmd['pitch']:.2f} yaw={cmd['yaw']:.2f}")
                self._last_log_t = now
            return self.state, cmd

        # ---- VISUAL_TRACK ----
        else:
            if self.vision is None:
                logger.warning("VISUAL_TRACK istendi ama vision kapali; SEARCH'e donuluyor.")
                self.state = MissionState.SEARCH
                tel = drone.get_telemetry()
                cmd = self.gnss_approach.calculate_control_inputs(tel)
                self._last_gnss_cmd = dict(cmd)
                return self.state, cmd

            detection = self.vision.get_last_detection()

            if detection.bbox_center_norm is None:
                if self._last_visual_cmd is not None:
                    cmd = dict(self._last_visual_cmd)
                    # Hedef anlik kayboldugunda DALIŞA DEVAM ET! Fren yapma!
                    # Sadece yaw'i cok hafif yavaslat ki savrulmasin
                    cmd["yaw"] = cmd.get("yaw", 0.0) * 0.90
                    # Pitch ve Throttle ayni kalir (son bilinen rotaya dogru ucus)
                    cmd["centered"] = False
                else:
                    cmd = {
                        "throttle": -0.40, "pitch": 0.30, "roll": 0.0, "yaw": 0.0,
                        "cx": 0.0, "cy": 0.0, "centered": False, "altitude_error_m": 0.0,
                    }
            else:
                current_altitude_m = drone.get_drone_location()[2] * CM_TO_M
                if self._visual_entry_altitude_m is None:
                    self._visual_entry_altitude_m = current_altitude_m
                altitude_error_m = self._visual_entry_altitude_m - current_altitude_m
                cmd = self.visual_controller.compute_command(
                    detection.bbox_center_norm,
                    altitude_error_m=altitude_error_m,
                    bbox_area_ratio=detection.bbox_area_ratio,
                )
                cmd["visual_ref_altitude_m"] = self._visual_entry_altitude_m
                cmd["drone_altitude_m"]      = current_altitude_m
                cmd["cx"] = detection.bbox_center_norm[0]
                cmd["cy"] = detection.bbox_center_norm[1]
                cmd["centered"] = (abs(cmd["cx"]) < 0.15 and abs(cmd["cy"]) < 0.15)
                self._last_visual_cmd = dict(cmd)

            # MISSION COMPLETED KONTROLU
            dp = drone.get_drone_location()
            tp = drone.get_target_location()
            dist_3d_m = math.sqrt((tp[0]-dp[0])**2 + (tp[1]-dp[1])**2 + (tp[2]-dp[2])**2) * CM_TO_M
            
            if dist_3d_m < 3.5:
                logger.info("=" * 60)
                logger.info("       MISSION COMPLETED! HEDEF VURULDU!      ")
                logger.info("       (Mesafe: %.1f m)", dist_3d_m)
                logger.info("=" * 60)
                return "MISSION_COMPLETED", cmd

            drone.set_control_surfaces(
                throttle=cmd["throttle"],
                pitch=cmd["pitch"],
                roll=cmd["roll"],
                yaw=cmd["yaw"],
                arm=True,
            )
            self._log_visual(detection, cmd, dist_3d_m)
            return self.state, cmd

    # ------------------------------------------------------------------
    # LOGLAMA
    # ------------------------------------------------------------------
    def _log_takeoff(self, cmd):
        now = time.time()
        if now - self._last_log_t < 0.3:
            return
        self._last_log_t = now
        logger.info(
            "[TAKEOFF] z=%.1fm hedef=%.1fm alt_err=%.1fm | thr=%.2f | done=%s",
            cmd["current_alt_m"], cmd["target_alt_m"], cmd["alt_error_m"],
            cmd["throttle"], cmd["done"],
        )

    def _log_search(self, gnss_result, cmd):
        now = time.time()
        if now - self._last_log_t < 0.3:
            return
        self._last_log_t = now
        logger.info(
            "[SEARCH] GNSS=%s mesafe=%.1fm yaw_hata=%.1f° | thr=%.2f pitch=%.2f yaw=%.2f",
            gnss_result.status.value, cmd["horizontal_dist_m"], cmd["yaw_error_deg"],
            cmd["throttle"], cmd["pitch"], cmd["yaw"],
        )

    def _log_visual(self, detection, cmd, dist_3d_m):
        now = time.time()
        if now - self._last_log_t < 0.3:
            return
        self._last_log_t = now
        logger.info(
            "[KAMİKAZE_TRACK] dist=%.1fm conf=%.2f cx=%.2f cy=%.2f merkezde=%s | "
            "z=%.1fm alt_err=%.1fm | thr=%.2f pitch=%.2f roll=%.2f yaw=%.2f",
            dist_3d_m,
            detection.confidence, cmd.get("cx", 0.0), cmd.get("cy", 0.0), cmd.get("centered", False),
            cmd.get("drone_altitude_m", 0.0), cmd.get("altitude_error_m", 0.0),
            cmd["throttle"], cmd["pitch"], cmd["roll"], cmd["yaw"],
        )


# =============================================================================
# ANA DONGU
# =============================================================================
def build_vision_config(weights_path: str = "talon_v7.pt",
                        camera_index: int = 1,
                        show_window: bool = False,
                        inference_size: int = 640,
                        confidence_threshold: float = 0.45,
                        device: str = None,
                        half: bool = False,
                        enable_csrt_tracking: bool = True,
                        detector_interval_frames: int = 8,
                        tracker_max_age_frames: int = 45):
    from vision_tracker import VisionConfig

    return VisionConfig(
        weights_path=weights_path,
        camera_index=camera_index,
        inference_size=inference_size,
        confidence_threshold=confidence_threshold,
        device=device,
        half=half,
        show_window=show_window,
        enable_csrt_tracking=enable_csrt_tracking,
        detector_interval_frames=detector_interval_frames,
        tracker_max_age_frames=tracker_max_age_frames,
    )


def run_mission(loop_hz: float = 50.0, show_window: bool = False,
                weights_path: str = "talon_v7.pt", camera_index: int = 1,
                enable_visual_tracking: bool = False,
                inference_size: int = 640,
                confidence_threshold: float = 0.45,
                device: str = None,
                half: bool = False,
                enable_csrt_tracking: bool = True,
                detector_interval_frames: int = 8,
                tracker_max_age_frames: int = 45):

    vision_cfg = None
    if enable_visual_tracking:
        vision_cfg = build_vision_config(
            weights_path=weights_path,
            camera_index=camera_index,
            inference_size=inference_size,
            confidence_threshold=confidence_threshold,
            device=device,
            half=half,
            show_window=show_window,
            enable_csrt_tracking=enable_csrt_tracking,
            detector_interval_frames=detector_interval_frames,
            tracker_max_age_frames=tracker_max_age_frames,
        )
    fsm = MissionFSM(
        vision_cfg=vision_cfg,
        enable_visual_tracking=enable_visual_tracking,
    )
    fsm.start()
    dt_target = 1.0 / loop_hz

    try:
        while True:
            loop_start = time.time()
            state, cmd = fsm.step()
            
            if state == "MISSION_COMPLETED":
                break
            
            if enable_visual_tracking and fsm.vision is not None:
                if not fsm.vision.update_window():
                    break
                    
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt_target - elapsed))
    except KeyboardInterrupt:
        logger.info("Durduruldu (Ctrl+C).")
    finally:
        fsm.stop()
        if show_window:
            import cv2
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Avci Drone - GNSS + Vision Mix Conf v2")
    parser.add_argument("--weights",      default="talon_v7.pt")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--imgsz",        type=int, default=640,
                        help="YOLO inference boyutu. FPS dusukse 320 deneyin.")
    parser.add_argument("--conf",         type=float, default=0.55,
                        help="Tespit guven esigi.")
    parser.add_argument("--device",       default=None,
                        help='Ultralytics device: "cpu" veya CUDA icin "0".')
    parser.add_argument("--half",         action="store_true",
                        help="CUDA'da FP16 kullan. CPU'da acmayin.")
    parser.add_argument("--show",         action="store_true", help="Tespit penceresini ac")
    parser.add_argument("--no-visual-track", action="store_true", help="Kamera takibini kapat")
    parser.add_argument("--no-csrt", action="store_true", help="YOLO sonrasi SortCSRT takibini kapat")
    parser.add_argument("--detector-interval", type=int, default=8, help="SortCSRT kilitliyken YOLO tazeleme araligi")
    parser.add_argument("--tracker-max-age", type=int, default=150, help="YOLO yokken CSRT kutusunu yasatma suresi")
    args = parser.parse_args()

    run_mission(
        show_window=args.show,
        weights_path=args.weights,
        camera_index=args.camera_index,
        inference_size=args.imgsz,
        confidence_threshold=args.conf,
        device=args.device,
        half=args.half,
        enable_visual_tracking=not args.no_visual_track,
        enable_csrt_tracking=not args.no_csrt,
        detector_interval_frames=args.detector_interval,
        tracker_max_age_frames=args.tracker_max_age,
    )
