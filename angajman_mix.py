"""
================================================================================
AVCI DRONE - ANGAJMAN MIX v2 (VISION KAMIKAZE)
================================================================================
GNSS yaklaşma + Kamera Takip + Dalış/Vurma sistemi.

STATE GECISLERI:
  TAKEOFF        -> Kalkış (v1 mantığı)
  GNSS_APPROACH  -> Jamming korumalı yaklaşma (v1 GuidanceController)
  VISUAL_TRACK   -> 5 saniye kadrajdan çıkarmadan görsel takip (Kalman + CSRT)
  DIVE_ATTACK    -> 5 saniye tam gaz dalış, hedefi vur
  MISSION_COMPLETE -> Hedef düşürüldü (mesafe < 3.5m)

Kalman Filter: Hungarian Algorithm (SortCSRT) + BBoxKalman ile pürüzsüz takip.
Grayscale CSRT: Kontrast artışı ile daha stabil takip (--gray-tracking).
Smorodov/Multitarget-tracker ilhamı: IoU eşleştirme + Kalman tahmin entegre.
================================================================================
"""

import sys
import os
import time
import math
import logging
from enum import Enum
import argparse

# --- PATH AYARLARI ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'gnss_vision_mix_conf_v1'))

import drone_sdk as drone

# --- V1 GNSS (ANTI-JAMMING) MODULLERI ---
from guidance_controller import GuidanceController, GuidanceConfig
from angajman import TakeoffController, TakeoffConfig

# --- V3 VISION MODULLERI ---
from kamikaze_vision_v3 import VisionThread, VisionResult

# ===========================================================================
# MIX VİZYON KONTROLCÜSÜ (QUADCOPTER DİNAMİKLERİ)
# Hız = Pitch, İrtifa = Throttle
# ===========================================================================
class MixVisionDiveController:
    KP_YAW = 0.35; KD_YAW = 0.10; KFF_YAW = 0.05
    KP_THR = 0.35; KD_THR = 0.10; KFF_THR = 0.05

    CENTER_TOL = 0.15   
    DIVE_LOCK_COUNT = 4      
    BASE_PITCH = 0.85   # Hızlanmak için yüksek ileri pitch (120+ speed)
    DIVE_PITCH = 1.00   # Son dalışta tam sürat füze
    TRACK_PITCH = 0.30  # Takip modunda yavaş ilerleme (kadrajdan kaçırmamak için)

    def __init__(self):
        self._yaw_int = 0.0; self._yaw_prev = 0.0; self._yaw_t = None
        self._thr_int = 0.0; self._thr_prev = 0.0; self._thr_t = None
        self._center_count = 0
        self._dive_locked = False
        self._current_pitch = 0.6  # GNSS fazından gelen tahmini başlangıç hızı

    def reset(self):
        self.__init__()

    def _pid(self, err, integral, prev, kp, kd, ff_val, kff, t, max_i=0.5):
        now = time.time()
        dt = max((now - t) if t else 0.02, 0.005)
        integral = max(-max_i, min(max_i, integral + err * dt))
        deriv = (err - prev) / dt
        out = kp * err + kd * deriv + kff * ff_val
        return out, integral, err, now

    def compute(self, r: VisionResult, mode: str = "track") -> dict:
        if not r.kf_valid:
            return {"thr": 0.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0, 
                    "dive": False, "center_count": 0}

        cx = r.kf_cx + r.kf_vcx * 0.15
        cy = r.kf_cy + r.kf_vcy * 0.15
        vcx, vcy = r.kf_vcx, r.kf_vcy

        # YAW (Sağ-Sol)
        yaw_out, self._yaw_int, self._yaw_prev, self._yaw_t = self._pid(
            cx, self._yaw_int, self._yaw_prev, self.KP_YAW, self.KD_YAW, vcx, self.KFF_YAW, self._yaw_t)
        yaw_cmd = max(-0.8, min(0.8, yaw_out))

        # THROTTLE (İrtifa: Quadcopter'de Throttle Dikey Komuttur)
        thr_out, self._thr_int, self._thr_prev, self._thr_t = self._pid(
            -cy, self._thr_int, self._thr_prev, self.KP_THR, self.KD_THR, -vcy, self.KFF_THR, self._thr_t)
        thr_cmd = max(-0.6, min(0.6, thr_out))

        # PITCH (Hız)
        if mode == "track":
            self._current_pitch = self._current_pitch * 0.90 + self.TRACK_PITCH * 0.10
            pitch_cmd = self._current_pitch
        elif mode == "dive":
            self._current_pitch = self.DIVE_PITCH
            pitch_cmd = self.DIVE_PITCH
        else:
            pitch_cmd = self._current_pitch

        centered = (abs(cx) < self.CENTER_TOL) and (abs(cy) < self.CENTER_TOL)
        if centered:
            self._center_count += 1
        else:
            self._center_count = max(0, self._center_count - 1)

        if self._center_count >= self.DIVE_LOCK_COUNT:
            self._dive_locked = True

        is_dive = (mode == "dive")

        return {"thr": thr_cmd, "pitch": pitch_cmd, "roll": 0.0, "yaw": yaw_cmd, 
                "dive": is_dive, "center_count": self._center_count}


logger = logging.getLogger("angajman_mix")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_h)

CM_TO_M = 0.01

# ===========================================================================
# KADRAJ KORUMA İZLEYİCİSİ
# ===========================================================================
class FrameKeeper:
    def __init__(self, frame_threshold: float = 0.85, grace_sec: float = 1.5):
        self.frame_threshold = frame_threshold
        self.grace_sec = grace_sec
        self._track_start = None
        self._last_in_frame = None
        self._lost_since = None
    
    def reset(self):
        self._track_start = None
        self._last_in_frame = None
        self._lost_since = None
    
    def update(self, kf_cx: float, kf_cy: float, kf_valid: bool, now: float) -> float:
        in_frame = (kf_valid and 
                    abs(kf_cx) < self.frame_threshold and 
                    abs(kf_cy) < self.frame_threshold)
        
        if in_frame:
            self._lost_since = None
            if self._track_start is None:
                self._track_start = now
            self._last_in_frame = now
            return now - self._track_start
        
        if self._track_start is None:
            return 0.0
        
        if self._lost_since is None:
            self._lost_since = now
        
        if now - self._lost_since < self.grace_sec:
            return now - self._track_start
        
        self._track_start = None
        self._last_in_frame = None
        self._lost_since = None
        return -1.0


# ===========================================================================
# ATTACK PHASE ENUM
# ===========================================================================
class AttackPhase(Enum):
    TAKEOFF          = "TAKEOFF"
    GNSS_APPROACH    = "GNSS_APPROACH"
    VISUAL_TRACK     = "VISUAL_TRACK"       # 5 saniye kadrajda tutma
    DIVE_ATTACK      = "DIVE_ATTACK"        # 5 saniye vurma dalışı
    MISSION_COMPLETE = "MISSION_COMPLETE"


# ===========================================================================
# ANA MİSYON SINIFI
# ===========================================================================
class KamikazeMixMission:
    def __init__(self, args):
        self.args = args
        self.phase = AttackPhase.TAKEOFF
        
        self.takeoff_ctrl = TakeoffController()
        self.gnss_ctrl = GuidanceController()
        
        self.vision_ctrl = MixVisionDiveController()
        
        self.frame_keeper = FrameKeeper(
            frame_threshold=args.frame_threshold,
            grace_sec=args.frame_grace_sec
        )
        
        self.vision = VisionThread(
            weights=args.weights,
            window_name=args.window_name,
            conf=args.conf,
            lost_streak=args.lost_streak,
            use_obs=args.use_obs,
            camera_index=args.camera_index,
            show_window=args.show,
            confirm_window=args.confirm_window,
            confirm_ratio=args.confirm_ratio
        )
        self.vision_started = False
        
        self._takeoff_target_alt_m = args.cruise_altitude
        self._last_log = 0.0
        self._last_locked_yaw = 0.0
        self._last_locked_pitch = 0.0
        self._last_locked_thr = 0.0
        self.visual_lock_start_time = None
        self.dive_start_time = None
        self._last_alt_hold_log = 0.0
        self._mission_completed = False

    def _read_altitude_m(self, drone_pos=None) -> float:
        alt_candidates = []
        if drone_pos:
            alt_candidates.append(drone_pos[2] * CM_TO_M)
        try:
            sdk_alt = float(drone.get_drone_altitude())
            alt_candidates.append(sdk_alt * CM_TO_M if abs(sdk_alt) > 250.0 else sdk_alt)
        except Exception:
            pass
        for alt in reversed(alt_candidates):
            if math.isfinite(alt) and -5.0 <= alt <= 1000.0:
                return alt
        return float("inf")

    def _altitude_hold_throttle(self, current_alt_m: float) -> float:
        if not math.isfinite(current_alt_m):
            return 0.0
        err = self.args.cruise_altitude - current_alt_m
        if abs(err) <= self.args.cruise_alt_deadband:
            return 0.0
        climb = max(0.0, min(self.args.max_climb_throttle, err * self.args.alt_hold_kp))
        descent = min(0.0, max(-self.args.max_descent_throttle, err * self.args.alt_hold_kp))
        return climb if err > 0.0 else descent

    def _apply_cruise_altitude_hold(self, cmd: dict, current_alt_m: float,
                                    phase_name: str, visual_priority: bool = False) -> dict:
        guarded = dict(cmd)
        hold_thr = self._altitude_hold_throttle(current_alt_m)
        old_thr = guarded.get("throttle", guarded.get("thr", 0.0))

        if visual_priority:
            if current_alt_m < self.args.cruise_altitude - self.args.visual_alt_floor_margin:
                new_thr = max(old_thr, hold_thr)
            elif current_alt_m > self.args.cruise_altitude + self.args.visual_alt_ceiling_margin:
                new_thr = min(old_thr, hold_thr)
            else:
                new_thr = old_thr
        else:
            new_thr = hold_thr

        if "thr" in guarded:
            guarded["thr"] = new_thr
        else:
            guarded["throttle"] = new_thr

        now = time.time()
        if abs(new_thr - old_thr) > 0.03 and now - self._last_alt_hold_log > 0.5:
            self._last_alt_hold_log = now
            logger.info("[ALT_HOLD:%s] alt=%.1fm target=%.1fm thr %.2f -> %.2f",
                        phase_name, current_alt_m, self.args.cruise_altitude, old_thr, new_thr)
        return guarded

    def _safe_set_control(self, throttle: float, pitch: float, roll: float,
                          yaw: float, arm: bool, current_alt_m: float,
                          phase_name: str):
        if math.isfinite(current_alt_m) and current_alt_m <= self.args.altitude_low_recovery_floor:
            old = (throttle, pitch, roll, yaw)
            throttle = self.args.altitude_low_recovery_throttle
            pitch = self.args.altitude_low_recovery_pitch
            roll = 0.0
            yaw = 0.0
            now = time.time()
            if now - self._last_alt_hold_log > 0.5:
                self._last_alt_hold_log = now
                logger.info("[ALT_LOW_RECOVERY:%s] alt=%.1fm cmd=(%.2f,%.2f,%.2f,%.2f)->(%.2f,%.2f,%.2f,%.2f)",
                            phase_name, current_alt_m,
                            old[0], old[1], old[2], old[3],
                            throttle, pitch, roll, yaw)

        elif math.isfinite(current_alt_m) and current_alt_m >= self.args.altitude_emergency_ceiling:
            old = (throttle, pitch, roll, yaw)
            throttle = self.args.altitude_emergency_throttle
            pitch = self.args.altitude_emergency_pitch
            roll = 0.0
            yaw = 0.0
            now = time.time()
            if now - self._last_alt_hold_log > 0.5:
                self._last_alt_hold_log = now
                logger.info("[ALT_EMERGENCY:%s] alt=%.1fm cmd=(%.2f,%.2f,%.2f,%.2f)->(%.2f,%.2f,%.2f,%.2f)",
                            phase_name, current_alt_m,
                            old[0], old[1], old[2], old[3],
                            throttle, pitch, roll, yaw)

        drone.set_control_surfaces(throttle, pitch, roll, yaw, arm=arm)

    def _check_impact(self, dist_m: float) -> bool:
        return math.isfinite(dist_m) and dist_m < self.args.impact_distance

    def _finish_mission(self, dist_m: float):
        self.phase = AttackPhase.MISSION_COMPLETE
        self._mission_completed = True
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, arm=False)
        logger.info("=" * 60)
        logger.info("     ██╗  ██╗███████╗██████╗ ███████╗███████╗    ")
        logger.info("     ██║  ██║██╔════╝██╔══██╗██╔════╝██╔════╝    ")
        logger.info("     ███████║█████╗  ██║  ██║█████╗  █████╗      ")
        logger.info("     ██╔══██║██╔══╝  ██║  ██║██╔══╝  ██╔══╝      ")
        logger.info("     ██║  ██║███████╗██████╔╝███████╗██║          ")
        logger.info("     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚══════╝╚═╝          ")
        logger.info("")
        logger.info("       >>> MISSION COMPLETED <<<")
        logger.info("       HEDEF İHA DÜŞÜRÜLDÜ!")
        logger.info("       Vuruş mesafesi: %.1f m", dist_m)
        logger.info("=" * 60)

    def start(self):
        logger.info("drone_sdk baglantisi deneniyor (127.0.0.1:12345)...")
        if not drone.connect():
            raise RuntimeError("Baglanti basarisiz. Unreal sim 'Play' modunda mi?")
        logger.info("Baglandi. Angajman Mix v2 (Vision Kamikaze) devrede!")

    def stop(self):
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, arm=False)
        if self.vision_started:
            self.vision.stop()
        drone.disconnect()
        logger.info("Gorev durduruldu.")

    def step(self):
        now = time.time()
        
        drone_pos = drone.get_drone_location()
        target_pos = drone.get_target_location()
        current_alt_m = self._read_altitude_m(drone_pos)
        
        if drone_pos and target_pos:
            dx = target_pos[0] - drone_pos[0]
            dy = target_pos[1] - drone_pos[1]
            dz = target_pos[2] - drone_pos[2]
            dist_m = math.sqrt(dx*dx + dy*dy + dz*dz) * CM_TO_M
        else:
            dist_m = float('inf')

        if not self.vision_started and dist_m < self.args.vision_arm_distance and self.phase == AttackPhase.GNSS_APPROACH:
            logger.info(">>> %.1fm'ye girildi -- V3 YOLO/CSRT/BYTETRACK BAŞLATILIYOR!", dist_m)
            self.vision.start()
            time.sleep(0.5)
            self.vision_started = True
            gray_mode = "AKTIF" if getattr(self.args, 'gray_tracking', False) else "KAPALI"
            logger.info(">>> Vision ayarlari: conf=%.2f confirm=%d/%.0f%% use_obs=%s camera=%d gray=%s",
                        self.args.conf, self.args.confirm_window,
                        self.args.confirm_ratio * 100.0, self.args.use_obs,
                        self.args.camera_index, gray_mode)

        vis = self.vision.get() if self.vision_started else VisionResult()

        if self.phase == AttackPhase.TAKEOFF:
            cmd = self.takeoff_ctrl.step(current_alt_m, self._takeoff_target_alt_m)
            takeoff_err = self._takeoff_target_alt_m - current_alt_m
            if takeoff_err > 5.0:
                cmd["throttle"] = max(cmd["throttle"], self.args.takeoff_min_climb_throttle)
            elif takeoff_err > self.args.takeoff_alt_tolerance:
                cmd["throttle"] = max(cmd["throttle"], self.args.takeoff_final_climb_throttle)
            
            self._safe_set_control(
                cmd["throttle"], cmd["pitch"], cmd["roll"], cmd["yaw"], True,
                current_alt_m, "TAKEOFF"
            )
            if current_alt_m >= (self._takeoff_target_alt_m - self.args.takeoff_alt_tolerance):
                logger.info(">>> TAKEOFF TAMAMLANDI. (%.1fm) GNSS_APPROACH BAŞLIYOR.", current_alt_m)
                self.gnss_ctrl.seed_throttle_state(self.takeoff_ctrl._current_throttle)
                self.phase = AttackPhase.GNSS_APPROACH

        elif self.phase == AttackPhase.GNSS_APPROACH:
            gnss_result, cmd = self.gnss_ctrl.step()
            cmd = self._apply_cruise_altitude_hold(cmd, current_alt_m, "GNSS")
            cmd["pitch"] = max(-self.args.gnss_pitch_cap, min(self.args.gnss_pitch_cap, cmd["pitch"]))
            self._safe_set_control(
                cmd["throttle"], cmd["pitch"], cmd["roll"], cmd["yaw"], True,
                current_alt_m, "GNSS"
            )
            
            if vis.confirmed:
                logger.info(">>> HEDEF KAMERADA ONAYLANDI! found=%s conf=%.2f kf=%s",
                            vis.found, vis.confidence, vis.kf_valid)
                logger.info(">>> 5 SANİYELİK KADRAJ TAKİBİ (VISUAL_TRACK) BAŞLIYOR!")
                self.phase = AttackPhase.VISUAL_TRACK
                self.vision_ctrl.reset()
                self.frame_keeper.reset()
                self.visual_lock_start_time = now
            else:
                if now - self._last_log > 0.3:
                    self._last_log = now
                    if self.vision_started:
                        hist = getattr(self.vision, "_found_history", [])
                        hit_ratio = (sum(hist) / len(hist)) if hist else 0.0
                        miss_conf = getattr(self.vision, "_miss_conf", 0)
                        logger.info("[GNSS_APPROACH (Anti-Jam: %s)] dist=%.1fm | thr=%.2f pitch=%.2f yaw=%.2f | vis=%s conf=%.2f ratio=%.0f%%(%d/%d) miss=%s kf=%s",
                                    gnss_result.status.name, dist_m, cmd["throttle"], cmd["pitch"], cmd["yaw"],
                                    vis.found, vis.confidence, hit_ratio * 100.0,
                                    sum(hist), len(hist), miss_conf, vis.kf_valid)
                    else:
                        logger.info("[GNSS_APPROACH (Anti-Jam: %s)] dist=%.1fm | thr=%.2f pitch=%.2f yaw=%.2f",
                                    gnss_result.status.name, dist_m, cmd["throttle"], cmd["pitch"], cmd["yaw"])

        elif self.phase == AttackPhase.VISUAL_TRACK:
            if self._check_impact(dist_m):
                self._finish_mission(dist_m)
                return True
            
            frame_time = self.frame_keeper.update(vis.kf_cx, vis.kf_cy, vis.kf_valid, now)
            
            if frame_time < 0:
                logger.info(">>> HEDEF KADRAJDAN ÇIKTI! (grace=%.1fs) GNSS_APPROACH'A DÖNÜLÜYOR.",
                            self.args.frame_grace_sec)
                self.phase = AttackPhase.GNSS_APPROACH
                self.visual_lock_start_time = None
                self.frame_keeper.reset()
                self.vision_ctrl.reset()
                return False
            
            if frame_time >= self.args.track_duration:
                logger.info(">>> %.1f SANİYE KADRAJ TAKİBİ TAMAMLANDI! DIVE_ATTACK BAŞLIYOR!", frame_time)
                self.phase = AttackPhase.DIVE_ATTACK
                self.dive_start_time = now
                self.vision_ctrl._dive_locked = True
                return False
            
            vcmd = self.vision_ctrl.compute(vis, mode="track")
            vcmd = self._apply_cruise_altitude_hold(vcmd, current_alt_m, "VISUAL", visual_priority=True)
            self._safe_set_control(
                vcmd["thr"], vcmd["pitch"], vcmd["roll"], vcmd["yaw"], True,
                current_alt_m, "VISUAL_TRACK"
            )

            if now - self._last_log > 0.2:
                self._last_log = now
                logger.info("[VISUAL_TRACK] kadraj=%.1f/%.1fs dist=%.1fm alt=%.1fm yaw=%.2f thr=%.2f pitch=%.2f conf=%.2f kf=%s",
                            frame_time, self.args.track_duration, dist_m,
                            current_alt_m, vcmd["yaw"], vcmd["thr"], vcmd["pitch"],
                            vis.confidence, vis.kf_valid)

        elif self.phase == AttackPhase.DIVE_ATTACK:
            if self._check_impact(dist_m):
                self._finish_mission(dist_m)
                return True
            
            dive_elapsed = now - self.dive_start_time
            
            if dive_elapsed >= self.args.dive_duration:
                if vis.confirmed and vis.kf_valid:
                    logger.info(">>> DALIŞ SÜRESİ DOLDU ama hedef kamerada! Dalışa devam...")
                    self.dive_start_time = now
                else:
                    logger.info(">>> DALIŞ SÜRESİ DOLDU, hedef kayıp. GNSS_APPROACH'A DÖNÜLÜYOR.")
                    self.phase = AttackPhase.GNSS_APPROACH
                    self.dive_start_time = None
                    self.visual_lock_start_time = None
                    self.frame_keeper.reset()
                    self.vision_ctrl.reset()
                    return False
            
            vcmd = self.vision_ctrl.compute(vis, mode="dive")
            self._safe_set_control(
                vcmd["thr"], vcmd["pitch"], vcmd["roll"], vcmd["yaw"], True,
                current_alt_m, "DIVE_ATTACK"
            )
            
            if vis.kf_valid:
                self._last_locked_yaw = vcmd["yaw"]
                self._last_locked_thr = vcmd["thr"]
                self._last_locked_pitch = vcmd["pitch"]

            if now - self._last_log > 0.2:
                self._last_log = now
                logger.info("[DIVE_ATTACK] dalış=%.1f/%.1fs dist=%.1fm alt=%.1fm yaw=%.2f thr=%.2f pitch=%.2f conf=%.2f",
                            dive_elapsed, self.args.dive_duration, dist_m,
                            current_alt_m, vcmd["yaw"], vcmd["thr"], vcmd["pitch"],
                            vis.confidence)

        elif self.phase == AttackPhase.MISSION_COMPLETE:
            return True

        return False

    def run(self, hz: float = 50.0):
        self.start()
        dt = 1.0 / hz
        try:
            while True:
                t0 = time.time()
                if self.step():
                    break
                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            logger.info("Ctrl+C ile durduruldu.")
        finally:
            self.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avci Drone - Angajman Mix v2 (Vision Kamikaze)")
    
    parser.add_argument("--weights",      default="talon_v7.pt", help="YOLO model dosyası")
    parser.add_argument("--window-name",  default="", help="Ekran yakalama için pencere ismi (mss modu)")
    parser.add_argument("--show",         action="store_true", help="Kamera tespit penceresini göster")
    parser.add_argument("--conf",         type=float, default=0.45, help="YOLO confidence eşiği")
    parser.add_argument("--confirm-window", type=int, default=10, help="Sliding-window frame sayısı")
    parser.add_argument("--confirm-ratio", type=float, default=0.4, help="Hedef onayı için doğrulama oranı")
    parser.add_argument("--lost-streak", type=int, default=8, help="Kaç frame kaybolursa takip bırakılır")
    parser.add_argument("--hz",           type=float, default=50.0, help="Döngü frekansı")
    parser.add_argument("--use-obs",      action="store_true", help="OBS Virtual Camera kullan (Önerilir)")
    parser.add_argument("--camera-index", type=int, default=1, help="OBS modu kamera index")
    parser.add_argument("--vision-arm-distance", type=float, default=400.0, help="Kaç metrede YOLO/vision başlatılsın")
    parser.add_argument("--manual-roi", action="store_true", help="Pencerede hedefe tıklayarak CSRT tracker başlat")
    parser.add_argument("--manual-roi-size", type=int, default=86, help="Mouse click etrafındaki ROI kutu boyutu")
    
    parser.add_argument("--gray-tracking", action="store_true", 
                        help="Kamera görüntüsünü grayscale modda işle (daha stabil CSRT takibi)")
    
    parser.add_argument("--track-duration", type=float, default=5.0, 
                        help="VISUAL_TRACK süresi: kaç saniye kadrajda tutulsun (varsayılan: 5)")
    parser.add_argument("--dive-duration", type=float, default=5.0, 
                        help="DIVE_ATTACK süresi: kaç saniye dalış yapılsın (varsayılan: 5)")
    parser.add_argument("--impact-distance", type=float, default=3.5, 
                        help="MISSION COMPLETED için hedef mesafesi (metre, varsayılan: 3.5)")
    
    parser.add_argument("--frame-threshold", type=float, default=0.85,
                        help="Kadraj sınırı: |cx|,|cy| bu değerin altındaysa kadrajda sayılır")
    parser.add_argument("--frame-grace-sec", type=float, default=1.5,
                        help="Kadrajdan çıkınca kaç saniye beklensin (grace period)")
    
    parser.add_argument("--cruise-altitude", type=float, default=45.0, help="Kruaz irtifa (m)")
    parser.add_argument("--takeoff-alt-tolerance", type=float, default=1.0, help="Takeoff toleransı (m)")
    parser.add_argument("--takeoff-max-throttle", type=float, default=0.85, help="Takeoff max throttle")
    parser.add_argument("--takeoff-ramp-rate", type=float, default=0.02, help="Takeoff throttle rampa hızı")
    parser.add_argument("--takeoff-min-climb-throttle", type=float, default=0.30, help="Uzakken minimum tırmanma throttle")
    parser.add_argument("--takeoff-final-climb-throttle", type=float, default=0.12, help="Yakınken minimum tırmanma throttle")
    
    parser.add_argument("--alt-hold-kp", type=float, default=0.035, help="İrtifa tutucu P kazancı")
    parser.add_argument("--cruise-alt-deadband", type=float, default=1.5, help="İrtifa ölü bölgesi (m)")
    parser.add_argument("--max-climb-throttle", type=float, default=0.35, help="Max tırmandırma throttle")
    parser.add_argument("--max-descent-throttle", type=float, default=0.16, help="Max alçaltma throttle")
    parser.add_argument("--gnss-pitch-cap", type=float, default=0.18, help="GNSS pitch sınırı")
    parser.add_argument("--altitude-low-recovery-floor", type=float, default=38.0, help="Düşük irtifa kurtarma eşiği")
    parser.add_argument("--altitude-low-recovery-throttle", type=float, default=0.65, help="Düşük irtifa kurtarma throttle")
    parser.add_argument("--altitude-low-recovery-pitch", type=float, default=0.0, help="Düşük irtifa kurtarma pitch")
    parser.add_argument("--altitude-emergency-ceiling", type=float, default=60.0, help="Acil irtifa tavanı")
    parser.add_argument("--altitude-emergency-throttle", type=float, default=-0.65, help="Acil irtifa throttle")
    parser.add_argument("--altitude-emergency-pitch", type=float, default=0.0, help="Acil irtifa pitch")
    parser.add_argument("--visual-alt-floor-margin", type=float, default=3.0, help="Görsel takip alt marjin")
    parser.add_argument("--visual-alt-ceiling-margin", type=float, default=6.0, help="Görsel takip üst marjin")
    
    args = parser.parse_args()

    args.show = True

    if args.conf > 0.55:
        logger.warning("--conf %.2f çok yüksek. İlk deneme için 0.45 önerilir.", args.conf)

    print("=" * 65)
    print("  AVCI DRONE -- ANGAJMAN MIX v2 (VISION KAMIKAZE)")
    print("  - VISUAL_TRACK : %.1f saniye kadrajda tut" % args.track_duration)
    print("  - DIVE_ATTACK  : %.1f saniye tam gaz dalış" % args.dive_duration)
    print("  - IMPACT       : < %.1f metre = MISSION COMPLETED" % args.impact_distance)
    print("  - GRAY CSRT    : %s" % ("AKTİF" if args.gray_tracking else "KAPALI"))
    print("  - Kalman+SortCSRT: Hungarian Algorithm Takip")
    print("=" * 65)
    
    mission = KamikazeMixMission(args)
    mission.run(hz=args.hz)