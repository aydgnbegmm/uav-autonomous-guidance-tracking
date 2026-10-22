"""
================================================================================
AVCI DRONE - YONELME / ARAMA FAZI KONTROLCUSU (Eszamanli P Kontrol)
================================================================================
DUZELTMELER (bu versiyon):
  1. _log() - TypeError duzeltildi: fazla horizontal_dist_m argumani kaldirildi,
     fast_gain argumani dogru siraya eklendi.
  2. compute_command() - altitude_error < 0 oldugunda (drone hedeften yuksekte)
     throttle pozitif kalamaz: hard clamp eklendi.
  3. throttle_sign parametresi KALDIRILDI - gereksizdi, sorun 2. maddeydi.
================================================================================
"""
import logging
import math
import time
import csv
import drone_sdk as drone
from gnss_reliability import GnssReliabilityAnalyzer, GnssStatus

logger = logging.getLogger("guidance")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(_h)

CM_TO_M = 0.01
M_TO_CM = 100.0
MISSION_NAV_CSV_PATH = "mission_nav_log.csv"
MISSION_NAV_CSV_FIELDS = [
    "t", "mode", "gnss_status", "reason", "measurement_age_s",
    "drone_x", "drone_y", "drone_z", "drone_yaw", "drone_speed_mps",
    "target_x", "target_y", "target_z", "target_vx", "target_vy", "target_vz",
    "aim_x", "aim_y", "aim_z", "horizontal_dist_m", "raw_horizontal_dist_m",
    "dist_3d_m", "altitude_error_m", "target_altitude_gap_m", "yaw_error_deg", "follow_along_m",
    "follow_lateral_m", "search_radius", "nis", "uncertainty_scale",
    "approach_scale", "fast_gain", "cmd_throttle", "cmd_pitch",
    "cmd_roll", "cmd_yaw", "armed",
]
_mission_nav_csv_initialized = False


class GuidanceConfig:
    # --- YAW P kontrol (gain-scheduled + deadband) ---
    # Kamera SABIT (gimbal yok, govdeye bagli) -- bu yuzden donus agirligi
    # ESAS OLARAK yaw'da olmali, roll/banking sadece hafif bir destek.
    # kp_yaw bu yuzden biraz geri yukari cekildi (yaw artik donusun asil
    # yuku oldugu icin tek basina yeterli donus hizini saglamali).
    kp_yaw: float = 2.6                 # ESKI: 2.2 -> yaw artik donusun esas
                                         # tasiyicisi, roll cok az destek veriyor
    kp_yaw_near: float = 0.9            # yakin terminal bolgede spin'i onlemek icin daha yumusak
    yaw_command_max: float = 1.0
    yaw_deadband_deg: float = 2.0
    terminal_yaw_radius_m: float = 12.0
    terminal_yaw_command_max: float = 0.22
    terminal_roll_command_max: float = 0.06
    close_camera_yaw_radius_m: float = 24.0
    close_camera_yaw_command_max: float = 0.28
    terminal_fast_yaw_command_max: float = 0.14
    terminal_fast_roll_command_max: float = 0.025
    below_target_slowdown_band_m: float = 4.0
    below_target_slowdown_pitch_cap: float = 0.18
    hold_slowdown_radius_m: float = 6.0
    hold_slowdown_pitch_cap: float = 0.18
    hold_radius_m: float = 4.0
    hold_inner_radius_m: float = 2.0
    hold_pitch_cap: float = 0.12
    hold_inner_pitch_cap: float = 0.08
    follow_behind_radius_m: float = 70.0
    follow_behind_distance_m: float = 8.0
    follow_hold_pitch_max: float = 0.42
    follow_catchup_pitch_max: float = 0.85
    follow_ahead_pitch_max: float = 0.22
    follow_catchup_gap_m: float = 12.0
    follow_slow_gap_m: float = 4.0
    follow_speed_match_radius_m: float = 18.0
    follow_speed_match_speed_margin_mps: float = 3.0
    follow_speed_match_pitch_max: float = 0.08
    follow_velocity_yaw_radius_m: float = 12.0
    follow_velocity_yaw_ahead_margin_m: float = 4.0
    close_mode_hysteresis_radius_m: float = 20.0
    close_mode_bad_count_required: int = 3
    close_yaw_reference_hold_radius_m: float = 20.0
    close_yaw_reference_max_step_deg: float = 25.0
    # --- PITCH (ileri ucus) ---
    pitch_command_base: float = 0.98    # carpma pozisyonuna daha kararli kapanma
                                         # mesafeyi cok acmasin diye biraz atik
                                         # icin ileri itme genel olarak dusuruldu
    yaw_error_for_zero_pitch_deg: float = 130.0  # ESKI: 90.0 -> donus sirasinda
                                                   # pitch (ileri itme) cok daha
                                                   # gec sifirlaniyor, banking ile
                                                   # donerken hiz/ivme kaybetmiyor
    yaw_error_for_min_pitch_factor: float = 0.45  # ESKI: 0.55 -> donus icinde
                                                   # pitch biraz daha kisilsin,
                                                   # donus + ileri hiz ayni anda
                                                   # asiri olmasin (irtifa/FOV icin)
    # ESKI: slowdown_start_radius_m=35, slowdown_min_pitch=0.30 -- GNSS modunda
    # hedefe yaklasinca pitch'i guclu kisip yavaslatiyordu, bu da "yakina gelince
    # pesini birakiyor/yavasliyor" hissi yaratiyordu. Siki/agresif takip icin
    # esik kucultuldu (sadece cok yakinda hafif yavasla) ve min pitch yukseltildi.
    slowdown_start_radius_m: float = 18.0
    slowdown_min_pitch: float = 0.50    # ESKI: 0.55 -> hafif yumusatma
    terminal_follow_pitch_max: float = 0.30
    pursuit_min_pitch_command: float = 0.95
    pursuit_min_pitch_radius_m: float = 260.0
    # --- ROLL / BANKING (koordineli donus) ---
    # ONEMLI: Kamera SABIT, drone govdesine bagli (gimbal YOK). Roll yaptikca
    # kameranin GERCEK bakis acisi da fiziksel olarak yan yatiyor -- onceki
    # turlarda roll'u "gercek sabit kanat banking" gibi agresiflestirmek
    # tam da "kameradan hedefi tutamiyor" sorununun sebebiydi: drone yan
    # yattikca ufuk kayiyor, hedef goruntu disina cikiyor. Bu yuzden roll
    # artik SADECE GORUNUR/HAFIF bir destek (max ~%22 banka), donusun esas
    # agirligi yaw'da. Banking tamamen kaldirilmadi cunku "0 roll ile donus"
    # hedefi gene de kacirtiyordu (onceki sikayet) -- ama artik kamerayi
    # ciddi bozmayacak kadar kucuk.
    kp_roll: float = 0.22                # Yumusak hizalama: roll sadece ince destek
    roll_command_max: float = 0.18       # Sert bank/donusleri azaltmak icin dusuk cap
    roll_error_full_deg: float = 55.0    # ayni mantik: bu hatada MAX (kucuk) banka
    roll_deadband_deg: float = 1.5      # bu hata altinda roll sifirlanir (titremeyi onler)
    roll_speed_scale_min: float = 0.5   # dron cok yavasken banking etkisi azalir (stall riski)
    roll_speed_scale_full_mps: float = 8.0  # bu hizin ustunde tam banking uygulanir
    # --- GNSS kalite tepkisi ---
    gnss_reliable_approach_scale: float = 1.35
    gnss_suspect_approach_scale: float = 0.85
    gnss_unreliable_approach_scale: float = 0.35
    gnss_raw_jump_approach_scale: float = 0.20
    gnss_reliable_yaw_scale: float = 1.20
    gnss_suspect_yaw_scale: float = 0.75
    gnss_unreliable_yaw_scale: float = 0.45
    gnss_raw_jump_yaw_scale: float = 0.25
    gnss_reliable_throttle_scale: float = 1.10
    gnss_suspect_throttle_scale: float = 0.90
    gnss_unreliable_throttle_scale: float = 0.75
    gnss_raw_jump_throttle_scale: float = 0.75
    search_radius_min: float = 2.0
    uncertainty_slowdown_radius_m: float = 60.0
    uncertainty_min_scale: float = 0.55
    # --- Irtifa onceligi ---
    # ESKI: 6m'den sonra pitch'i 25m'de %50'ye kadar kisiyordu -- "GNSS modunda
    # hedefi siki/agresif takip etsin, pesini birakmasin" istegiyle celisiyordu.
    # Esikler buyutuldu ve minimum pitch scale yukseltildi: artik sadece COK
    # buyuk irtifa farklarinda (40m+) hafif kisitlama var.
    altitude_priority_start_m: float = 15.0
    altitude_priority_full_m: float = 40.0
    altitude_priority_min_pitch_scale: float = 0.75
    # --- THROTTLE ---
    kp_throttle: float = 0.06           # tanimli ama kullanilmiyor (kp_throttle_abs kullaniliyor)
    throttle_command_max: float = 1.0
    throttle_min_scale_when_turning: float = 1.0
    # --- Hedef irtifa ofseti (mesafeye gore kademeli sonum) ---
    # ESKI: sabit -30m -- yakinda dahi dron hedefin 30m USTUNDE ucmaya
    # calisiyordu, bu da "yaklasinca aniden tirmaniyor" hissinin sebebiydi.
    # YENI: uzaktayken guvenlik icin offset hala -30m (hedefin biraz ustunden
    # yaklas), ama mesafe dustukce ofset 0'a soner -> yakinda dron GERCEKTEN
    # hedefle ayni irtifaya iner (siki/yakin takip).
    # Quadcopter pursuit: uzakta hedefin 10m üstünden yaklaş (GNSS hatası toleransı),
    # yakında (20m altında) hedefle aynı irtifaya in. Eski -30m çok agresifti,
    # drone hedefin çok üstünden yaklaşıp son anda aşağı inmek zorunda kalıyordu.
    target_alt_offset_far_m: float = -20.0
    target_alt_offset_near_m: float = -20.0
    alt_offset_fade_start_radius_m: float = 60.0  # ESKI: 70.0
    alt_offset_fade_end_radius_m: float = 20.0
    target_alt_offset_m: float = 10.0  # GERIYE UYUMLULUK: artik dogrudan kullanilmiyor,
                                         # yukaridaki iki deger ile mesafeye gore hesaplaniyor
    # aim_point z bileseninin saniyede en fazla degisebilecegi miktar (m/s).
    # ESKI: alt_offset mesafeye gore SIFIR-ROTLAMA olmadan aninda hesaplaniyordu;
    # drone hedefe yaklasirken (60m->15m, birkac saniye) aim_z 30m birden
    # degisiyor, throttle da peslerinden tirmaniyordu -> "yaklasinca aniden
    # yukseliyor" hissi. Artik aim_z'nin KENDISI rate-limitli; throttle da
    # buna ZATEN yumusak bir gradyana tepki veriyor.
    # LOG ANALİZİ: hedef Z ekseninde 48-123m arası uçuyor (75m span), hızlı irtifa
    # değişimi yapıyor. 3 m/s rate limit ile aim_z hedefi takip etmekte zorlanıyor.
    # Quadcopter throttle ile irtifayı hızlı değiştirebilir, 8 m/s daha gerçekçi.
    aim_alt_max_rate_mps: float = 18.0
    # --- Gorsel devralma bolgesi ---
    visual_handoff_radius_m: float = 0.0
    arrival_radius_m: float = 15.0
    arrival_alt_tolerance_m: float = 6.0
    # --- Fast GNSS approach ---
    enable_fast_gnss_approach: bool = True
    # LOG ANALİZİ: search_radius RELIABLE'da bile ort 79m → strong_predict eşiği
    # 85m ile neredeyse her zaman GNSS_FAST_APPROACH moduna giriyor ama
    # SUSPECT'te medium_predict=60m, radius ort 80m → GNSS_SEARCH'e düşüyor.
    # SUSPECT için de fast approach istiyoruz. Eşikler yukarı çekildi.
    strong_predict_radius_m: float = 120.0  # ESKI: 85.0
    medium_predict_radius_m: float = 100.0  # ESKI: 60.0
    fast_fade_start_radius_m: float = 60.0  # ESKI: 45.0 → daha erken hızlan
    fast_stop_radius_m: float = 10.0        # Bozuk GNSS'i daha yakina kadar zorla
    pitch_command_fast: float = 1.0
    fast_throttle_boost: float = 0.0    # v5: throttle yatay hiz degil, sadece dikey komut
    base_hover_throttle: float = 0.0    # v5: throttle dikey komut; 0.0 hover/irtifa koruma
    kp_throttle_abs: float = 0.050
    high_altitude_pitch_cap: float = 0.55
    terminal_throttle_cap_radius_m: float = 18.0
    terminal_throttle_cap: float = 0.78
    # --- Lead point ---
    # v5 logunda hedef GPS'i ~5Hz ve gecikme kisa. Eski uzun gecikme varsayimi
    # yakin takipte hedefin cok onune nisan aldiriyordu; 0.4sn cap yeterli.
    target_nominal_speed_mps: float = 18.0   # v5 hedef log ortalamasi yaklasik 18 m/s
    lead_time_min_s: float = 0.08
    lead_time_max_s: float = 0.4
    lead_time_factor: float = 0.35
    # --- Uzak mesafe kapatma ---
    far_close_start_radius_m: float = 35.0
    far_close_full_radius_m: float = 85.0
    far_close_yaw_aligned_deg: float = 25.0
    far_close_pitch_boost: float = 0.45
    # --- Area search / UNRELIABLE intercept (lead-point kestirme) ---
    # ESKI: sabit pitch + yaw_amp=0 ama yaw_cmd hic sifirlanmiyordu -> dron son
    # bilinen (artik guvenilmez) GNSS noktasina kilitli kalip onun etrafinda
    # surekli donuyordu (yaw sifirlanmadigi icin "circle" olusuyordu).
    # YENI: son guvenilir nokta + tahmini hedef hizi * sabit bir intercept
    # suresi ile ONE KESTIRILMIS bir noktaya, NORMAL yaw/pitch mantigiyla
    # (kontrollu, slew-limited) gidilir. Hedef bulununca veya nokta civarina
    # gelininde dogal olarak yavaslar -- daire cizmez.
    area_search_intercept_time_s: float = 2.0   # UNRELIABLE icin tahmini hizla kac saniye ileri kestirilsin
    stale_area_search_intercept_time_s: float = 1.2
    lost_area_search_intercept_time_s: float = 0.8
    area_search_pitch: float = 0.55             # ESKI: 0.25 -> kestirme noktaya kararli ilerleme
    area_search_pitch_min: float = 0.30
    area_search_arrival_radius_m: float = 12.0  # kestirilen noktaya bu kadar yaklasinca yavasla
    area_search_yaw_amp: float = 0.0
    area_search_yaw_period_s: float = 6.0
    area_search_throttle_min: float = -0.55
    area_search_throttle_max: float = 0.55
    # --- Search center smoothing ---
    search_center_alpha_reliable: float = 0.25
    search_center_alpha_suspect: float = 0.12
    search_center_alpha_bad: float = 0.05
    # --- Slew-rate limiter ---
    # NOT: max_delta_yaw ve max_delta_roll, dronun ACISAL HIZINI fiilen
    # belirleyen parametreler -- "roll/yaw cok hizli yaptigi icin hedefi
    # kamerada goremiyor / irtifasinda kalamiyor" sikayetinin dogrudan
    # sebebi buydu. Ikisi de yumusatildi.
    max_delta_pitch: float = 0.25       # carpma takibinde pitch hizli cevap vermeli
    max_delta_yaw: float = 0.35         # Safety clamp; asil yumusatma alpha ile.
                                         # Dusuk olursa yaw merdiven gibi step-step akar.
    yaw_smoothing_alpha: float = 0.18   # Gercek low-pass yaw; dusuk deger daha akici.
    max_delta_throttle: float = 0.18    # v5 dikey komut; hedef irtifasina hizli cevap
    max_delta_roll: float = 0.04        # Banking hizi yumusak; ani/sert yatislari onler
    # --- Kontrol dongusu ---
    loop_hz: float = 50.0

    def __init__(self, **overrides):
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise TypeError(f"Unknown GuidanceConfig field: {key}")
            setattr(self, key, value)


def _wrap_angle_deg(angle_deg: float) -> float:
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a


def _bearing_to_target_deg(drone_xy, target_xy) -> float:
    dx = target_xy[0] - drone_xy[0]
    dy = target_xy[1] - drone_xy[1]
    return math.degrees(math.atan2(dy, dx))


class GuidanceController:
    def __init__(self, config: GuidanceConfig = None, analyzer: GnssReliabilityAnalyzer = None):
        self.cfg = config or GuidanceConfig()
        self.analyzer = analyzer or GnssReliabilityAnalyzer()
        self._last_log_t = 0.0
        self._last_reliable_center = None
        self._smoothed_search_center = None
        self._prev_target_pos_m = None
        self._prev_target_time = None
        self._prev_target_vel_m = None   # smoothed hız vektörü (EKF veya pozisyon farkı)
        self._prev_pitch_cmd = None
        self._prev_yaw_cmd = None
        self._prev_throttle_cmd = None
        self._prev_roll_cmd = None
        self._prev_aim_z_m = None
        self._prev_aim_z_time = None
        self._prev_guidance_mode = None
        self._close_bad_mode_count = 0
        self._prev_close_yaw_bearing_deg = None
        self._scan_start_t = time.time()

    def _is_strong_gnss_prediction(self, gnss_result):
        if getattr(gnss_result, "raw_jump_flag", False):
            return False
        radius = getattr(gnss_result, "search_radius", None)
        if radius is None:
            return False
        return gnss_result.status == GnssStatus.RELIABLE and radius <= self.cfg.strong_predict_radius_m

    def _is_medium_gnss_prediction(self, gnss_result):
        if getattr(gnss_result, "raw_jump_flag", False):
            return False
        radius = getattr(gnss_result, "search_radius", None)
        if radius is None:
            return False
        return gnss_result.status == GnssStatus.SUSPECT and radius <= self.cfg.medium_predict_radius_m

    def _fast_approach_fade(self, horizontal_dist_m):
        # KAMIKAZE MODU: ESKI mantik UZAKTA fast_gain=1 (agresif), YAKINDA
        # (fast_stop_radius_m altinda) fast_gain=0 (normal/yavas) idi --
        # "yaklas, sonra yavasla/devral" tasarimi icin dogruydu. Carpma
        # hedefli tasarimda bu TERSINE CEVRILDI: artik COK UZAKTA fast_gain
        # daha dusuk (henuz emin degiliz, normal yaklasim), YAKINA GELDIKCE
        # fast_gain 1'e cikar (maksimum agresiflik tam carpma aninda).
        cfg = self.cfg
        if horizontal_dist_m <= cfg.fast_stop_radius_m:
            return 1.0
        if horizontal_dist_m >= cfg.fast_fade_start_radius_m:
            return 0.0
        span = max(1e-6, cfg.fast_fade_start_radius_m - cfg.fast_stop_radius_m)
        return (cfg.fast_fade_start_radius_m - horizontal_dist_m) / span

    def _fast_camera_yaw_cap(self, horizontal_dist_m):
        cfg = self.cfg
        if horizontal_dist_m <= cfg.terminal_yaw_radius_m:
            return cfg.terminal_fast_yaw_command_max
        if horizontal_dist_m >= cfg.close_camera_yaw_radius_m:
            return cfg.yaw_command_max
        span = max(1e-6, cfg.close_camera_yaw_radius_m - cfg.terminal_yaw_radius_m)
        t = (horizontal_dist_m - cfg.terminal_yaw_radius_m) / span
        return cfg.terminal_fast_yaw_command_max + (
            (cfg.close_camera_yaw_command_max - cfg.terminal_fast_yaw_command_max) * t
        )

    def _area_search_intercept_time(self, gnss_status):
        status_value = getattr(gnss_status, "value", gnss_status)
        status_value = str(status_value).upper() if status_value is not None else ""
        if status_value == GnssStatus.LOST.value:
            return self.cfg.lost_area_search_intercept_time_s
        if status_value == GnssStatus.STALE.value:
            return self.cfg.stale_area_search_intercept_time_s
        return self.cfg.area_search_intercept_time_s

    @staticmethod
    def _smooth_point(prev, new, alpha):
        if prev is None:
            return new
        return (
            prev[0] * (1.0 - alpha) + new[0] * alpha,
            prev[1] * (1.0 - alpha) + new[1] * alpha,
            prev[2] * (1.0 - alpha) + new[2] * alpha,
        )

    @staticmethod
    def _slew_limit(prev, target, max_delta):
        if prev is None:
            return target
        delta = target - prev
        if delta > max_delta:
            return prev + max_delta
        if delta < -max_delta:
            return prev - max_delta
        return target

    @staticmethod
    def _smooth_command(prev, target, alpha):
        if prev is None:
            return target
        alpha = max(0.0, min(1.0, alpha))
        smoothed = prev + (target - prev) * alpha
        if abs(target) < 1e-6 and abs(smoothed) < 0.005:
            return 0.0
        return smoothed

    def seed_throttle_state(self, throttle: float):
        """
        TAKEOFF -> SEARCH gecisinde cagrilir. _prev_throttle_cmd None oldugu
        surece _slew_limit ilk komutu CIPLAK (rampasiz) geciriyordu; TAKEOFF
        thr=0.0'dan bitip SEARCH'in ilk adiminda dogrudan tirmanis boost'u
        verince bu ANI YUKSELME olarak hissediliyordu. Bu metod, gecis aninda
        slew state'ini takeoff'un SON throttle degeriyle baslatir, boylece ilk
        SEARCH adimi da normal rampa (max_delta_throttle) ile sinirlanir.
        """
        self._prev_throttle_cmd = throttle
        self._prev_pitch_cmd = 0.0
        self._prev_yaw_cmd = 0.0
        self._prev_roll_cmd = 0.0

    def _estimate_target_velocity_mps(self, target_pos_m, ekf_vel_mps=None):
        """
        Hız tahmini için EKF çıktısını (gnss_result'tan gelen ekf_vx/vy/vz) önceliklendir.
        EKF yoksa pozisyon farkından hesapla — ama sadece YENİ ölçüm geldiğinde güncelle
        (50Hz döngüde smoothed pos değişmediği için her adımda ~0 hız çıkıyordu).

        LOG ANALİZİ: gnss_reliability.py EKF zaten 18-19 m/s doğrulukta hız vektörü
        üretiyor. Bunu direkt kullanmak pozisyon farkından 50Hz'de hesaplamaktan çok
        daha temiz — smoothed pos her adımda çok az değiştiği için fark/dt ≈ 0 çıkıyor.
        """
        now = time.time()

        # EKF hız vektörü varsa onu kullan (birincil kaynak)
        if ekf_vel_mps is not None:
            vx, vy, vz = ekf_vel_mps
            speed = math.sqrt(vx*vx + vy*vy + vz*vz)
            if speed < 60.0:   # fiziksel limit: sabit kanat bu simde max ~60 m/s
                # Exponential smoothing: ani hız sıçramalarını yumuşat
                if self._prev_target_vel_m is None:
                    self._prev_target_vel_m = (vx, vy, vz)
                alpha = 0.35   # yeni ölçüme %35 ağırlık → stabil ama reaktif
                svx = self._prev_target_vel_m[0] * (1-alpha) + vx * alpha
                svy = self._prev_target_vel_m[1] * (1-alpha) + vy * alpha
                svz = self._prev_target_vel_m[2] * (1-alpha) + vz * alpha
                self._prev_target_vel_m = (svx, svy, svz)
                return (svx, svy, svz)

        # Fallback: pozisyon farkından hesapla (sadece konum değiştiğinde anlamlı)
        if self._prev_target_pos_m is None or self._prev_target_time is None:
            self._prev_target_pos_m = target_pos_m
            self._prev_target_time = now
            return self._prev_target_vel_m or (0.0, 0.0, 0.0)

        dt = now - self._prev_target_time
        # Sadece yeterince zaman geçtiyse güncelle (50Hz döngüde her adımda değil)
        if dt < 0.08:   # 80ms → ~12 Hz yeterli, her frame'de güncelleme gürültü üretir
            return self._prev_target_vel_m or (0.0, 0.0, 0.0)

        dx = target_pos_m[0] - self._prev_target_pos_m[0]
        dy = target_pos_m[1] - self._prev_target_pos_m[1]
        dz = target_pos_m[2] - self._prev_target_pos_m[2]
        self._prev_target_pos_m = target_pos_m
        self._prev_target_time = now

        vx, vy, vz = dx/dt, dy/dt, dz/dt
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed > 60.0:
            return self._prev_target_vel_m or (0.0, 0.0, 0.0)

        if self._prev_target_vel_m is None:
            self._prev_target_vel_m = (vx, vy, vz)
        alpha = 0.25
        svx = self._prev_target_vel_m[0] * (1-alpha) + vx * alpha
        svy = self._prev_target_vel_m[1] * (1-alpha) + vy * alpha
        svz = self._prev_target_vel_m[2] * (1-alpha) + vz * alpha
        self._prev_target_vel_m = (svx, svy, svz)
        return (svx, svy, svz)

    def _select_guidance_mode_raw(self, gnss_result, horizontal_dist_m):
        if gnss_result.raw_jump_flag:
            return "GNSS_AREA_SEARCH"
        if gnss_result.status == GnssStatus.RELIABLE:
            if (
                self.cfg.enable_fast_gnss_approach
                and gnss_result.search_radius <= self.cfg.strong_predict_radius_m
            ):
                return "GNSS_FAST_APPROACH"
            return "GNSS_SEARCH"
        if gnss_result.status == GnssStatus.SUSPECT:
            if (
                gnss_result.search_radius <= self.cfg.medium_predict_radius_m
                and horizontal_dist_m > self.cfg.fast_stop_radius_m
            ):
                return "GNSS_SOFT_APPROACH"
            return "GNSS_SEARCH"
        if gnss_result.status in (GnssStatus.UNRELIABLE, GnssStatus.STALE, GnssStatus.LOST):
            return "GNSS_AREA_SEARCH"
        return "GNSS_SEARCH"

    def _select_guidance_mode(self, gnss_result, horizontal_dist_m):
        proposed_mode = self._select_guidance_mode_raw(gnss_result, horizontal_dist_m)
        prev_mode = self._prev_guidance_mode

        if (
            prev_mode is not None
            and horizontal_dist_m <= self.cfg.close_mode_hysteresis_radius_m
            and proposed_mode == "GNSS_AREA_SEARCH"
            and not getattr(gnss_result, "raw_jump_flag", False)
            and prev_mode != "GNSS_AREA_SEARCH"
        ):
            self._close_bad_mode_count += 1
            if self._close_bad_mode_count < self.cfg.close_mode_bad_count_required:
                return prev_mode
        else:
            self._close_bad_mode_count = 0

        self._prev_guidance_mode = proposed_mode
        return proposed_mode

    def compute_command(self, drone_pos_m, drone_yaw_deg, target_pos_m, target_radius_m,
                        gnss_status=GnssStatus.RELIABLE, raw_jump_flag=False,
                        target_vel_mps=None, drone_speed_mps=0.0,
                        mode="GNSS_SEARCH"):
        cfg = self.cfg
        target_vel_mps = target_vel_mps or (0.0, 0.0, 0.0)

        # --- HEDEF ---
        # Yatay mesafeye (kaba, henuz lead-point uygulanmamis) gore irtifa
        # ofsetini kademeli sondur: uzakta hedefin biraz ustunden yaklas
        # (guvenlik + GNSS hatasi toleransi), yakinda GERCEKTEN hedefle ayni
        # irtifaya in (siki takip). Boylece "yaklasinca aniden tirmaniyor"
        # davranisi ortadan kalkar -- yakinda zaten tirmanacak bir sebep yok.
        raw_dx = target_pos_m[0] - drone_pos_m[0]
        raw_dy = target_pos_m[1] - drone_pos_m[1]
        raw_horizontal_dist = math.hypot(raw_dx, raw_dy)

        if raw_horizontal_dist >= cfg.alt_offset_fade_start_radius_m:
            alt_offset_t = 0.0
        elif raw_horizontal_dist <= cfg.alt_offset_fade_end_radius_m:
            alt_offset_t = 1.0
        else:
            span = max(1e-6, cfg.alt_offset_fade_start_radius_m - cfg.alt_offset_fade_end_radius_m)
            alt_offset_t = (cfg.alt_offset_fade_start_radius_m - raw_horizontal_dist) / span
        alt_offset = (
            cfg.target_alt_offset_far_m
            + (cfg.target_alt_offset_near_m - cfg.target_alt_offset_far_m) * alt_offset_t
        )

        raw_aim_z = target_pos_m[2] + alt_offset

        # aim_z rate-limit: alt_offset mesafeyle aninda degisse de, dronun
        # GERCEKTE takip ettigi irtifa hedefi saniyede en fazla
        # aim_alt_max_rate_mps kadar degisebilir. Bu, throttle'a giden
        # altitude_error'in (ve dolayisiyla aggressive_boost/vertical_correction'in)
        # ani sicrama yapmasini, kademeli ve kontrollu bir tirmanma/inis olmasini saglar.
        now_t = time.time()
        if self._prev_aim_z_m is None or self._prev_aim_z_time is None:
            aim_z = raw_aim_z
        else:
            dt = max(1e-3, now_t - self._prev_aim_z_time)
            max_step = cfg.aim_alt_max_rate_mps * dt
            delta = raw_aim_z - self._prev_aim_z_m
            if delta > max_step:
                aim_z = self._prev_aim_z_m + max_step
            elif delta < -max_step:
                aim_z = self._prev_aim_z_m - max_step
            else:
                aim_z = raw_aim_z
        self._prev_aim_z_m = aim_z
        self._prev_aim_z_time = now_t

        base_aim = (target_pos_m[0], target_pos_m[1], aim_z)
        aim_point = base_aim

        if mode == "GNSS_FAST_APPROACH":
            dx0 = base_aim[0] - drone_pos_m[0]
            dy0 = base_aim[1] - drone_pos_m[1]
            dz0 = base_aim[2] - drone_pos_m[2]
            dist_3d = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
            target_speed = math.sqrt(
                target_vel_mps[0] ** 2 +
                target_vel_mps[1] ** 2 +
                target_vel_mps[2] ** 2
            )
            if target_speed < 3.0:
                target_speed = cfg.target_nominal_speed_mps

            # closing_speed: drone kendi hızı + hedefin drone'a yaklaşma bileşeni.
            # ESKI: drone_speed + 0.35*target_speed — bu hedef kaçıyorsa closing
            # speed'i abartıyordu (ters yöne koşan birinin hızını toplayamazsın).
            # YENİ: sadece drone hızı kullan; lead_time_factor bunu telafi ediyor.
            closing_speed = max(5.0, drone_speed_mps)
            lead_time = dist_3d / closing_speed * cfg.lead_time_factor
            lead_time = max(cfg.lead_time_min_s, min(cfg.lead_time_max_s, lead_time))
            # NOT: yakın mesafede lead time küçültme KALDIRILDI.
            # ESKI: dist_3d < fast_fade_start_radius_m olunca lead_time *= 0.25..1.0
            # Bu, tam yakalamak üzereyken lead point'i hedefin çok ötesine götürüyor
            # ve overshoota neden oluyordu. Artık lead time mesafeden bağımsız sabit.
            aim_point = (
                base_aim[0] + target_vel_mps[0] * lead_time,
                base_aim[1] + target_vel_mps[1] * lead_time,
                base_aim[2] + target_vel_mps[2] * lead_time,
            )

        elif mode == "GNSS_AREA_SEARCH":
            # ESKI: GNSS guvenilmez oldugunda dron son (artik gecersiz) GNSS
            # noktasina kilitlenip yaw hic sifirlanmadigi icin o nokta etrafinda
            # surekli donuyordu (circle). YENI: son GUVENILIR konum + tahmini
            # hedef hizi * sabit bir intercept suresi ile ONE KESTIRILMIS bir
            # noktaya normal yaw/pitch mantigiyla kontrollu gidilir. Hedefin
            # hareketsiz/yavas oldugu durumda bu nokta ~ayni yerde kalir ve
            # dron oraya gidip dogal olarak yavaslar -- daire cizmez.
            intercept_time_s = self._area_search_intercept_time(gnss_status)
            aim_point = (
                base_aim[0] + target_vel_mps[0] * intercept_time_s,
                base_aim[1] + target_vel_mps[1] * intercept_time_s,
                base_aim[2],
            )

        target_horizontal_speed = math.hypot(target_vel_mps[0], target_vel_mps[1])
        follow_along_m = None
        follow_lateral_m = None
        # FAST_APPROACH da yakin bandda takip modudur: hedefin onune lead etmek
        # yakin geciste yaw sarmasi/overshoot uretir. Bu blok, guvenilir hiz
        # vektoru varken aim point'i hedefin arkasindaki stabil takip noktasina
        # alir; boylece fast yaklasma dogrudan smooth follow'a baglanir.
        if (
            mode != "GNSS_AREA_SEARCH"
            and raw_horizontal_dist <= cfg.follow_behind_radius_m
            and target_horizontal_speed >= 3.0
        ):
            ux = target_vel_mps[0] / target_horizontal_speed
            uy = target_vel_mps[1] / target_horizontal_speed
            rel_drone_x = drone_pos_m[0] - target_pos_m[0]
            rel_drone_y = drone_pos_m[1] - target_pos_m[1]
            follow_along_m = rel_drone_x * ux + rel_drone_y * uy
            follow_lateral_m = rel_drone_x * (-uy) + rel_drone_y * ux
            aim_point = (
                target_pos_m[0] - ux * cfg.follow_behind_distance_m,
                target_pos_m[1] - uy * cfg.follow_behind_distance_m,
                aim_point[2],
            )

        # --- Yatay mesafe ve bearing ---
        dx = aim_point[0] - drone_pos_m[0]
        dy = aim_point[1] - drone_pos_m[1]
        horizontal_dist = math.hypot(dx, dy)
        # Kamera govdenin onunde sabit: yaw normalde gorulmesi gereken hedef
        # IHA'ya bakar. Ancak cok yakinda drone hedefin onune gecmisse hedef
        # noktasi arkada/yanlarda kalir ve bearing 180 dereceye ziplar; hedef
        # duz ucsa bile yaw tek tarafa donup durur. Bu durumda burunu hedefin
        # ucus yonune hizalayip pitch'i kisiyoruz, hedef tekrar one gelsin.
        yaw_point_xy = (aim_point[0], aim_point[1]) if mode == "GNSS_AREA_SEARCH" else target_pos_m[:2]
        if (
            mode == "GNSS_FAST_APPROACH"
            and follow_along_m is not None
            and target_horizontal_speed >= 3.0
            and raw_horizontal_dist <= cfg.follow_velocity_yaw_radius_m
            and follow_along_m > -cfg.follow_velocity_yaw_ahead_margin_m
        ):
            ux = target_vel_mps[0] / target_horizontal_speed
            uy = target_vel_mps[1] / target_horizontal_speed
            yaw_point_xy = (
                target_pos_m[0] + ux * cfg.follow_behind_radius_m,
                target_pos_m[1] + uy * cfg.follow_behind_radius_m,
            )
        bearing_deg = _bearing_to_target_deg(drone_pos_m[:2], yaw_point_xy)
        if raw_horizontal_dist <= cfg.close_yaw_reference_hold_radius_m:
            if self._prev_close_yaw_bearing_deg is not None:
                bearing_delta = _wrap_angle_deg(bearing_deg - self._prev_close_yaw_bearing_deg)
                bearing_delta = max(
                    -cfg.close_yaw_reference_max_step_deg,
                    min(cfg.close_yaw_reference_max_step_deg, bearing_delta),
                )
                bearing_deg = _wrap_angle_deg(self._prev_close_yaw_bearing_deg + bearing_delta)
            self._prev_close_yaw_bearing_deg = bearing_deg
        else:
            self._prev_close_yaw_bearing_deg = None
        yaw_error_deg = _wrap_angle_deg(bearing_deg - drone_yaw_deg)
        yaw_error_rad = math.radians(yaw_error_deg)

        quality = self._gnss_quality_response(gnss_status, raw_jump_flag, target_radius_m)

        # --- YAW komutu ---
        if horizontal_dist >= cfg.slowdown_start_radius_m:
            kp_eff = cfg.kp_yaw
        else:
            t_interp = horizontal_dist / cfg.slowdown_start_radius_m
            kp_eff = cfg.kp_yaw_near + t_interp * (cfg.kp_yaw - cfg.kp_yaw_near)

        if abs(yaw_error_deg) < cfg.yaw_deadband_deg:
            yaw_cmd = 0.0
        else:
            yaw_cmd = kp_eff * yaw_error_rad
            yaw_cmd = max(-cfg.yaw_command_max, min(cfg.yaw_command_max, yaw_cmd))
            yaw_cmd *= quality["yaw_scale"]
            yaw_cmd = max(-cfg.yaw_command_max, min(cfg.yaw_command_max, yaw_cmd))

        terminal_control_dist = min(horizontal_dist, raw_horizontal_dist)

        yaw_cap = cfg.yaw_command_max
        if terminal_control_dist <= cfg.terminal_yaw_radius_m:
            yaw_cap = cfg.terminal_yaw_command_max
            yaw_cmd = max(-yaw_cap, min(yaw_cap, yaw_cmd))
        if (
            mode == "GNSS_FAST_APPROACH"
            and raw_horizontal_dist <= cfg.close_camera_yaw_radius_m
        ):
            yaw_cap = min(yaw_cap, self._fast_camera_yaw_cap(raw_horizontal_dist))
            yaw_cmd = max(-yaw_cap, min(yaw_cap, yaw_cmd))

        # --- YAW FACTOR ---
        # ESKI: donus 90 derecede pitch tam sifirlaniyordu -> banking yokken
        # dron donus icinde ivme/hiz kaybedip genis ark cizip hedefi kacirirdi.
        # YENI: yaw_error_for_zero_pitch_deg buyutuldu (130) VE alt sinir eklendi
        # (min_pitch_factor) -- artik agresif donuste de pitch tamamen olmuyor,
        # roll banking ile birlesince dron donus icinde de momentum tutuyor.
        yaw_factor_raw = max(0.0, 1.0 - abs(yaw_error_deg) / cfg.yaw_error_for_zero_pitch_deg)
        yaw_factor = cfg.yaw_error_for_min_pitch_factor + (
            (1.0 - cfg.yaw_error_for_min_pitch_factor) * yaw_factor_raw
        )

        # --- ROLL / BANKING komutu (koordineli donus) ---
        # Yaw hatasi ile orantili, hiza gore olceklenen bank acisi. Yaw hatasinin
        # ISARETI roll isaretini de belirler (saga don -> saga yat) -- boylece
        # roll yaw'i "destekler", donus yaricapi kuculur, dron hedefin disina
        # savrulmadan (overshoot etmeden) icine girer.
        if abs(yaw_error_deg) < cfg.roll_deadband_deg:
            roll_cmd = 0.0
        else:
            # DUZELTME: drone_sdk'da roll'un fiziksel isaret konvansiyonu yaw
            # ile AYNI degil -- pozitif roll_cmd gercekte SOLA yatiriyordu.
            # Gozlemle dogrulandi: yaw_error>0 (hedef saginda, saga donulmesi
            # gerekiyor) iken eski kod sola yatiriyordu (banking yaw'in TERSINE
            # calisiyordu, donus yaricapini buyutup hedefi daha cok kacirtiyordu).
            # Isaret ters cevrildi.
            roll_sign = -1.0 if yaw_error_deg > 0 else 1.0
            roll_mag_t = min(1.0, abs(yaw_error_deg) / cfg.roll_error_full_deg)
            roll_cmd_raw = roll_sign * cfg.kp_roll * roll_mag_t
            roll_cmd_raw = max(-cfg.roll_command_max, min(cfg.roll_command_max, roll_cmd_raw))

            speed_t = min(
                1.0,
                max(0.0, drone_speed_mps / max(1e-3, cfg.roll_speed_scale_full_mps)),
            )
            speed_scale = cfg.roll_speed_scale_min + (1.0 - cfg.roll_speed_scale_min) * speed_t
            roll_cmd = roll_cmd_raw * speed_scale
            roll_cmd *= quality["yaw_scale"]   # GNSS kotuyse banking de yumusasin
            roll_cmd = max(-cfg.roll_command_max, min(cfg.roll_command_max, roll_cmd))

        if terminal_control_dist <= cfg.terminal_yaw_radius_m:
            roll_cmd = max(
                -cfg.terminal_roll_command_max,
                min(cfg.terminal_roll_command_max, roll_cmd),
            )

        # --- PITCH komutu ---
        # KAMIKAZE MODU: ESKI slow_factor yakina gelince (slowdown_start_radius_m
        # altinda) pitch'i kademeli kisip dronu yavaslatiyordu -- "yaklasinca
        # devral/dur" mantigi icin dogruydu. Carpma hedefli tasarimda bu TERS
        # etki yapar: hedefe en yakinken en yavas olmak ister. Kaldirildi --
        # slow_factor artik HER ZAMAN 1.0 (yavaslama yok).
        slow_factor = 1.0

        fast_gain = 0.0
        if mode == "GNSS_FAST_APPROACH":
            fast_gain = self._fast_approach_fade(horizontal_dist)
        elif mode == "GNSS_SOFT_APPROACH":
            fast_gain = 0.35 * self._fast_approach_fade(horizontal_dist)

        pitch_base = (
            cfg.pitch_command_base * (1.0 - fast_gain)
            + cfg.pitch_command_fast * fast_gain
        )
        pitch_cmd = pitch_base * yaw_factor * slow_factor
        pitch_cmd *= quality["approach_scale"]
        pitch_cmd = max(0.0, min(1.0, pitch_cmd))

        # --- THROTTLE komutu ---
        # altitude_error: pozitif = drone ALCAKTA (tirman), negatif = drone YUKSEKTE (in)
        altitude_error = aim_point[2] - drone_pos_m[2]
        target_altitude_gap_m = target_pos_m[2] - drone_pos_m[2]

        altitude_priority_scale = self._altitude_priority_scale(altitude_error)
        pitch_cmd *= altitude_priority_scale
        pitch_cmd = max(0.0, min(1.0, pitch_cmd))

        if (
            cfg.terminal_yaw_radius_m < horizontal_dist <= cfg.pursuit_min_pitch_radius_m
        ):
            pursuit_min_pitch = cfg.pursuit_min_pitch_command * min(1.0, quality["approach_scale"])
            pitch_cmd = max(pursuit_min_pitch, pitch_cmd)

        # Hedefe kabaca hizaliyken ve mesafe acikken daha kararli kapan.
        # Yaw hatasi buyukken boost verme; once kamerayi/hedef yonunu toplasin.
        if abs(yaw_error_deg) < cfg.far_close_yaw_aligned_deg and altitude_error > -60.0:
            far_span = max(1e-6, cfg.far_close_full_radius_m - cfg.far_close_start_radius_m)
            far_t = (horizontal_dist - cfg.far_close_start_radius_m) / far_span
            far_t = max(0.0, min(1.0, far_t))
            yaw_align_t = 1.0 - (abs(yaw_error_deg) / max(1e-6, cfg.far_close_yaw_aligned_deg))
            pitch_cmd += cfg.far_close_pitch_boost * far_t * yaw_align_t * quality["approach_scale"]
            pitch_cmd = max(0.0, min(1.0, pitch_cmd))

        vertical_correction = cfg.kp_throttle_abs * altitude_error
        vertical_correction *= quality["throttle_scale"]

        aggressive_boost = cfg.fast_throttle_boost * fast_gain
        aggressive_boost *= quality["approach_scale"]

        # Drone hedeften yuksekteyse (alt_err negatif) boost kesinlikle 0
        if altitude_error < -2.0:
            aggressive_boost = 0.0
 
        throttle_cmd = cfg.base_hover_throttle + vertical_correction + aggressive_boost

        # ---------------------------------------------------------------
        # DUZELTME: Drone hedeften yuksekte (altitude_error < 0) iken
        # throttle pozitif OLAMAZ — bu durum boost veya hover_throttle
        # sebebiyle olusabiliyordu ve dronu sonsuza tirmaniyordu.
        # vertical_correction zaten negatif; throttle en fazla bu kadar olabilir.
        # ---------------------------------------------------------------
        if altitude_error < 0:
            throttle_cmd = min(throttle_cmd, vertical_correction)

        # KAMIKAZE MODU: ESKI terminal_throttle_cap, 18m altinda throttle'i
        # 0.78'e sinirlayip dronu "varis/devralma" icin yavaslatiyordu.
        # Carpma hedefli tasarimda bu kaldirildi -- yakinda da throttle
        # tam guc kullanabilir.

        throttle_cmd = max(-cfg.throttle_command_max, min(cfg.throttle_command_max, throttle_cmd))

        if mode == "GNSS_AREA_SEARCH":
            # ESKI: pitch sabit + yaw_cmd HIC sifirlanmiyordu -> dron gecersiz
            # son noktaya kilitlenip onun etrafinda donup duruyordu (circle).
            # YENI: aim_point zaten yukarida lead-point ile kestirilmis nokta;
            # buradaki yaw_cmd/roll_cmd de O noktaya gore (normal mantikla)
            # hesaplanmis durumda -- ekstra sifirlama YOK. Sadece pitch'i
            # kestirilen noktaya yaklasinca dogal olarak yavaslatiyoruz, boylece
            # nokta civarinda salinim/daire degil, kontrollu bir "yaklas ve
            # yavasla" davranisi olusur.
            area_t = min(1.0, horizontal_dist / max(1e-6, cfg.area_search_arrival_radius_m))
            pitch_cmd = cfg.area_search_pitch_min + (
                (cfg.area_search_pitch - cfg.area_search_pitch_min) * area_t
            )
            pitch_cmd *= yaw_factor   # hala donus icindeyse pitch'i tutarli kisitla

            # Nihai throttle karari asagida gnss_bad guard'inda verilir.
            # Burada sadece yatay arama pitch/yaw mantigini koruyoruz.

        status_value = getattr(gnss_status, "value", gnss_status)
        status_value = str(status_value).upper() if status_value is not None else GnssStatus.RELIABLE.value
        gnss_bad = raw_jump_flag or status_value in (
            GnssStatus.UNRELIABLE.value,
            GnssStatus.STALE.value,
            GnssStatus.LOST.value,
        )
        gnss_suspect = status_value == GnssStatus.SUSPECT.value
        is_area_search = mode == "GNSS_AREA_SEARCH"

        # v5'te throttle yatay hizdan bagimsiz dikey komut oldugu icin,
        # yuksekteyken de hedefin pesinden gitmeye devam et. Sadece asiri
        # irtifa fazlasinda ileri pitch'i hafif kis.
        if altitude_error < -12.0:
            pitch_cmd = min(pitch_cmd, cfg.high_altitude_pitch_cap)
        if altitude_error < -60.0:
            pitch_cmd = min(pitch_cmd, 0.45)

        if raw_horizontal_dist <= cfg.follow_behind_radius_m:
            follow_pitch_cap = cfg.follow_hold_pitch_max
            if follow_along_m is not None:
                desired_along_m = -cfg.follow_behind_distance_m
                if follow_along_m < desired_along_m - cfg.follow_catchup_gap_m:
                    follow_pitch_cap = cfg.follow_catchup_pitch_max
                elif follow_along_m > desired_along_m + cfg.follow_slow_gap_m:
                    follow_pitch_cap = cfg.follow_ahead_pitch_max
                if (
                    raw_horizontal_dist <= cfg.follow_speed_match_radius_m
                    and follow_along_m > 0.0
                    and target_horizontal_speed >= 3.0
                    and drone_speed_mps > target_horizontal_speed + cfg.follow_speed_match_speed_margin_mps
                ):
                    follow_pitch_cap = min(follow_pitch_cap, cfg.follow_speed_match_pitch_max)
            pitch_cmd = min(pitch_cmd, follow_pitch_cap)

        if terminal_control_dist <= cfg.terminal_yaw_radius_m:
            pitch_cmd = min(pitch_cmd, cfg.terminal_follow_pitch_max)
        if (
            mode == "GNSS_FAST_APPROACH"
            and raw_horizontal_dist <= cfg.terminal_yaw_radius_m
            and 0.0 <= target_altitude_gap_m <= cfg.below_target_slowdown_band_m
        ):
            pitch_cmd = min(pitch_cmd, cfg.below_target_slowdown_pitch_cap)
        if (
            mode == "GNSS_FAST_APPROACH"
            and cfg.hold_radius_m < raw_horizontal_dist <= cfg.hold_slowdown_radius_m
        ):
            span = max(1e-6, cfg.hold_slowdown_radius_m - cfg.hold_radius_m)
            hold_t = (raw_horizontal_dist - cfg.hold_radius_m) / span
            hold_cap = cfg.hold_pitch_cap + (
                (cfg.hold_slowdown_pitch_cap - cfg.hold_pitch_cap) * hold_t
            )
            pitch_cmd = min(pitch_cmd, hold_cap)
        if (
            mode == "GNSS_FAST_APPROACH"
            and cfg.hold_inner_radius_m < raw_horizontal_dist <= cfg.hold_radius_m
        ):
            span = max(1e-6, cfg.hold_radius_m - cfg.hold_inner_radius_m)
            hold_t = (raw_horizontal_dist - cfg.hold_inner_radius_m) / span
            hold_cap = cfg.hold_inner_pitch_cap + (
                (cfg.hold_pitch_cap - cfg.hold_inner_pitch_cap) * hold_t
            )
            pitch_cmd = min(pitch_cmd, hold_cap)
        if mode == "GNSS_FAST_APPROACH" and raw_horizontal_dist <= cfg.hold_inner_radius_m:
            pitch_cmd = min(pitch_cmd, cfg.hold_inner_pitch_cap)

        if gnss_suspect:
            pitch_cmd *= 0.65
            yaw_cmd *= 0.75
            roll_cmd *= 0.65
        if is_area_search:
            throttle_cmd = max(
                cfg.area_search_throttle_min,
                min(cfg.area_search_throttle_max, throttle_cmd),
            )
            roll_cmd = 0.0
        elif gnss_bad:
            pitch_cmd = 0.0
            roll_cmd = 0.0
            throttle_cmd = 0.0

        # --- Slew / smoothing ---
        pitch_cmd    = self._slew_limit(self._prev_pitch_cmd,    pitch_cmd,    cfg.max_delta_pitch)
        yaw_cmd      = self._smooth_command(self._prev_yaw_cmd,  yaw_cmd,      cfg.yaw_smoothing_alpha)
        yaw_cmd      = self._slew_limit(self._prev_yaw_cmd,      yaw_cmd,      cfg.max_delta_yaw)
        yaw_cmd      = max(-yaw_cap, min(yaw_cap, yaw_cmd))
        if gnss_bad and not is_area_search:
            throttle_cmd = 0.0
        else:
            throttle_cmd = self._slew_limit(self._prev_throttle_cmd, throttle_cmd, cfg.max_delta_throttle)
        roll_cmd     = self._slew_limit(self._prev_roll_cmd,     roll_cmd,     cfg.max_delta_roll)
        if terminal_control_dist <= cfg.terminal_yaw_radius_m:
            roll_cmd = max(
                -cfg.terminal_roll_command_max,
                min(cfg.terminal_roll_command_max, roll_cmd),
            )
        if mode == "GNSS_FAST_APPROACH" and raw_horizontal_dist <= cfg.terminal_yaw_radius_m:
            yaw_cmd = max(
                -cfg.terminal_fast_yaw_command_max,
                min(cfg.terminal_fast_yaw_command_max, yaw_cmd),
            )
            roll_cmd = max(
                -cfg.terminal_fast_roll_command_max,
                min(cfg.terminal_fast_roll_command_max, roll_cmd),
            )
        if is_area_search:
            roll_cmd = 0.0
            throttle_cmd = max(
                cfg.area_search_throttle_min,
                min(cfg.area_search_throttle_max, throttle_cmd),
            )
        elif gnss_bad:
            pitch_cmd = 0.0
            roll_cmd = 0.0
            throttle_cmd = 0.0

        self._prev_pitch_cmd    = pitch_cmd
        self._prev_yaw_cmd      = yaw_cmd
        self._prev_throttle_cmd = throttle_cmd
        self._prev_roll_cmd     = roll_cmd

        return {
            "throttle": throttle_cmd,
            "pitch": pitch_cmd,
            "roll": roll_cmd,
            "yaw": yaw_cmd,
            "horizontal_dist_m": horizontal_dist,
            "dist_3d_m": math.sqrt(
                horizontal_dist * horizontal_dist + altitude_error * altitude_error
            ),
            "yaw_error_deg": yaw_error_deg,
            "altitude_error_m": altitude_error,
            "target_altitude_gap_m": target_altitude_gap_m,
            "yaw_factor": yaw_factor,
            "guidance_mode": mode,
            "gnss_response_mode": quality["mode"],
            "approach_scale": quality["approach_scale"],
            "yaw_scale": quality["yaw_scale"],
            "throttle_scale": quality["throttle_scale"],
            "uncertainty_scale": quality["uncertainty_scale"],
            "altitude_priority_scale": altitude_priority_scale,
            "fast_gain": fast_gain,
            "aim_point": aim_point,
            "alt_offset_m": alt_offset,
            "raw_horizontal_dist_m": raw_horizontal_dist,
            "follow_along_m": follow_along_m,
            "follow_lateral_m": follow_lateral_m,
        }

    def _altitude_priority_scale(self, altitude_error_m):
        cfg = self.cfg
        abs_err = abs(altitude_error_m)
        if abs_err <= cfg.altitude_priority_start_m:
            return 1.0
        if abs_err >= cfg.altitude_priority_full_m:
            return cfg.altitude_priority_min_pitch_scale
        span = max(1e-6, cfg.altitude_priority_full_m - cfg.altitude_priority_start_m)
        t = (abs_err - cfg.altitude_priority_start_m) / span
        return 1.0 - ((1.0 - cfg.altitude_priority_min_pitch_scale) * t)

    def _gnss_quality_response(self, gnss_status, raw_jump_flag, search_radius_m):
        cfg = self.cfg
        if raw_jump_flag:
            mode = "RAW_JUMP"
            approach_scale  = cfg.gnss_raw_jump_approach_scale
            yaw_scale       = cfg.gnss_raw_jump_yaw_scale
            throttle_scale  = cfg.gnss_raw_jump_throttle_scale
        else:
            status_value = getattr(gnss_status, "value", gnss_status)
            status_value = str(status_value).upper() if status_value is not None else "RELIABLE"
            if status_value == GnssStatus.SUSPECT.value:
                mode = "SUSPECT"
                approach_scale  = cfg.gnss_suspect_approach_scale
                yaw_scale       = cfg.gnss_suspect_yaw_scale
                throttle_scale  = cfg.gnss_suspect_throttle_scale
            elif status_value == GnssStatus.UNRELIABLE.value:
                mode = "UNRELIABLE"
                approach_scale  = cfg.gnss_unreliable_approach_scale
                yaw_scale       = cfg.gnss_unreliable_yaw_scale
                throttle_scale  = cfg.gnss_unreliable_throttle_scale
            elif status_value in (GnssStatus.STALE.value, GnssStatus.LOST.value):
                mode = status_value
                approach_scale  = cfg.gnss_unreliable_approach_scale * 0.7
                yaw_scale       = cfg.gnss_unreliable_yaw_scale * 0.7
                throttle_scale  = cfg.gnss_unreliable_throttle_scale
            else:
                mode = "RELIABLE"
                approach_scale  = cfg.gnss_reliable_approach_scale
                yaw_scale       = cfg.gnss_reliable_yaw_scale
                throttle_scale  = cfg.gnss_reliable_throttle_scale

        if mode == "RELIABLE":
            uncertainty_scale = 1.0
        else:
            radius = cfg.search_radius_min if search_radius_m is None else max(cfg.search_radius_min, search_radius_m)
            span = max(1.0, cfg.uncertainty_slowdown_radius_m - cfg.search_radius_min)
            radius_t = min(1.0, max(0.0, (radius - cfg.search_radius_min) / span))
            uncertainty_scale = 1.0 - ((1.0 - cfg.uncertainty_min_scale) * radius_t)

        return {
            "mode": mode,
            "approach_scale":  approach_scale * uncertainty_scale,
            "yaw_scale":       yaw_scale * uncertainty_scale,
            "throttle_scale":  throttle_scale,
            "uncertainty_scale": uncertainty_scale,
        }

    def step(self):
        # --- Telemetriyi oku ---
        drone_pos_cm = drone.get_drone_location()
        drone_rot    = drone.get_drone_rotation()   # (roll, pitch, yaw) derece
        target_pos_cm = drone.get_target_location()

        drone_pos_m  = tuple(c * CM_TO_M for c in drone_pos_cm)
        drone_yaw_deg = drone_rot[2]

        # --- GNSS guvenilirlik analizi ---
        gnss_result  = self.analyzer.step(target_pos_cm, drone_pos_native=drone_pos_cm)
        target_pos_m = gnss_result.search_center

        if gnss_result.status == GnssStatus.RELIABLE and not gnss_result.raw_jump_flag:
            self._last_reliable_center = gnss_result.search_center

        if (
            gnss_result.raw_jump_flag
            or gnss_result.status in (GnssStatus.UNRELIABLE, GnssStatus.STALE, GnssStatus.LOST)
        ):
            if self._last_reliable_center is not None:
                target_pos_m = self._last_reliable_center

        if gnss_result.status == GnssStatus.RELIABLE:
            alpha = self.cfg.search_center_alpha_reliable
        elif gnss_result.status == GnssStatus.SUSPECT:
            alpha = self.cfg.search_center_alpha_suspect
        else:
            alpha = self.cfg.search_center_alpha_bad

        self._smoothed_search_center = self._smooth_point(
            self._smoothed_search_center, target_pos_m, alpha,
        )
        target_pos_m = self._smoothed_search_center

        dx = target_pos_m[0] - drone_pos_m[0]
        dy = target_pos_m[1] - drone_pos_m[1]
        horizontal_dist_pre = math.hypot(dx, dy)

        mode = self._select_guidance_mode(gnss_result, horizontal_dist_pre)

        # EKF hız vektörünü gnss_result'tan al — pozisyon farkından hesaplamaktan çok daha doğru
        # GnssAnalysisResult.ekf_vel zaten metre/s cinsinden EKF hız tahmini içeriyor
        ekf_vel = getattr(gnss_result, "ekf_vel", None)
        target_vel_mps  = self._estimate_target_velocity_mps(target_pos_m, ekf_vel_mps=ekf_vel)
        drone_speed_mps = drone.get_drone_speed() * CM_TO_M

        # --- Kontrol komutu hesapla ---
        cmd = self.compute_command(
            drone_pos_m,
            drone_yaw_deg,
            target_pos_m,
            gnss_result.search_radius,
            gnss_status=gnss_result.status,
            raw_jump_flag=gnss_result.raw_jump_flag,
            target_vel_mps=target_vel_mps,
            drone_speed_mps=drone_speed_mps,
            mode=mode,
        )

        cmd["drone_pos_m"]       = drone_pos_m
        cmd["drone_yaw_deg"]     = drone_yaw_deg
        cmd["drone_speed_mps"]   = drone_speed_mps
        cmd["target_pos_m"]      = target_pos_m
        cmd["guidance_mode"]     = mode
        cmd["nis"]               = getattr(gnss_result, "nis", 0.0)
        cmd["measurement_age_s"] = getattr(gnss_result, "measurement_age_s", 0.0)
        cmd["reason"]            = getattr(gnss_result, "reason", "")

        # --- Komutu gonder ---
        drone.set_control_surfaces(
            throttle=cmd["throttle"],
            pitch=cmd["pitch"],
            roll=cmd["roll"],
            yaw=cmd["yaw"],
            arm=True,
        )

        self._log(gnss_result, cmd)
        return gnss_result, cmd

    def is_close_enough_for_visual_handoff(self, cmd) -> bool:
        radius_m  = getattr(self.cfg, "visual_handoff_radius_m", self.cfg.arrival_radius_m)
        in_radius   = cmd["horizontal_dist_m"] <= radius_m
        in_altitude = abs(cmd["altitude_error_m"]) <= self.cfg.arrival_alt_tolerance_m
        return in_radius and in_altitude

    def has_reached_search_region(self, cmd) -> bool:
        return self.is_close_enough_for_visual_handoff(cmd)

    def _log(self, gnss_result, cmd):
        now = time.time()
        if now - self._last_log_t < 0.3:
            return
        self._last_log_t = now
        _append_mission_nav_csv(now, gnss_result, cmd)

        dp = cmd.get("drone_pos_m", (0, 0, 0))
        ap = cmd.get("aim_point",   (0, 0, 0))

        # DUZELTME: onceden horizontal_dist_m iki kez geciyordu (fast_gain yerine).
        # Simdi 14 format specifier = 14 arguman -- eslesiyor.
        logger.info(
            "GNSS=%s mode=%s radius=%.1fm nis=%.2f age=%.2fs reason=%s | "
            "drone_z=%.1fm aim_z=%.1fm alt_err=%.1fm | "
            "dist=%.1fm fast=%.2f yaw_err=%.1f deg yaw_f=%.2f | "
            "cmd(thr=%.2f pitch=%.2f roll=%.2f yaw=%.2f)",
            gnss_result.status.value,
            cmd.get("guidance_mode", "GNSS_SEARCH"),
            gnss_result.search_radius,
            getattr(gnss_result, "nis", 0.0),
            getattr(gnss_result, "measurement_age_s", 0.0),
            getattr(gnss_result, "reason", ""),
            dp[2],
            ap[2],
            cmd["altitude_error_m"],
            cmd["horizontal_dist_m"],
            cmd.get("fast_gain", 0.0),   # <-- duzeltme: onceden horizontal_dist_m tekrar yaziliyordu
            cmd["yaw_error_deg"],
            cmd["yaw_factor"],
            cmd["throttle"],
            cmd["pitch"],
            cmd["roll"],
            cmd["yaw"],
        )


def _fmt(value, digits=3):
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def _append_mission_nav_csv(t, gnss_result, cmd):
    global _mission_nav_csv_initialized
    mode = "a" if _mission_nav_csv_initialized else "w"
    dp = cmd.get("drone_pos_m", (0.0, 0.0, 0.0))
    tp = cmd.get("target_pos_m", (0.0, 0.0, 0.0))
    ap = cmd.get("aim_point", (0.0, 0.0, 0.0))
    tv = getattr(gnss_result, "ekf_vel", (0.0, 0.0, 0.0))

    with open(MISSION_NAV_CSV_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MISSION_NAV_CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerow({
            "t": _fmt(t, 3),
            "mode": cmd.get("guidance_mode", ""),
            "gnss_status": getattr(getattr(gnss_result, "status", ""), "value", ""),
            "reason": getattr(gnss_result, "reason", ""),
            "measurement_age_s": _fmt(getattr(gnss_result, "measurement_age_s", 0.0), 3),
            "drone_x": _fmt(dp[0], 2),
            "drone_y": _fmt(dp[1], 2),
            "drone_z": _fmt(dp[2], 2),
            "drone_yaw": _fmt(cmd.get("drone_yaw_deg", 0.0), 1),
            "drone_speed_mps": _fmt(cmd.get("drone_speed_mps", 0.0), 2),
            "target_x": _fmt(tp[0], 2),
            "target_y": _fmt(tp[1], 2),
            "target_z": _fmt(tp[2], 2),
            "target_vx": _fmt(tv[0], 2),
            "target_vy": _fmt(tv[1], 2),
            "target_vz": _fmt(tv[2], 2),
            "aim_x": _fmt(ap[0], 2),
            "aim_y": _fmt(ap[1], 2),
            "aim_z": _fmt(ap[2], 2),
            "horizontal_dist_m": _fmt(cmd.get("horizontal_dist_m"), 2),
            "raw_horizontal_dist_m": _fmt(cmd.get("raw_horizontal_dist_m"), 2),
            "dist_3d_m": _fmt(cmd.get("dist_3d_m"), 2),
            "altitude_error_m": _fmt(cmd.get("altitude_error_m"), 2),
            "target_altitude_gap_m": _fmt(cmd.get("target_altitude_gap_m"), 2),
            "yaw_error_deg": _fmt(cmd.get("yaw_error_deg"), 1),
            "follow_along_m": _fmt(cmd.get("follow_along_m"), 2),
            "follow_lateral_m": _fmt(cmd.get("follow_lateral_m"), 2),
            "search_radius": _fmt(getattr(gnss_result, "search_radius", 0.0), 2),
            "nis": _fmt(getattr(gnss_result, "nis", 0.0), 3),
            "uncertainty_scale": _fmt(cmd.get("uncertainty_scale"), 3),
            "approach_scale": _fmt(cmd.get("approach_scale"), 3),
            "fast_gain": _fmt(cmd.get("fast_gain"), 3),
            "cmd_throttle": _fmt(cmd.get("throttle"), 4),
            "cmd_pitch": _fmt(cmd.get("pitch"), 4),
            "cmd_roll": _fmt(cmd.get("roll"), 4),
            "cmd_yaw": _fmt(cmd.get("yaw"), 4),
            "armed": 1,
        })
    _mission_nav_csv_initialized = True


# =============================================================================
# ANA KONTROL DONGUSU (standalone test)
# =============================================================================
def run_guidance_loop(vision_override_check=None):
    logger.info("drone_sdk baglantisi deneniyor (127.0.0.1:12345)...")
    if not drone.connect():
        logger.error("Baglanti basarisiz. Unreal sim 'Play' modunda mi?")
        return
    logger.info("Baglandi. ARM ediliyor ve ARAMA FAZI baslatiliyor.")
    logger.info("Cikmak icin Ctrl+C.")

    controller = GuidanceController()
    dt_target  = 1.0 / controller.cfg.loop_hz

    try:
        while True:
            loop_start = time.time()
            if vision_override_check is not None and vision_override_check():
                time.sleep(dt_target)
                continue
            gnss_result, cmd = controller.step()
            if controller.is_close_enough_for_visual_handoff(cmd):
                logger.info(
                    "Gorsel devralma bolgesinde (mesafe=%.1fm <= %.1fm). "
                    "Tespit gelene kadar GNSS yaklasmasi devam ediyor.",
                    cmd["horizontal_dist_m"], controller.cfg.visual_handoff_radius_m,
                )
            elapsed    = time.time() - loop_start
            sleep_time = max(0.0, dt_target - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Durduruldu (Ctrl+C). Dron disarm ediliyor.")
        drone.set_control_surfaces(0.0, 0.0, 0.0, 0.0, arm=False)
    finally:
        drone.disconnect()


if __name__ == "__main__":
    run_guidance_loop()
