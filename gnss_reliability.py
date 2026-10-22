"""
================================================================================
AVCI DRONE - GNSS GUVENILIRLIK ANALIZI (CUSUM + Adaptif EKF)
================================================================================
Amac:
  Yarisma sunucusundan gelen HEDEF GNSS verisi (drone.get_target_location)
  kasitli olarak bozulmus olabilir (gurultu, sabit kayma, ani ziplama,
  dropout, rate-limit, gecikme). Bu modul:

    1) Gelen her olcumu bir EKF (sabit hiz modeli, CV-EKF) ile filtreler.
    2) Innovation (olcum - tahmin) buyuklugu uzerinde iki yonlu CUSUM testi
       calistirarak YAVAS/SUREKLI sapmalari (offset, drift, rate-limit'in
       yarattigi tutarsizlik) yakalar.
    3) Ardisik ham olcumler arasi fiziksel olarak imkansiz sicramalari
       (ani spike) ayrica, CUSUM'dan BAGIMSIZ bir "raw jump" testiyle yakalar
       (CUSUM'un birikip alarm vermesini beklemeden ANINDA tepki).
    4) Bu ikisinin sonucuna gore GNSS guvenilirlik durumunu (RELIABLE /
       SUSPECT / UNRELIABLE) belirler ve EKF adaptasyonunu buna gore yapar:
         - RELIABLE  -> olcum normal guvenle EKF'e enjekte edilir.
         - SUSPECT   -> olcum gurultusu (R) buyutulerek dusuk agirlikla
                        enjekte edilir (EKF'i bozuk veriye karsi yumusatir).
         - UNRELIABLE-> olcum HIC enjekte edilmez; sadece EKF predict adimi
                        calisir (kor ucus). Belirsizlik (P) buyumeye devam
                        eder -> "olasi arama bolgesi" (pozisyon + yaricap)
                        buyuyerek genisler.
    5) Her adimi konsola ve CSV'ye loglar; varsa get_active_corruption() ile
       o anki GERCEK bozulma etiketini (sadece bilgi/karsilastirma icin,
       karar mekanizmasi BUNU KULLANMAZ) ayrica kaydeder.

Bu dosya STANDALONE calisir: sadece drone_sdk'ya baglanir, dronu ARM ETMEZ,
sadece hedef telemetrisini okuyup GNSS guvenilirlik analizini canli loglar.
Yonelme/CRSF komut katmani bu asamada YOK (siradaki modul).
================================================================================
"""

import csv
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

import drone_sdk as drone

# =============================================================================
# LOGLAMA
# =============================================================================
logger = logging.getLogger("gnss_reliability")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                       datefmt="%H:%M:%S"))
    logger.addHandler(_h)

CSV_PATH = "gnss_reliability_log.csv"
CSV_FIELDS = [
    "t", "dt",
    # --- Hedef (target) ---
    "raw_x", "raw_y", "raw_z",
    "ekf_x", "ekf_y", "ekf_z",
    "ekf_vx", "ekf_vy", "ekf_vz",
    "innov_norm",
    "nis",
    "cusum_pos", "cusum_neg",
    "raw_jump_dist", "raw_jump_flag",
    "status",
    "search_radius",
    "measurement_age_s",
    "is_new_measurement",
    "reason",
    "ground_truth_corruption",  # sadece bilgi - karar buna bakmaz
    # --- Drone (avcı) ---
    "drone_x", "drone_y", "drone_z",       # pozisyon (m)
    "drone_vx", "drone_vy", "drone_vz",    # hız vektörü (m/s)
    "drone_speed_mps",                     # toplam hız (m/s)
    "drone_yaw_deg",                       # yaw açısı
    # --- Türetilmiş metrikler ---
    "dist_horizontal_m",   # yatay mesafe drone-hedef
    "dist_3d_m",           # 3D mesafe drone-hedef
    "alt_error_m",         # irtifa farkı: hedef_z - drone_z (+ = drone alçakta)
]


# =============================================================================
# GUVENILIRLIK DURUMU
# =============================================================================
class GnssStatus(Enum):
    RELIABLE = "RELIABLE"
    SUSPECT = "SUSPECT"
    UNRELIABLE = "UNRELIABLE"
    STALE = "STALE"
    LOST = "LOST"


# =============================================================================
# AYARLANABILIR PARAMETRELER
# =============================================================================
@dataclass
class GnssReliabilityConfig:
    # --- BIRIM DONUSUMU ---
    # ONEMLI: drone_sdk.get_target_location() Unreal Engine'in varsayilan
    # birimiyle (CENTIMETRE) konum donduruyor (gercek logda dogrulandi:
    # ardisik farkli olcumler arasi ~17800-18000 birim/s hiz gozlemlendi,
    # bu m/s olarak imkansiz (640+ km/h) ama cm/s olarak makul bir UAV/hedef
    # hizi (~178-180 m/s = ~640 km/h, rate-limited GNSS ornekleme araliginda
    # biriken mesafe nedeniyle boyle gorunuyor). EKF ve butun fizik tabanli
    # esikler METRE uzerinden tanimli; bu yuzden ham veri SDK'dan gelir gelmez
    # bu carpanla metreye cevrilir, btun config esikleri METRE cinsindendir.
    input_unit_scale: float = 0.01  # cm -> m (Unreal varsayilani). Sim baska
                                     # birim kullaniyorsa burayi degistirin.

    # --- EKF surec/olcum gurultusu (baslangic varsayimlari, METRE biriminde) ---
    process_noise_pos: float = 1.4       # Q: pozisyon surec gurultusu (hizli hareket eden hedef icin yuksek tutuldu)
    process_noise_vel: float = 3.0       # Q: hiz surec gurultusu (rate-limited GNSS -> hiz tahmini hizli adapte olmali)
    process_noise_bias: float = 0.002    # Q: BIAS surec gurultusu - COK KUCUK ("sabit offset" varsayimi:
                                          # bias zamanla neredeyse degismez, bu yuzden EKF'in bias tahminini
                                          # her olcumde rasgele dolastirmasini onlemek icin dt ile carpilsa
                                          # bile ihmal edilebilir kalmali; gercek logda offset 1492 sn boyunca
                                          # sabit kalmis -> bias gercekten "sabit" davraniyor).
    meas_noise_base: float = 9.0         # R: bias AYRISTIRILDIKTAN SONRA kalan gercek sensor gurultusu
                                          # varyansi (m^2) -> ~3m std. (Eskiden 25.0 idi; o zaman bias
                                          # ayristirilmiyordu ve R'nin offset+gecikme etkisini de
                                          # "tolere etmesi" gerekiyordu. Artik bias ayri kestirildigi icin
                                          # R sadece GERCEK olcum gurultusunu (~konum gurultusu +-0.1m,
                                          # hiz gurultusunun pozisyona yansimasi) temsil etmeli.)
    meas_noise_suspect_mult: float = 12.0  # SUSPECT durumunda R'yi bu katla buyut (olcume az guven)

    # --- CUSUM parametreleri (innovation normu uzerinde) ---
    # ONEMLI: Bias artik EKF durumunda ayrica kestirildigi icin, CUSUM'a giden
    # innovation BIAS DUSULMUS innovation'dir (bkz. ConstantVelocityEKF.update).
    # Bu yuzden CUSUM artik "kalici offset+gecikme" gibi YAPISAL/SUREKLI bir
    # sapmayi degil, GERCEKTEN ANORMAL (spoofing, ani sicrama, beklenmeyen
    # davranis degisikligi) durumlari yakalamali. Esikler buna gore daha
    # SIKI tutulabilir (eskiden gevsekti cunku sistematik sapmayi da
    # "tolere etmesi" gerekiyordu).
    cusum_k: float = 2.5     # referans deger (slack)
    cusum_h: float = 10.0    # alarm esigi - bu deger asilinca SUSPECT
    cusum_h_unreliable: float = 25.0  # CUSUM bu kadar buyurse direkt UNRELIABLE'a gec

    # --- Raw jump (ham siçrama) testi: CUSUM'dan bagimsiz, anlik ---
    # Hedef icin "fiziksel olarak makul" maksimum hiz varsayimi (m/s).
    # Gercek logda gozlemlenen normal-rejim hiz ~178-180 m/s; GNSS'in kendi
    # rate-limit/gecikme etkisiyle olusan mesafe sicramalarini RAW JUMP
    # SANMAMAK icin bu deger gercek hedef hizinin agirlikli ustunde tutulur.
    max_plausible_speed_ms: float = 220.0
    raw_jump_margin: float = 1.5   # guvenlik payi carpani (max_plausible_speed * margin)

    # --- Ardisik SUSPECT/UNRELIABLE sayaci esikleri ---
    suspect_streak_to_unreliable: int = 5   # ust uste bu kadar SUSPECT -> UNRELIABLE'a yukselt

    # --- Arama bolgesi (search region) ---
    search_radius_sigma: float = 3.0  # P kovaryansindan kac sigma ile yaricap hesapla
    search_radius_min: float = 2.0    # metre, alt sinir (RELIABLE haldeyken bile sifir olmasin)

    # --- EKF warm-up: bağlantı sonrası gelen (0,0,0) gibi "henüz dolmamış"
    # ilk telemetri karelerini EKF init'i için GEÇERSİZ saymak için kaç örnek
    # bekleyelim (bu sürede EKF hiç initialize edilmez, ilk gerçek veri gelince
    # initialize edilir).
    warmup_max_samples: int = 20

    stale_epsilon_m: float = 0.05
    lost_timeout_s: float = 1.4
    stale_timeout_s: float = 0.45

    nis_reliable_gate: float = 30.0
    nis_unreliable_gate: float = 120.0

    min_measurement_dt_s: float = 0.05


# =============================================================================
# BASIT 3D SABIT-HIZ (CONSTANT VELOCITY) EKF
# Durum vektoru: [x, y, z, vx, vy, vz]
# Olcum: [x, y, z]  (dogrusal H -> aslinda KF ile ozdes ama "EKF" olarak
#                    birakildi: ileride non-linear olcum modeli -orn. menzil/
#                    acisal olcum- eklemek icin genisletilebilir iskelet budur)
# =============================================================================
class ConstantVelocityEKF:
    """
    Durum vektoru artik 9 boyutlu: [x, y, z, vx, vy, vz, bx, by, bz]
    (pozisyon + hiz + KALICI BIAS).

    NEDEN BIAS DURUMU EKLENDI:
    Gercek sim verisinde GNSS bozulmasi GECICI bir "olay" degil, SUREKLI
    AKTIF bir durum olabilir (sabit offset, gecikme ve guncelleme hizi siniri
    ayni anda gorulebilir). Bu etkiler birlesince ham olcum ile "su anki gercek
    pozisyon" arasinda kalici bir fark olusur. Bu fark hic SIFIRA INMIYOR, bu yuzden
    klasik CUSUM (innovation'in ge\u00e7ici sapmasini yakalamak icin tasarlanmis)
    bunu surekli "anomali" sanip sistemi surekli UNRELIABLE/SUSPECT'e itiyor
    (gercek logda 287 ayri UNRELIABLE bloğu, ortalama 0.14s - yani sistem
    surekli RELIABLE<->UNRELIABLE arasinda \"cirpiniyor\").

    COZUM: Bias'i (bx,by,bz) ayri bir durum degiskeni olarak EKF'in kendisi
    KESTIRSIN. Olcum modeli artik: z_olcum = pos_gercek + bias + gurultu
    EKF zamanla "bu sabit ~(2,2,1)m'lik kaymayi" ogrenip cikarir; CUSUM'a
    giden innovation, bias CIKARILMIS innovation olur -> CUSUM sadece
    GERCEKTEN ANORMAL (bias'in da aciklayamadigi) sapmalari yakalar.

    Bias surec gurultusu (process_noise_bias) COK KUCUK tutulur (bias yavas
    degisir/sabittir varsayimi); pozisyon/hiz surec gurultusu ise normal
    kalir (hedef hareket edebilir).
    """

    N = 9  # durum boyutu

    def __init__(self, cfg: GnssReliabilityConfig):
        self.cfg = cfg
        self.x = [0.0] * self.N      # durum: x,y,z,vx,vy,vz,bx,by,bz
        self.P = self._init_P()
        self.initialized = False
        self.last_t = None

    def _init_P(self):
        # Baslangic belirsizligi genis tutulur (ilk olcumden once hicbir sey
        # bilinmiyor). Bias icin de makul bir baslangic belirsizligi (bias'in
        # +-30m araliginda olabilecegini varsayan bir std) veriyoruz.
        n = self.N
        P = [[0.0] * n for _ in range(n)]
        for i in range(3):
            P[i][i] = 50.0 ** 2     # pozisyon: 50m std baslangic
        for i in range(3, 6):
            P[i][i] = 30.0 ** 2     # hiz: 30 m/s std baslangic
        for i in range(6, 9):
            P[i][i] = 30.0 ** 2     # bias: 30m std baslangic (gercek logdaki ~2-20m mertebesini kapsar)
        return P

    def initialize(self, pos):
        # Bias baslangicta 0 kabul edilir; ilk birkac olcumde EKF bunu
        # kendisi ogrenip ayarlayacaktir.
        self.x = [pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.P = self._init_P()
        self.initialized = True

    # ---- Matris yardimcilari (NxN, numpy'siz - boyuttan bagimsiz calisir) ----
    @staticmethod
    def _mat_mul(A, B):
        rA, cA = len(A), len(A[0])
        rB, cB = len(B), len(B[0])
        assert cA == rB
        out = [[0.0] * cB for _ in range(rA)]
        for i in range(rA):
            for k in range(cA):
                if A[i][k] == 0.0:
                    continue
                for j in range(cB):
                    out[i][j] += A[i][k] * B[k][j]
        return out

    @staticmethod
    def _transpose(A):
        return [list(row) for row in zip(*A)]

    @staticmethod
    def _mat_add(A, B):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _mat_sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _inv3(M):
        # 3x3 matris tersini analitik (cofactor) yontemle al -> numpy bagimliligi yok
        a, b, c = M[0]
        d, e, f = M[1]
        g, h, i = M[2]
        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        if abs(det) < 1e-9:
            det = 1e-9 if det >= 0 else -1e-9
        inv_det = 1.0 / det
        return [
            [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
            [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
            [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
        ]

    def predict(self, dt):
        if dt <= 0:
            return
        cfg = self.cfg
        x = self.x
        n = self.N
        # Durum gecisi: pozisyon += hiz*dt (sabit hiz modeli); bias DEGISMEZ
        # (F'de bias-bias bloğu birim matris, bias'in kendi process_noise'u
        # COK KUCUK -> "yavas degisen/sabit" varsayimi).
        new_x = list(x)
        new_x[0] = x[0] + x[3] * dt
        new_x[1] = x[1] + x[4] * dt
        new_x[2] = x[2] + x[5] * dt
        # x[3:6] (hiz) ve x[6:9] (bias) degismez (sabit hiz / sabit bias modeli)

        # F (durum gecis matrisi) - NxN birim matris + pozisyon<-hiz bloğu
        F = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        F[0][3] = dt
        F[1][4] = dt
        F[2][5] = dt
        Ft = self._transpose(F)

        # Surec gurultusu Q (basit diagonal, dt ile olceklenir)
        Q = [[0.0] * n for _ in range(n)]
        for i in range(3):
            Q[i][i] = cfg.process_noise_pos * dt
        for i in range(3, 6):
            Q[i][i] = cfg.process_noise_vel * dt
        for i in range(6, 9):
            Q[i][i] = cfg.process_noise_bias * dt  # cok kucuk: bias "sabit" kabul edilir

        P_pred = self._mat_mul(self._mat_mul(F, self.P), Ft)
        P_pred = self._mat_add(P_pred, Q)

        self.x = new_x
        self.P = P_pred

    def update(self, meas, R_scale=1.0):
        """meas: (x,y,z) HAM olcum (bias dahil, henuz ayiklanmamis).
        Olcum modeli: z = pos + bias + gurultu  =>  H = [I3 | 0 | I3]
        (pozisyon sutunlari birim, hiz sutunlari sifir, BIAS SUTUNLARI BIRIM).
        Bu sayede innovation = olcum - (pos_tahmin + bias_tahmin) olur; EKF
        zamanla bias'i ogrenip pos_tahmin'i GERCEK pozisyona yakinsatir.
        R_scale ile olcum gurultusunu adaptif buyut/kucult.
        Doner: innovation vektoru (y) ve innovation kovaryansi (S) -- CUSUM
        ve loglama bunlari kullanir."""
        cfg = self.cfg
        n = self.N
        H = [[0.0] * n for _ in range(3)]
        for i in range(3):
            H[i][i] = 1.0       # pozisyon sutunlari
            H[i][i + 6] = 1.0   # bias sutunlari (AYNI eksenin biası, ayni satirda)

        R = [[cfg.meas_noise_base * R_scale, 0, 0],
             [0, cfg.meas_noise_base * R_scale, 0],
             [0, 0, cfg.meas_noise_base * R_scale]]

        # innovation y = z - H x = olcum - (pos_tahmin + bias_tahmin)
        Hx = [self.x[0] + self.x[6], self.x[1] + self.x[7], self.x[2] + self.x[8]]
        y = [meas[0] - Hx[0], meas[1] - Hx[1], meas[2] - Hx[2]]

        # S = H P H^T + R
        Ht = self._transpose(H)
        HP = self._mat_mul(H, self.P)          # 3xN
        HPHt = self._mat_mul(HP, Ht)           # 3x3
        S = self._mat_add(HPHt, R)
        S_inv = self._inv3(S)

        # K = P H^T S^-1  (Nx3)
        PHt = self._mat_mul(self.P, Ht)        # Nx3
        K = self._mat_mul(PHt, S_inv)          # Nx3

        # x = x + K y
        Ky = [sum(K[i][j] * y[j] for j in range(3)) for i in range(n)]
        self.x = [self.x[i] + Ky[i] for i in range(n)]

        # P = (I - K H) P
        KH = self._mat_mul(K, H)               # NxN
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        I_KH = self._mat_sub(I, KH)
        self.P = self._mat_mul(I_KH, self.P)

        return y, S

    def predicted_measurement(self):
        return (
            self.x[0] + self.x[6],
            self.x[1] + self.x[7],
            self.x[2] + self.x[8],
        )

    def innovation_stats(self, meas, R_scale=1.0):
        cfg = self.cfg
        n = self.N

        H = [[0.0] * n for _ in range(3)]
        for i in range(3):
            H[i][i] = 1.0
            H[i][i + 6] = 1.0

        R = [
            [cfg.meas_noise_base * R_scale, 0.0, 0.0],
            [0.0, cfg.meas_noise_base * R_scale, 0.0],
            [0.0, 0.0, cfg.meas_noise_base * R_scale],
        ]

        Hx = self.predicted_measurement()
        y = [
            meas[0] - Hx[0],
            meas[1] - Hx[1],
            meas[2] - Hx[2],
        ]

        Ht = self._transpose(H)
        HP = self._mat_mul(H, self.P)
        HPHt = self._mat_mul(HP, Ht)
        S = self._mat_add(HPHt, R)
        S_inv = self._inv3(S)

        nis = 0.0
        for i in range(3):
            for j in range(3):
                nis += y[i] * S_inv[i][j] * y[j]

        return y, S, nis

    def position(self):
        """BIAS DUZELTILMIS en iyi pozisyon tahmini (gercek konum tahmini)."""
        return (self.x[0], self.x[1], self.x[2])

    def velocity(self):
        return (self.x[3], self.x[4], self.x[5])

    def bias(self):
        """O anda kestirilen kalici bias (debug/log amacli)."""
        return (self.x[6], self.x[7], self.x[8])

    def position_std(self):
        return (math.sqrt(max(self.P[0][0], 0.0)),
                math.sqrt(max(self.P[1][1], 0.0)),
                math.sqrt(max(self.P[2][2], 0.0)))


# =============================================================================
# IKI YONLU CUSUM (innovation normu uzerinde)
# =============================================================================
class TwoSidedCusum:
    """Klasik CUSUM: g_pos ve g_neg birikimli sapma istatistikleri.
    Burada innovation normu HER ZAMAN pozitif oldugu icin (norm),
    "iki yonlu" CUSUM'u innovation normunun BEKLENEN degerden (≈0, normalize
    sonrasi ≈1 sigma) sapmasini iki yonde de izlemek icin kullaniyoruz:
    g_pos -> normun beklenenden BUYUK olmasi (ani/asiri sapma, spike/jump)
    g_neg -> burada pratikte az bilgi tasir ama simetri icin tutuluyor
             (ornegin sensorun "asiri durgun/sifir gurultu" vermesi -> spoof
             belirtisi olabilecek anormal durumlar icin genisletilebilir)."""

    def __init__(self, k: float, h: float):
        self.k = k
        self.h = h
        self.g_pos = 0.0
        self.g_neg = 0.0

    def update(self, normalized_innovation: float):
        # normalized_innovation: innovation normu / beklenen std (yaklasik ki-kare
        # benzeri bir buyukluk degil, basit skaler "kac sigma" olcusu)
        s = normalized_innovation - self.k
        self.g_pos = max(0.0, self.g_pos + s)
        s_neg = (-normalized_innovation) - self.k
        self.g_neg = max(0.0, self.g_neg + s_neg)
        return self.g_pos, self.g_neg

    def is_alarm(self):
        return (self.g_pos >= self.h) or (self.g_neg >= self.h)

    def is_severe_alarm(self, h_severe: float):
        return (self.g_pos >= h_severe) or (self.g_neg >= h_severe)

    def reset(self):
        self.g_pos = 0.0
        self.g_neg = 0.0


# =============================================================================
# ANALIZ SONUCU (her adimda donen, yonelme modulunun de kullanacagi yapi)
# =============================================================================
@dataclass
class GnssAnalysisResult:
    t: float
    dt: float
    raw_pos: tuple
    ekf_pos: tuple
    ekf_vel: tuple
    innovation_norm: float
    cusum_pos: float
    cusum_neg: float
    raw_jump_dist: float
    raw_jump_flag: bool
    status: GnssStatus
    search_center: tuple   # = ekf_pos (en iyi tahmin)
    search_radius: float   # metre - "olasi arama bolgesi" yaricapi
    nis: float = 0.0
    measurement_age_s: float = 0.0
    is_new_measurement: bool = True
    reason: str = ""
    ground_truth_corruption: list = field(default_factory=list)
    # --- Drone (avcı) telemetrisi — step()'e geçilirse doldurulur ---
    drone_pos_m: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))
    drone_vel_mps: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))
    drone_speed_mps: float = 0.0
    drone_yaw_deg: float = 0.0
    dist_horizontal_m: float = 0.0
    dist_3d_m: float = 0.0
    alt_error_m: float = 0.0   # hedef_z - drone_z: pozitif = drone alçakta


# =============================================================================
# ANA SINIF: GNSS GUVENILIRLIK ANALIZORU
# =============================================================================
class GnssReliabilityAnalyzer:
    def __init__(self, config: GnssReliabilityConfig = None):
        self.cfg = config or GnssReliabilityConfig()
        self.ekf = ConstantVelocityEKF(self.cfg)
        self.cusum = TwoSidedCusum(self.cfg.cusum_k, self.cfg.cusum_h)
        self._last_raw_pos = None
        self._last_accepted_raw_pos = None
        self._last_seen_raw_pos = None
        self._last_measurement_t = None
        self._last_returned_status = GnssStatus.SUSPECT
        self._last_t = None
        self._suspect_streak = 0
        self._warmup_count = 0

    def step(self, raw_pos_native: tuple, t: float = None,
             drone_pos_native: tuple = None) -> GnssAnalysisResult:
        cfg = self.cfg
        t = time.time() if t is None else t

        # --- BIRIM DONUSUMU: SDK'dan gelen native (cm) -> metre ---
        # Tum EKF/CUSUM/esik mantigi METRE uzerinden calisir; ham deger (orijinal
        # birimde) loglamada referans icin saklanir.
        s = cfg.input_unit_scale
        raw_pos = (raw_pos_native[0] * s, raw_pos_native[1] * s, raw_pos_native[2] * s)

        # --- dt hesapla ---
        if self._last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self._last_t)
        self._last_t = t

        # --- EKF ILK GUVENILIR OLCUMDE INITIALIZE EDILIR ---
        # ONEMLI BUG DUZELTMESI: Sim baglandiktan sonraki ILK telemetri karesi
        # genelde henuz doldurulmamis (0,0,0) gelebiliyor. Eger EKF bu (0,0,0)
        # ile initialize edilirse, ilk GERCEK ölçüm (örn. -10926,-238104,...)
        # geldiginde innovation devasa cikar (binlerce sigma), CUSUM bir kerede
        # patlar ve sistem SONSUZA DEK UNRELIABLE'da kilitli kalir (gercek
        # logda gozlemlendi). Çözüm: (0,0,0)'a çok yakin VE hâlâ "warm-up"
        # asamasindaysak (ilk WARMUP_MAX_SAMPLES örnek içinde), bu ölçümü
        # gecerli ilk ölçüm SAYMA; bir sonraki gercek ölçümle yeniden dene.
        if not self.ekf.initialized:
            is_likely_empty_frame = (
                abs(raw_pos[0]) < 1e-6 and abs(raw_pos[1]) < 1e-6 and abs(raw_pos[2]) < 1e-6
            )
            if is_likely_empty_frame and self._warmup_count < cfg.warmup_max_samples:
                self._warmup_count += 1
                # Henuz initialize etmiyoruz; bu kareyi "henuz veri yok" olarak
                # logluyoruz ve bir sonraki karede tekrar deniyoruz.
                result = GnssAnalysisResult(
                    t=t, dt=dt, raw_pos=raw_pos, ekf_pos=(0.0, 0.0, 0.0),
                    ekf_vel=(0.0, 0.0, 0.0), innovation_norm=0.0,
                    cusum_pos=0.0, cusum_neg=0.0, raw_jump_dist=0.0,
                    raw_jump_flag=False, status=GnssStatus.SUSPECT,
                    search_center=(0.0, 0.0, 0.0),
                    search_radius=cfg.search_radius_min,
                    is_new_measurement=False,
                    reason="WARMUP_EMPTY_FRAME",
                    ground_truth_corruption=self._ground_truth_tag(),
                )
                self._log(result)
                return result

            self.ekf.initialize(raw_pos)
            self._last_raw_pos = raw_pos
            self._last_accepted_raw_pos = raw_pos
            self._last_seen_raw_pos = raw_pos
            self._last_measurement_t = t
            self._last_returned_status = GnssStatus.RELIABLE
            result = GnssAnalysisResult(
                t=t, dt=dt, raw_pos=raw_pos, ekf_pos=self.ekf.position(),
                ekf_vel=self.ekf.velocity(), innovation_norm=0.0,
                cusum_pos=0.0, cusum_neg=0.0, raw_jump_dist=0.0,
                raw_jump_flag=False, status=GnssStatus.RELIABLE,
                search_center=self.ekf.position(),
                search_radius=cfg.search_radius_min,
                reason="EKF_INITIALIZED",
                ground_truth_corruption=self._ground_truth_tag(),
            )
            self._log(result)
            return result

        # --- Paket tazeligi: ayni GNSS paketini EKF'ye tekrar enjekte etme ---
        is_new_measurement = True
        measurement_age_s = 0.0
        if self._last_seen_raw_pos is not None:
            same_as_last = _dist3(raw_pos, self._last_seen_raw_pos) < cfg.stale_epsilon_m
            is_new_measurement = not same_as_last
        if not is_new_measurement and self._last_measurement_t is not None:
            measurement_age_s = max(0.0, t - self._last_measurement_t)

        if not is_new_measurement:
            self.ekf.predict(dt)

            if measurement_age_s >= cfg.lost_timeout_s:
                status = GnssStatus.LOST
                reason = "GNSS_LOST_TIMEOUT"
            elif measurement_age_s >= cfg.stale_timeout_s:
                status = GnssStatus.STALE
                reason = "STALE_REPEATED_PACKET"
            else:
                status = self._last_returned_status
                reason = "REPEATED_PACKET_SHORT"

            std = self.ekf.position_std()
            radius = max(cfg.search_radius_min, cfg.search_radius_sigma * max(std))
            result = GnssAnalysisResult(
                t=t, dt=dt, raw_pos=raw_pos, ekf_pos=self.ekf.position(),
                ekf_vel=self.ekf.velocity(), innovation_norm=0.0,
                nis=0.0,
                cusum_pos=self.cusum.g_pos, cusum_neg=self.cusum.g_neg,
                raw_jump_dist=0.0, raw_jump_flag=False, status=status,
                search_center=self.ekf.position(), search_radius=radius,
                measurement_age_s=measurement_age_s,
                is_new_measurement=False,
                reason=reason,
                ground_truth_corruption=self._ground_truth_tag(),
            )
            self._last_returned_status = status
            self._fill_drone_fields(result, drone_pos_native)
            self._log(result)
            return result

        # --- 1) RAW JUMP TESTI (CUSUM'dan bagimsiz, anlik) ---
        raw_jump_dist = 0.0
        raw_jump_flag = False
        if self._last_accepted_raw_pos is not None and dt >= cfg.min_measurement_dt_s:
            raw_jump_dist = _dist3(raw_pos, self._last_accepted_raw_pos)
            max_allowed = cfg.max_plausible_speed_ms * cfg.raw_jump_margin * dt
            raw_jump_flag = raw_jump_dist > max_allowed

        # --- 2) EKF PREDICT (her zaman calisir) ---
        self.ekf.predict(dt)

        # --- 3) Olcumu ON-DEGERLENDIR: once predict-only innovation'a bak ---
        # (update'i cagirmadan once "bu olcum ne kadar sapmis" olcmek icin
        #  S matrisini predict sonrasi P'den hesapliyoruz)
        _, _, nis = self.ekf.innovation_stats(raw_pos)
        innovation_norm = math.sqrt(max(nis, 0.0) / 3.0)

        # --- 4) CUSUM GUNCELLE ---
        g_pos, g_neg = self.cusum.update(innovation_norm)

        # --- 5) DURUM KARARI ---
        # ONEMLI: CUSUM bir kez alarm/severe-alarm esigini astiginda g_pos/g_neg
        # cok buyuk bir degere firlar. Eger resetlenmezse, sonraki yuzlerce
        # normal olcumde bile (s = innovation_norm - k) negatif oldugu icin
        # g_pos'un sifira donmesi cok uzun surer -> sistem sonsuza dek
        # UNRELIABLE'da KILITLI KALIR. Bu yuzden her alarm/severe-alarm karari
        # SONRASINDA CUSUM'u hemen resetliyoruz: "olay tespit edildi, simdi
        # temiz sayfadan yeniden izlemeye basla" mantigi. Boylece bozulma
        # gectiginde sistem makul sayida (suspect_streak_to_unreliable kadar)
        # adim icinde RELIABLE'a geri donebilir.
        if raw_jump_flag:
            # Ani, fiziksel olarak imkansiz sicrama -> aninda UNRELIABLE,
            # CUSUM'un birikmesini beklemeden tepki ver.
            status = GnssStatus.UNRELIABLE
            reason = "RAW_JUMP"
            self._suspect_streak += 1
            self.cusum.reset()
        elif nis >= cfg.nis_unreliable_gate or self.cusum.is_severe_alarm(cfg.cusum_h_unreliable):
            status = GnssStatus.UNRELIABLE
            reason = "NIS_OR_CUSUM_SEVERE"
            self._suspect_streak += 1
            self.cusum.reset()
        elif nis >= cfg.nis_reliable_gate or self.cusum.is_alarm():
            status = GnssStatus.SUSPECT
            reason = "NIS_OR_CUSUM_WARNING"
            self._suspect_streak += 1
            if self._suspect_streak >= cfg.suspect_streak_to_unreliable:
                status = GnssStatus.UNRELIABLE
                reason = "SUSPECT_STREAK_TO_UNRELIABLE"
                self.cusum.reset()
        else:
            status = GnssStatus.RELIABLE
            reason = "NIS_CUSUM_OK"
            self._suspect_streak = 0

        # --- 6) EKF UPDATE (duruma gore adaptif R, ya da hic update yok) ---
        if status == GnssStatus.RELIABLE:
            self.ekf.update(raw_pos, R_scale=1.0)
        elif status == GnssStatus.SUSPECT:
            self.ekf.update(raw_pos, R_scale=cfg.meas_noise_suspect_mult)
        else:  # UNRELIABLE -> olcumu hic enjekte etme, sadece predict ile yetin (kor ucus)
            pass

        # --- 7) ARAMA BOLGESI (search region) ---
        std = self.ekf.position_std()
        radius = cfg.search_radius_sigma * max(std)
        radius = max(radius, cfg.search_radius_min)

        result = GnssAnalysisResult(
            t=t, dt=dt, raw_pos=raw_pos, ekf_pos=self.ekf.position(),
            ekf_vel=self.ekf.velocity(), innovation_norm=innovation_norm,
            nis=nis, cusum_pos=g_pos, cusum_neg=g_neg, raw_jump_dist=raw_jump_dist,
            raw_jump_flag=raw_jump_flag, status=status,
            search_center=self.ekf.position(), search_radius=radius,
            measurement_age_s=measurement_age_s,
            is_new_measurement=is_new_measurement,
            reason=reason,
            ground_truth_corruption=self._ground_truth_tag(),
        )

        self._last_raw_pos = raw_pos
        if is_new_measurement:
            self._last_seen_raw_pos = raw_pos
            self._last_measurement_t = t
        if not raw_jump_flag and status in (GnssStatus.RELIABLE, GnssStatus.SUSPECT):
            self._last_accepted_raw_pos = raw_pos
        self._last_returned_status = status

        self._fill_drone_fields(result, drone_pos_native)
        self._log(result)
        return result

    def _fill_drone_fields(self, result: "GnssAnalysisResult", drone_pos_native):
        """Drone telemetrisini result'a yazar. drone_pos_native None ise no-op."""
        if drone_pos_native is None:
            return
        try:
            _s = self.cfg.input_unit_scale   # cm → m
            dp = (drone_pos_native[0]*_s,
                  drone_pos_native[1]*_s,
                  drone_pos_native[2]*_s)
            tel   = drone.get_telemetry()
            drot  = drone.get_drone_rotation()          # (roll, pitch, yaw) derece
            dvel  = tel["drone"]["velocity"]            # (vx, vy, vz) cm/s
            dspd  = drone.get_drone_speed() * _s        # m/s
            dv_m  = (dvel[0]*_s, dvel[1]*_s, dvel[2]*_s)
            tx, ty, tz = result.search_center
            dx = tx - dp[0]; dy = ty - dp[1]; dz = tz - dp[2]
            result.drone_pos_m       = dp
            result.drone_vel_mps     = dv_m
            result.drone_speed_mps   = dspd
            result.drone_yaw_deg     = drot[2]
            result.dist_horizontal_m = math.sqrt(dx*dx + dy*dy)
            result.dist_3d_m         = math.sqrt(dx*dx + dy*dy + dz*dz)
            result.alt_error_m       = dz   # hedef_z - drone_z: + = drone alçakta
        except Exception:
            pass

    @staticmethod
    def _ground_truth_tag():
        """Sadece bilgi/karsilastirma icin (debug modu acikken). Karar
        mekanizmasi BUNU OKUMAZ -- gercek yariste bu hep bos liste donecektir."""
        try:
            return drone.get_active_corruption()
        except Exception:
            return []

    def _log(self, r: GnssAnalysisResult):
        gt = ";".join(r.ground_truth_corruption) if r.ground_truth_corruption else ""
        logger.info(
            "t=%.2f dt=%.3f raw=(%.1f,%.1f,%.1f) ekf=(%.1f,%.1f,%.1f) "
            "innov_sigma=%.2f nis=%.2f age=%.2fs new=%s reason=%s "
            "cusum=(%.1f,%.1f) jump=%.1fm[%s] -> %s | r_arama=%.1fm%s",
            r.t, r.dt, r.raw_pos[0], r.raw_pos[1], r.raw_pos[2],
            r.ekf_pos[0], r.ekf_pos[1], r.ekf_pos[2],
            r.innovation_norm, r.nis, r.measurement_age_s,
            "EVET" if r.is_new_measurement else "HAYIR", r.reason,
            r.cusum_pos, r.cusum_neg,
            r.raw_jump_dist, "EVET" if r.raw_jump_flag else "-",
            r.status.value, r.search_radius,
            f"  [GT: {gt}]" if gt else "",
        )
        _append_csv_row(r)


def _dist3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# =============================================================================
# CSV LOGLAMA
# =============================================================================
_csv_initialized = False


def _append_csv_row(r: GnssAnalysisResult):
    global _csv_initialized
    mode = "a"
    if not _csv_initialized:
        mode = "w"
    with open(CSV_PATH, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()
        writer.writerow({
            "t": f"{r.t:.3f}", "dt": f"{r.dt:.4f}",
            "raw_x": r.raw_pos[0], "raw_y": r.raw_pos[1], "raw_z": r.raw_pos[2],
            "ekf_x": f"{r.ekf_pos[0]:.3f}", "ekf_y": f"{r.ekf_pos[1]:.3f}", "ekf_z": f"{r.ekf_pos[2]:.3f}",
            "ekf_vx": f"{r.ekf_vel[0]:.3f}", "ekf_vy": f"{r.ekf_vel[1]:.3f}", "ekf_vz": f"{r.ekf_vel[2]:.3f}",
            "innov_norm": f"{r.innovation_norm:.3f}",
            "nis": f"{r.nis:.3f}",
            "cusum_pos": f"{r.cusum_pos:.3f}", "cusum_neg": f"{r.cusum_neg:.3f}",
            "raw_jump_dist": f"{r.raw_jump_dist:.3f}", "raw_jump_flag": int(r.raw_jump_flag),
            "status": r.status.value,
            "search_radius": f"{r.search_radius:.3f}",
            "measurement_age_s": f"{r.measurement_age_s:.3f}",
            "is_new_measurement": int(r.is_new_measurement),
            "reason": r.reason,
            "ground_truth_corruption": ";".join(r.ground_truth_corruption),
            # Drone kolonları
            "drone_x":         f"{r.drone_pos_m[0]:.2f}",
            "drone_y":         f"{r.drone_pos_m[1]:.2f}",
            "drone_z":         f"{r.drone_pos_m[2]:.2f}",
            "drone_vx":        f"{r.drone_vel_mps[0]:.2f}",
            "drone_vy":        f"{r.drone_vel_mps[1]:.2f}",
            "drone_vz":        f"{r.drone_vel_mps[2]:.2f}",
            "drone_speed_mps": f"{r.drone_speed_mps:.2f}",
            "drone_yaw_deg":   f"{r.drone_yaw_deg:.1f}",
            "dist_horizontal_m": f"{r.dist_horizontal_m:.2f}",
            "dist_3d_m":       f"{r.dist_3d_m:.2f}",
            "alt_error_m":     f"{r.alt_error_m:.2f}",
        })
    _csv_initialized = True


# =============================================================================
# STANDALONE TEST DONGUSU
# =============================================================================
def main():
    logger.info("drone_sdk baglantisi deneniyor (127.0.0.1:12345)...")
    if not drone.connect():
        logger.error("Baglanti basarisiz. Unreal sim 'Play' modunda mi?")
        return
    logger.info("Baglandi. (ARM EDILMIYOR - sadece GNSS analizi izleniyor)")
    logger.info("Cikmak icin Ctrl+C.")

    analyzer = GnssReliabilityAnalyzer()

    try:
        while True:
            raw_target_pos = drone.get_target_location()
            analyzer.step(raw_target_pos)
            time.sleep(0.1)  # 10 Hz analiz dongusu (telemetri kendi hizinda gelir)
    except KeyboardInterrupt:
        logger.info("Durduruldu (Ctrl+C). CSV log: %s", CSV_PATH)
    finally:
        drone.disconnect()


if __name__ == "__main__":
    main()
