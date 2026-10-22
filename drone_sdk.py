"""
================================================================================
DRONE OF WAR - COMPETITION SDK (v2.2)
================================================================================
Bu dosya, Unreal Engine tabanlı Drone Of War oyunu ile Python arasındaki 
TCP iletişimini sağlayan resmi SDK dosyasıdır. 

YARIŞMACI NOTU:
Sadece bu dosyayı projenize import ederek dronunuzu kontrol edebilirsiniz.
Kendi AI mantığınızı, bu SDK'daki 'get_' fonksiyonları ile telemetri alıp,
'set_' fonksiyonları ile dronu yönlendirerek kurmalısınız.

Platform sabitleri (oyun içine gömülü, değiştirilemez):
    * Kamera: 25° yukarı tilt, 125° FOV
    * Hedef Talon İHA: kanat açıklığı 1718 mm, gövde uzunluğu 1100 mm
    * Hedef GPS telemetrisi: en fazla 1 Hz (saniyede 1 güncelleme)

Örnek Kullanım:
    import drone_sdk as drone
    drone.connect()
    drone.set_arm(True)
    pos = drone.get_drone_location()
================================================================================
"""

import socket
import threading
import time
import logging

# Log seviyesini sadece hataları gösterecek şekilde ayarlıyoruz.
logging.basicConfig(level=logging.ERROR, format='[DroneSDK] %(levelname)s: %(message)s')

# =============================================================================
#  (DEBUG) HEDEF GPS BOZMA EFEKT BAYRAKLARI
# -----------------------------------------------------------------------------
#  Oyun tarafindaki ETalonCorruptionFlag (TalonGPSSpoofComponent.h) ile AYNI
#  bit degerleri. Debug modunda gelen "maske" bu bayraklarla cozulur.
#  (Sira, GUI'de alt alta listeleme sirasini da belirler.)
# =============================================================================
FLAG_NOISE      = 1 << 0
FLAG_SPEEDNOISE = 1 << 1
FLAG_OFFSET     = 1 << 2
FLAG_JUMP       = 1 << 3
FLAG_DROPOUT    = 1 << 4
FLAG_RATELIMIT  = 1 << 5
FLAG_DELAY      = 1 << 6

CORRUPTION_FLAGS = [
    (FLAG_NOISE,      "Konum gurultusu"),
    (FLAG_SPEEDNOISE, "Hiz gurultusu"),
    (FLAG_OFFSET,     "Sabit offset (kayma)"),
    (FLAG_JUMP,       "Ani ziplama (spike)"),
    (FLAG_DROPOUT,    "Veri kesintisi (dropout)"),
    (FLAG_RATELIMIT,  "Guncelleme hizi siniri (max 1 Hz)"),
    (FLAG_DELAY,      "Gecikmeli veri"),
]

def decode_corruption_mask(mask):
    """Bit maskesini, o an aktif olan efekt adlari listesine cevirir."""
    try:
        m = int(mask)
    except (TypeError, ValueError):
        return []
    return [name for bit, name in CORRUPTION_FLAGS if m & bit]

def build_corruption_lines(mask, params):
    """Aktif efektleri DEGER + SURE (ne kadar, ne kadar sure) bilgisiyle birlikte
    satir satir aciklar. params yoksa sadece efekt adlarini dondurur."""
    try:
        m = int(mask)
    except (TypeError, ValueError):
        return []
    if not params:
        return decode_corruption_mask(m)

    p = params
    lines = []
    if m & FLAG_NOISE:
        lines.append(f"Konum gurultusu: +-{p['pos_noise_m']:.1f} m sapma")
    if m & FLAG_SPEEDNOISE:
        lines.append(
            f"Hiz gurultusu: +-{p['spd_noise_ms']:.1f} m/s + %{p['spd_noise_pct']:.0f}")
    if m & FLAG_OFFSET:
        ox, oy, oz = p["offset_m"]
        lines.append(
            f"Sabit offset: ({ox:.1f}, {oy:.1f}, {oz:.1f}) m kayma  |  "
            f"{p['offset_active_s']:.0f} sn'dir aktif")
    if m & FLAG_JUMP:
        lines.append(
            f"Ani ziplama: {p['jump_mag_m']:.0f} m sicrama  |  "
            f"kalan {p['jump_remain_s']:.1f} sn")
    if m & FLAG_DROPOUT:
        lines.append(
            f"Veri kesintisi: {p['dropout_dur_s']:.0f} sn donma  |  "
            f"kalan {p['dropout_remain_s']:.1f} sn")
    if m & FLAG_RATELIMIT:
        hz = p["rate_hz"]
        lines.append(f"Guncelleme hizi siniri: {hz:.2f} Hz (saniyede ~{hz:.1f} guncelleme)")
    if m & FLAG_DELAY:
        lines.append(f"Gecikmeli veri: {p['delay_s']:.1f} sn once-ki konum")
    return lines

class _DroneInternal:
    """
    Dahili iletişim sınıfı. Yarışmacıların bu sınıfı direkt kullanması gerekmez.
    Public API fonksiyonları bu sınıfın global örneği (_drone) üzerinden çalışır.
    """
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 12345
        self.sock = None
        self.is_connected = False
        self.lock = threading.Lock()
        
        # Anlık Kontrol Değerleri
        self.throttle = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.arm = False
        
        # Telemetri Veri Yapısı
        self.telemetry = {
            "drone": {
                "position": (0.0, 0.0, 0.0), # X, Y, Z
                "rotation": (0.0, 0.0, 0.0), # Roll, Pitch, Yaw
                "velocity": (0.0, 0.0, 0.0), # X, Y, Z hız vektörü
                "speed": 0.0,                # cm/s cinsinden toplam hız
                "altitude": 0.0              # Z eksenindeki yükseklik
            },
            "target": {
                "position": (0.0, 0.0, 0.0), # X, Y, Z
                "rotation": (0.0, 0.0, 0.0), # Roll, Pitch, Yaw
                "speed": 0.0                 # cm/s cinsinden hız
            }
        }

        # (Dahili/Debug) Gercek (bozulmamis) degerler. Sadece oyun tarafinda debug
        # secenegi acikken gelir; normal yarismada bu alanlar guncellenmez.
        # "corruption_mask": o an HEDEF veride aktif olan bozma efektlerinin bit maskesi.
        # "corruption_active": maskeden cozulen, su an aktif efekt adlari listesi.
        self.telemetry_truth = {
            "available": False,
            "drone": {"position": (0.0, 0.0, 0.0), "altitude": 0.0, "speed": 0.0},
            "target": {"position": (0.0, 0.0, 0.0), "speed": 0.0},
            "corruption_mask": 0,
            "corruption_active": [],
            "corruption_params": {}
        }
        
        self._stop_event = threading.Event()
        self._receive_thread = None

    def connect(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.is_connected = True
            self._stop_event.clear()
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()
            return True
        except Exception as e:
            return False

    def disconnect(self):
        self._stop_event.set()
        if self.sock: self.sock.close()
        self.is_connected = False

    def send_inputs(self):
        if not self.is_connected: return
        arm_val = 1 if self.arm else 0
        # Format: throttle,pitch,roll,yaw,arm\n
        msg = f"{self.throttle:.4f},{self.pitch:.4f},{self.roll:.4f},{self.yaw:.4f},{arm_val}\n"
        try:
            self.sock.sendall(msg.encode('utf-8'))
        except:
            self.is_connected = False

    def _receive_loop(self):
        buffer = ""
        while not self._stop_event.is_set():
            try:
                data = self.sock.recv(2048).decode('utf-8')
                if not data: break  # Karşı taraf bağlantıyı kapattı
                buffer += data
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line: self._parse_telemetry(line)
            except socket.timeout:
                # Hedef GPS telemetrisi 1 Hz'e kadar yavaş olabilir veya geçici olarak kesilebilir (dropout).
                # Bu durumda bağlantıyı KOPARMA; en son bilinen telemetri korunur, beklemeye devam et.
                continue
            except Exception:
                break
        self.is_connected = False

    def _parse_telemetry(self, line):
        try:
            v = [float(x) for x in line.split(",")]
            # Format: Drone(0-10), Target(11-17)
            if len(v) >= 18:
                with self.lock:
                    self.telemetry["drone"]["position"] = (v[0], v[1], v[2])
                    self.telemetry["drone"]["rotation"] = (v[3], v[4], v[5])
                    self.telemetry["drone"]["velocity"] = (v[6], v[7], v[8])
                    self.telemetry["drone"]["speed"] = v[9]
                    self.telemetry["drone"]["altitude"] = v[10]
                    self.telemetry["target"]["position"] = (v[11], v[12], v[13])
                    self.telemetry["target"]["rotation"] = (v[14], v[15], v[16])
                    self.telemetry["target"]["speed"] = v[17]
                    # (Debug) Gercek degerler eklenmisse (index 18..26) onlari da oku.
                    if len(v) >= 27:
                        self.telemetry_truth["available"] = True
                        self.telemetry_truth["drone"]["position"] = (v[18], v[19], v[20])
                        self.telemetry_truth["drone"]["altitude"] = v[21]
                        self.telemetry_truth["drone"]["speed"] = v[22]
                        self.telemetry_truth["target"]["position"] = (v[23], v[24], v[25])
                        self.telemetry_truth["target"]["speed"] = v[26]
                        # (Debug) Index 27: aktif bozma maskesi. Index 28..40: efekt
                        # degerleri + canli sureler (ne kadar, ne kadar sure).
                        if len(v) >= 28:
                            mask = int(v[27])
                            self.telemetry_truth["corruption_mask"] = mask
                            params = {}
                            if len(v) >= 41:
                                params = {
                                    "pos_noise_m": v[28],
                                    "spd_noise_ms": v[29],
                                    "spd_noise_pct": v[30],
                                    "offset_m": (v[31], v[32], v[33]),
                                    "offset_active_s": v[34],
                                    "jump_mag_m": v[35],
                                    "jump_remain_s": v[36],
                                    "dropout_dur_s": v[37],
                                    "dropout_remain_s": v[38],
                                    "rate_hz": v[39],
                                    "delay_s": v[40],
                                }
                            self.telemetry_truth["corruption_params"] = params
                            self.telemetry_truth["corruption_active"] = build_corruption_lines(mask, params)
                        else:
                            self.telemetry_truth["corruption_mask"] = 0
                            self.telemetry_truth["corruption_active"] = []
                            self.telemetry_truth["corruption_params"] = {}
        except: pass

# Global SDK Örneği
_drone = _DroneInternal()

# --- PUBLIC API (YARIŞMACI FONKSİYONLARI) ---

def connect(host='127.0.0.1', port=12345):
    """Oyuna bağlantı kurar. (Varsayılan: localhost:12345)"""
    return _drone.connect(host, port)

def disconnect():
    """Bağlantıyı güvenli bir şekilde kapatır."""
    _drone.disconnect()

def is_connected():
    """Bağlantı durumunu döner (True/False)."""
    return _drone.is_connected

# --- KONTROL FONKSİYONLARI ---

def set_throttle(value: float):
    """Dronun DİKEY komutunu ayarlar (ANGLE MODE). Aralığı: [-1.0 - 1.0]
    -1.0 = yerçekimi telafisi kapanır, dron FİZİKLE serbest DÜŞER, 0.0 = irtifasını
    korur (HOVER), +1.0 = maksimum hızla TIRMANIR. Maksimum tırmanma hızı sabit
    120 km/h ile sınırlıdır (oyun içine gömülüdür, değiştirilemez)."""
    _drone.throttle = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_pitch(value: float):
    """Dronun ileri/geri eğilme değerini ayarlar. Aralığı: [-1.0 - 1.0]"""
    _drone.pitch = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_roll(value: float):
    """Dronun sağa/sola yatış değerini ayarlar. Aralığı: [-1.0 - 1.0]"""
    _drone.roll = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_yaw(value: float):
    """Dronun kendi ekseninde dönme değerini ayarlar. Aralığı: [-1.0 - 1.0]"""
    _drone.yaw = max(-1.0, min(1.0, value))
    _drone.send_inputs()

def set_arm(state: bool):
    """Dronu aktif (True) veya pasif (False) hale getirir."""
    _drone.arm = state
    _drone.send_inputs()

def set_control_surfaces(throttle: float, pitch: float, roll: float, yaw: float, arm: bool):
    """Tüm kontrol yüzeylerini tek TCP satırında gönderir (ara karelerde throttle/pitch uyumsuzluğunu önler).
    throttle [-1..1]: -1 alçal, 0 hover, +1 tırman (dikey hız komutu)."""
    _drone.throttle = max(-1.0, min(1.0, throttle))
    _drone.pitch = max(-1.0, min(1.0, pitch))
    _drone.roll = max(-1.0, min(1.0, roll))
    _drone.yaw = max(-1.0, min(1.0, yaw))
    _drone.arm = arm
    _drone.send_inputs()

# --- TELEMETRİ FONKSİYONLARI ---

def get_drone_location():
    """Dronun anlık konumunu döner: (X, Y, Z)"""
    with _drone.lock: return _drone.telemetry["drone"]["position"]

def get_drone_rotation():
    """Dronun anlık rotasyonunu döner: (Roll, Pitch, Yaw)"""
    with _drone.lock: return _drone.telemetry["drone"]["rotation"]

def get_drone_speed():
    """Dronun anlık toplam hızını (cm/s) döner."""
    with _drone.lock: return _drone.telemetry["drone"]["speed"]

def get_drone_altitude():
    """Dronun anlık irtifasını (Z ekseni) döner."""
    with _drone.lock: return _drone.telemetry["drone"]["altitude"]

def get_target_location():
    """Hedefin anlık konumunu döner: (X, Y, Z)"""
    with _drone.lock: return _drone.telemetry["target"]["position"]

def get_target_rotation():
    """Hedefin anlık rotasyonunu döner: (Roll, Pitch, Yaw)"""
    with _drone.lock: return _drone.telemetry["target"]["rotation"]

def get_target_speed():
    """Hedefin anlık hızını (cm/s) döner."""
    with _drone.lock: return _drone.telemetry["target"]["speed"]

def get_telemetry():
    """Tüm telemetri verilerini bir 'dictionary' olarak döner."""
    with _drone.lock: return _drone.telemetry.copy()

def get_debug_truth():
    """(Dahili/Debug) Oyun tarafında debug seçeneği açıkken gelen GERÇEK (bozulmamış)
    değerleri döner. 'available' False ise bu veri gelmiyor demektir.
    Normal yarışmada kullanılmaz; sadece test/karşılaştırma içindir."""
    import copy
    with _drone.lock: return copy.deepcopy(_drone.telemetry_truth)

def get_active_corruption():
    """(Dahili/Debug) HEDEF veride O AN aktif olan bozma efektlerinin adlarini
    bir liste olarak döner (örn. ['Veri kesintisi (dropout)', 'Gecikmeli veri']).
    Debug seçeneği kapalıysa boş liste döner. Sadece test/izleme içindir."""
    with _drone.lock: return list(_drone.telemetry_truth.get("corruption_active", []))

# --- ÖNEMLİ NOT (GERÇEKÇİ SENSÖR DAVRANIŞI) ------------------------------------
# Telemetri verisi gerçek bir GPS/sensör gibi davranabilir:
#   * Hedef (Talon) konum/hız telemetrisi en fazla 1 Hz güncellenir.
#   * Kendi dron telemetrisi tam hızda ve temiz gelir.
#   * Konum/hız değerleri bir miktar gürültü, sabit kayma veya ani sıçrama içerebilir.
#   * Veri kısa süreliğine kesilebilir; bu sırada 'get_' fonksiyonları SON bilinen
#     değeri döndürmeye devam eder (bağlantı kopmaz).
#   * Gelen konum, anlık değil birkaç saniye gecikmeli olabilir.
# AI mantığınızı bu belirsizliklere dayanıklı (filtreleme/yumuşatma ile) kurun.
# ------------------------------------------------------------------------------
