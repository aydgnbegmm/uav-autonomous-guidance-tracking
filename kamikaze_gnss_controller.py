"""
================================================================================
DRONE KAMIKAZE GNSS CONTROLLER (STABLE APPROACH)
================================================================================
- İrtifayı tam 45m'de sabit tutar.
- Hedefe yaklaştıkça lineer olarak hızlanarak (pitch artırarak) saldırıya hazırlanır.
- Görsel kilitlenme (Visual Track) aşamasına mükemmel stabiliteyle girilmesini sağlar.
================================================================================
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

class KamikazeGNSSController:
    
    def __init__(self):
        # YAW (Sağ-Sol) Dönüşü: KD çok düşük, savrulma yapmaz.
        self.pid_yaw = PIDGains(kp=0.015, ki=0.001, kd=0.002)
        
        # THROTTLE PID'si: 45m'yi korumak için saf PD (sabit hedeflerde I birikmesi yo-yo yaptırır)
        self.pid_throttle = PIDGains(kp=0.010, ki=0.000, kd=0.005)
        
        self.yaw_integral = 0.0
        self.throttle_integral = 0.0
        
        self.prev_yaw_error = 0.0
        self.prev_throttle_error = 0.0
        
        self._prev_yaw_cmd = 0.0
        self._prev_pitch_cmd = 0.0
        self.target_z = None
        
        self.distance_to_target = float('inf')
        self._last_pid_call_t = None
        
    def _normalize_angle(self, angle: float) -> float:
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
        return angle
    
    def _pid_update(self, error: float, integral: float, prev_error: float, 
                    gains: PIDGains, dt: float) -> Tuple[float, float, float]:
        integral += error * dt
        integral = max(-1.0, min(1.0, integral))
        derivative = (error - prev_error) / dt if dt > 0 else 0.0
        output = (gains.kp * error + gains.ki * integral + gains.kd * derivative)
        return output, integral, error

    def _smooth(self, prev, current, alpha=0.2):
        return prev + (current - prev) * alpha
    
    def calculate_control_inputs(self, telemetry: Dict) -> Dict:
        d_pos = telemetry["drone"]["position"]
        d_rot = telemetry["drone"]["rotation"]
        
        tar_pos = telemetry["target"]["position"]
        
        now = time.time()
        dt = (now - self._last_pid_call_t) if self._last_pid_call_t else 0.02
        dt = max(0.005, min(0.1, dt))
        self._last_pid_call_t = now
        
        # 1. YATAY (XY) YAKLAŞIM
        dx = tar_pos[0] - d_pos[0]
        dy = tar_pos[1] - d_pos[1]
        dist_2d = math.sqrt(dx**2 + dy**2)
        
        # 2. İRTİFA (Z) KORUMASI: V3'teki "Yukarıdan Kartal Dalışı" Mantığı
        # Kamera 25 derece aşağı baktığı ve drone ileri gitmek için burnunu eğdiği için,
        # hedefi görebilmemiz için kesinlikle hedefin ÜSTÜNDE uçmalıyız.
        # Hedefin 35 metre üstünü (3500 cm) hedefliyoruz.
        self.target_z = tar_pos[2] + 3500.0
            
        dz = self.target_z - d_pos[2]
        self.distance_to_target = math.sqrt(dist_2d**2 + (tar_pos[2]-d_pos[2])**2)
        
        # 3. YAW KONTROLÜ (Hedefe Yönelme)
        target_yaw = math.degrees(math.atan2(dy, dx))
        yaw_error = self._normalize_angle(target_yaw - d_rot[2])
        yaw_output, self.yaw_integral, self.prev_yaw_error = self._pid_update(
            yaw_error, self.yaw_integral, self.prev_yaw_error, self.pid_yaw, dt
        )
        
        raw_yaw = max(-1.0, min(1.0, yaw_output))
        final_yaw = self._smooth(self._prev_yaw_cmd, raw_yaw, alpha=0.3)
        self._prev_yaw_cmd = final_yaw
        
        # 4. THROTTLE KONTROLÜ (İrtifayı Tutma)
        alt_error = dz
        thr_output, self.throttle_integral, self.prev_throttle_error = self._pid_update(
            alt_error, self.throttle_integral, self.prev_throttle_error, self.pid_throttle, dt
        )
        final_thr = max(-1.0, min(1.0, thr_output))
        
        # 5. PITCH KONTROLÜ (Acımasızca Hızlanma)
        # Hedef çok hızlı kaçıyor olabilir. 1 dakika içinde yakalamak için
        # başlar başlamaz %90 (0.90) pitch, 3000m'nin altına inince tam gaz (%100)
        if dist_2d > 3000:
            target_pitch = 0.90
        elif dist_2d > 800:
            target_pitch = 1.00 # Hedefi arayı kapatmak için en yüksek hız
        else:
            target_pitch = 0.70 # Görsel takibe geçmeden hemen önce stabilizasyon için yavaşla
            
        final_pitch = self._smooth(self._prev_pitch_cmd, target_pitch, alpha=0.15)
        self._prev_pitch_cmd = final_pitch
        
        return {
            "throttle": final_thr,
            "pitch": final_pitch,
            "roll": 0.0,
            "yaw": final_yaw,
            "arm": True,
            "phase": 1,
            "distance": self.distance_to_target,
            "status": f"GLIDING: alt={d_pos[2]/100:.1f}m target_z={self.target_z/100:.1f}m pitch={final_pitch:.2f}"
        }