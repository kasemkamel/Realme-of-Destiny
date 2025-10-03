import numpy as np
import asyncio
import websockets
import json
import logging
from typing import List, Tuple, Optional, Set
from enum import IntEnum
import aiohttp 
from typing import Dict, Any
from systems import MovementSystem, AISystem, CombatSystem, StaggerSystem, BerserkerRageSystem
from unit_definitions import UNIT_TYPE_MAP, UNIT_DATA, STATUS_MAP


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORLD_SERVER_URL = "http://localhost:8000/battle_results"  # عنوان سيرفر العالم

# ===== Constants =====
class UnitState(IntEnum):
    IDLE = 0
    MOVING = 1
    ATTACKING = 2

class Team(IntEnum):
    PLAYER = 0
    ENEMY = 1
    ALLY = 2



# ===== Quadtree Implementation =====
class Quadtree:
    def __init__(self, boundary, capacity=4):
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None
    
    def subdivide(self):
        x, y, w, h = self.boundary
        hw, hh = w / 2, h / 2
        
        self.northwest = Quadtree((x, y, hw, hh), self.capacity)
        self.northeast = Quadtree((x + hw, y, hw, hh), self.capacity)
        self.southwest = Quadtree((x, y + hh, hw, hh), self.capacity)
        self.southeast = Quadtree((x + hw, y + hh, hw, hh), self.capacity)
        self.divided = True
    
    def insert(self, point):
        x, y, idx = point
        bx, by, bw, bh = self.boundary
        
        if not (bx <= x < bx + bw and by <= y < by + bh):
            return False
        
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        
        if not self.divided:
            self.subdivide()
        
        return (self.northwest.insert(point) or self.northeast.insert(point) or
                self.southwest.insert(point) or self.southeast.insert(point))
    
    def query_radius(self, center, radius):
        cx, cy = center
        bx, by, bw, bh = self.boundary
        
        closest_x = max(bx, min(cx, bx + bw))
        closest_y = max(by, min(cy, by + bh))
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        
        if dist_sq > radius ** 2:
            return []
        
        found = []
        for x, y, idx in self.points:
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                found.append((x, y, idx))
        
        if self.divided:
            found.extend(self.northwest.query_radius(center, radius))
            found.extend(self.northeast.query_radius(center, radius))
            found.extend(self.southwest.query_radius(center, radius))
            found.extend(self.southeast.query_radius(center, radius))
        
        return found


# ===== World Class =====
class World:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        self.selected_units: Set[int] = set()
        self.units_reformed: Set[int] = set()
        self.units_need_reform: Set[int] = set()
        
        self.deployment_zone = {
            # منطقة اللاعب على اليسار
            Team.PLAYER: (50, 50, 300, self.height - 100),
            
            # منطقة العدو على اليمين
            Team.ENEMY: (self.width - 350, 50, 300, self.height - 100),
            
            # منطقة الحليف (يمكن تركها كما هي أو تعديلها)
            Team.ALLY: (50, 50, 300, self.height - 100) # مثال: الحليف يبدأ مع اللاعب
        }
        self.battle_started = False
        
        self.team_colors = {
            Team.PLAYER: (220, 50, 50),
            Team.ENEMY: (50, 120, 220),
            Team.ALLY: (50, 220, 120),
        }
        
        # DOD Arrays
        self.positions = np.empty((0, 2), dtype=np.float32)
        self.teams = np.empty(0, dtype=np.int32)
        self.unit_ids = np.empty(0, dtype=np.int32)
        self.target_positions = np.empty((0, 2), dtype=np.float32)
        self.healths = np.empty(0, dtype=np.float32)
        self.attack_ranges = np.empty(0, dtype=np.float32)
        self.attack_damages = np.empty(0, dtype=np.float32)
        self.attack_cooldowns = np.empty(0, dtype=np.float32)
        self.states = np.empty(0, dtype=np.int32)
        # احتمال الصد/التفادي لكل جندي (قابل للتخصيص لكل جندي)
        self.dodge_chances = np.empty(0, dtype=np.float32)

        # (اختياري) عدّادات تسجيل عدد المرات التي صد فيها كل جندي
        self.evade_counts = np.empty(0, dtype=np.int32)

        # timers to track how long each soldier has been in its current state
        self.state_timers = np.empty(0, dtype=np.float32)
        self.prev_states = np.empty(0, dtype=np.int32)

        # central quadtree (rebuilt once per frame)
        self.quadtree: Optional[Quadtree] = None
        
        self.soldier_count = 0
        self.next_unit_id = 0
        self.battle_id: Optional[str] = None
        self.player_units_map: Dict[int, str] = {}  # {unit_id: player_id}
        self.unit_type_map: Dict[int, str] = {}  # {unit_id: unit_type_id}
        
        self.obstacles = [
            (200, 150, 100, 50),
            (500, 300, 80, 80),
            (350, 450, 120, 40)
        ]
        
        # Systems
        self.combat_system = CombatSystem(self)
        self.ai_system = AISystem(self)
    
    def _validate_spawn_position(self, team_id: int, positions: np.ndarray) -> bool:
        """التحقق من أن جميع مواقع الجنود داخل منطقة النشر"""
        if self.battle_started or team_id not in self.deployment_zone:
            return True
        
        zone = self.deployment_zone[team_id]
        zone_x, zone_y, zone_w, zone_h = zone
        
        in_zone = (
            (positions[:, 0] >= zone_x) & (positions[:, 0] <= zone_x + zone_w) &
            (positions[:, 1] >= zone_y) & (positions[:, 1] <= zone_y + zone_h)
        )
        
        return np.all(in_zone)
    
    def spawn_unit(self, team_id: int, num_soldiers: int, start_position: Tuple[float, float], unit_type_id: str, unit_template: dict, player_id: Optional[str] = None) -> int:
        """
        إنشاء وحدة جديدة باستخدام خصائص من القالب المخزن
        
        Args:
            unit_template: قاموس يحتوي على {health, attack_damage, attack_range, dodge_chance}
        """
        unit_id = self.next_unit_id
        self.next_unit_id += 1
        self.unit_type_map[unit_id] = unit_type_id
        if player_id:
            self.player_units_map[unit_id] = player_id
        # إنشاء تشكيل شبكي
        cols = int(np.sqrt(num_soldiers))
        rows = (num_soldiers + cols - 1) // cols
        
        new_positions = []
        for i in range(num_soldiers):
            row = i // cols
            col = i % cols
            x = start_position[0] + col * FORMATION_SPACING
            y = start_position[1] + row * FORMATION_SPACING
            new_positions.append([x, y])
        
        new_positions = np.array(new_positions, dtype=np.float32)
        
        # التحقق من المنطقة
        if not self._validate_spawn_position(team_id, new_positions):
            logger.warning(f"Cannot spawn unit {unit_id} outside deployment zone")
            return -1
        
        new_teams = np.full(num_soldiers, team_id, dtype=np.int32)
        new_unit_ids = np.full(num_soldiers, unit_id, dtype=np.int32)
        new_targets = new_positions.copy()
        new_healths = np.full(num_soldiers, unit_template.get('health', DEFAULT_HEALTH), dtype=np.float32)
        new_attack_ranges = np.full(num_soldiers, unit_template.get('attack_range', DEFAULT_ATTACK_RANGE), dtype=np.float32)
        new_attack_damages = np.full(num_soldiers, unit_template.get('attack_damage', DEFAULT_ATTACK_DAMAGE), dtype=np.float32)
        new_dodge_chances = np.full(num_soldiers, unit_template.get('dodge_chance', DEFAULT_DODGE_CHANCE), dtype=np.float32)
        new_attack_cooldowns = np.zeros(num_soldiers, dtype=np.float32)
        new_states = np.full(num_soldiers, UnitState.IDLE, dtype=np.int32)
        new_timers = np.zeros(num_soldiers, dtype=np.float32)
        new_evade_counts = np.zeros(num_soldiers, dtype=np.int32)
        new_prev_states = new_states.copy()
        
        if self.soldier_count > 0:
            self.positions = np.vstack([self.positions, new_positions])
            self.teams = np.concatenate([self.teams, new_teams])
            self.unit_ids = np.concatenate([self.unit_ids, new_unit_ids])
            self.target_positions = np.vstack([self.target_positions, new_targets])
            self.healths = np.concatenate([self.healths, new_healths])
            self.attack_ranges = np.concatenate([self.attack_ranges, new_attack_ranges])
            self.attack_damages = np.concatenate([self.attack_damages, new_attack_damages])
            self.attack_cooldowns = np.concatenate([self.attack_cooldowns, new_attack_cooldowns])
            self.states = np.concatenate([self.states, new_states])
            self.state_timers = np.concatenate([self.state_timers, new_timers])
            self.prev_states = np.concatenate([self.prev_states, new_prev_states])
            self.dodge_chances = np.concatenate([self.dodge_chances, new_dodge_chances])
            self.evade_counts = np.concatenate([self.evade_counts, new_evade_counts])
        else:
            self.positions = new_positions
            self.teams = new_teams
            self.unit_ids = new_unit_ids
            self.target_positions = new_targets
            self.healths = new_healths
            self.attack_ranges = new_attack_ranges
            self.attack_damages = new_attack_damages
            self.attack_cooldowns = new_attack_cooldowns
            self.states = new_states
            self.state_timers = new_timers
            self.prev_states = new_prev_states
            self.dodge_chances = new_dodge_chances
            self.evade_counts = new_evade_counts

        
        self.soldier_count += num_soldiers
        logger.info(f"Spawned unit {unit_id} with {num_soldiers} soldiers (Team {team_id})")
        return unit_id
    
    def move_unit(self, unit_id: int, target: Tuple[float, float]):
        self.units_reformed.discard(unit_id)
        self.units_need_reform.discard(unit_id)
        
        mask = self.unit_ids == unit_id
        if not np.any(mask):
            logger.warning(f"Unit {unit_id} not found")
            return

        # تحديد فريق الوحدة (افتراض أن كل الجنود في الوحدة لديهم نفس الفريق)
        team_id = int(self.teams[mask][0])

        # أثناء وضع النشر: قفل الهدف داخل منطقة النشر الخاصة بالفريق
        if not self.battle_started and team_id in self.deployment_zone:
            target = self._clamp_point_to_zone(team_id, target)

        unit_positions = self.positions[mask]
        centroid = np.mean(unit_positions, axis=0)
        offsets = unit_positions - centroid

        new_targets = np.array(target, dtype=np.float32) + offsets

        # =====================================================
        # سلوك أثناء النشر: نُحَرِّك المواضع فعلاً (teleport on click)
        # سلوك أثناء القتال: نُحدِّث target_positions وتبدأ الحركة تدريجياً
        # =====================================================
        if not self.battle_started:
            # نقل فوري لمواضع الجنود ضمن الوحدة
            self.positions[mask] = new_targets
            # نحافظ أيضاً على target_positions متطابقة لتجنب قفز لاحق
            self.target_positions[mask] = new_targets
            # إبقاء الحالة IDLE أثناء النشر
            self.states[mask] = UnitState.IDLE
            # إعادة ضبط مؤقتات الحالة
            self.state_timers[mask] = 0.0
        else:
            # المعركة بدأت => سلوك الحركة التدريجية كما كان
            self.target_positions[mask] = new_targets
            self.states[mask] = UnitState.MOVING
            self.state_timers[mask] = 0.0

    def form_unit(self, unit_id: int, start: Tuple[float, float], end: Tuple[float, float]):
        self.units_reformed.discard(unit_id)
        self.units_need_reform.discard(unit_id)
        
        mask = self.unit_ids == unit_id
        if not np.any(mask):
            return
        

        team_id = int(self.teams[mask][0])

        # أثناء وضع النشر: تأكد أن نقاط البداية والنهاية داخل منطقة النشر
        if not self.battle_started and team_id in self.deployment_zone:
            start = self._clamp_point_to_zone(team_id, start)
            end = self._clamp_point_to_zone(team_id, end)

        num_soldiers = np.sum(mask)
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        front_length = np.sqrt(dx * dx + dy * dy)
        
        if front_length < 1:
            return
        
        front_dir = np.array([dx / front_length, dy / front_length], dtype=np.float32)
        side_dir = np.array([-front_dir[1], front_dir[0]], dtype=np.float32)
        
        soldiers_per_row = max(1, int(front_length / FORMATION_SPACING))
        num_rows = (num_soldiers + soldiers_per_row - 1) // soldiers_per_row
        
        new_targets = []
        soldier_idx = 0
        for row in range(num_rows):
            soldiers_in_row = min(soldiers_per_row, num_soldiers - soldier_idx)
            for col in range(soldiers_in_row):
                t = col / max(1, soldiers_in_row - 1) if soldiers_in_row > 1 else 0.5
                pos_on_line = np.array(start, dtype=np.float32) + front_dir * (t * front_length)
                pos_with_depth = pos_on_line + side_dir * (row * FORMATION_SPACING)
                new_targets.append(pos_with_depth)
                soldier_idx += 1
        
        self.target_positions[mask] = np.array(new_targets, dtype=np.float32)

        if not self.battle_started:
            # أثناء النشر: نضع المَواضِع مباشرة لتظهر التشكيلة فوراً
            self.positions[mask] = np.array(new_targets, dtype=np.float32)
            self.states[mask] = UnitState.IDLE
        else:
            # أثناء القتال: نعين التشكيل كهدف وتتحرك الجنود تدريجياً
            self.states[mask] = UnitState.MOVING

        self.state_timers[mask] = 0.0

    def select_units(self, unit_ids: List[int]):
        for uid in unit_ids:
            mask = self.unit_ids == uid
            if np.any(mask) and np.any(self.teams[mask] == Team.PLAYER):
                self.selected_units.add(uid)
    
    def deselect_all(self):
        self.selected_units.clear()
    
    def move_selected_units(self, target: Tuple[float, float]):
        if not self.selected_units:
            return

        # جمع كل المواقع الموجودة لوحدات المختارة (تجاهل الوحدات الفارغة)
        all_positions = []
        for unit_id in list(self.selected_units):
            mask = self.unit_ids == unit_id
            if np.any(mask):
                unit_pos = self.positions[mask]
                all_positions.extend(unit_pos)
            else:
                logger.warning(f"Selected unit {unit_id} has no soldiers; skipping.")

        if not all_positions:
            # لا توجد مواقع صالحة - لا نتحرك
            return

        centroid = np.mean(all_positions, axis=0)
        offset = np.array(target, dtype=np.float32) - centroid

        # الآن نحسب الهدف لكل وحدة فردياً ونطبق القفل (clamp) حسب فريق الوحدة إن كنا في وضع النشر
        for unit_id in list(self.selected_units):
            mask = self.unit_ids == unit_id
            if not np.any(mask):
                # تخطى الوحدات التي لم يعد لها جنود (تمت إزالتها مثلاً)
                continue

            unit_centroid = np.mean(self.positions[mask], axis=0)
            new_target = unit_centroid + offset  # معرفة الهدف الآن مضمونة لأن الماسك غير فارغ

            # قفل الهدف داخل منطقة النشر الخاصة بالفريق إذا لم تبدأ المعركة
            team_id = int(self.teams[mask][0])
            if not self.battle_started and team_id in self.deployment_zone:
                new_target = self._clamp_point_to_zone(team_id, tuple(new_target))

            # استخدم move_unit لعمل النقل (ستقوم هي بتحديث state_timers... الخ)
            self.move_unit(unit_id, tuple(new_target))
    
    def start_battle(self):
        """
        تفعيل بدء المعركة: نضع العلم ونحوّل أي أهداف مخططة أثناء النشر إلى حركة فعلية.
        """
        self.battle_started = True
        logger.info("Battle started!")

        if self.soldier_count == 0:
            return

        # كل جندي لديه هدف مختلف عن موقعه سيصبح MOVING
        # نحسب الفروق بين target_positions و positions
        diffs = np.linalg.norm(self.target_positions - self.positions, axis=1)
        moving_mask = diffs > 1e-3  # عتبة صغيرة

        if np.any(moving_mask):
            self.states[moving_mask] = UnitState.MOVING
            self.state_timers[moving_mask] = 0.0

    def apply_separation_force(self, movements: np.ndarray) -> np.ndarray:
        if self.soldier_count == 0:
            return movements
        
        qt = self.quadtree
        if qt is None:
            return movements
        
        separation_forces = np.zeros_like(self.positions)
        
        for i in range(self.soldier_count):
            pos = self.positions[i]
            nearby = qt.query_radius((pos[0], pos[1]), SEPARATION_RADIUS)
            
            if len(nearby) > 1:
                for x, y, idx in nearby:
                    if idx == i:
                        continue
                    
                    diff = pos - np.array([x, y], dtype=np.float32)
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0.1:
                        force = (diff / dist) * (SEPARATION_RADIUS - dist)
                        separation_forces[i] += force
        
        norms = np.linalg.norm(separation_forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 0.001)
        
        separation_forces = np.where(
            norms > SEPARATION_FORCE,
            separation_forces * (SEPARATION_FORCE / norms),
            separation_forces
        )
        
        return movements + separation_forces
    
    def remove_dead_soldiers(self):
        if self.soldier_count == 0:
            return
        
        alive_mask = self.healths > 0
        num_dead = np.sum(~alive_mask)
        
        if num_dead == 0:
            return
        
        logger.info(f"Removing {num_dead} dead soldiers")
        
        self.positions = self.positions[alive_mask]
        self.teams = self.teams[alive_mask]
        self.unit_ids = self.unit_ids[alive_mask]
        self.target_positions = self.target_positions[alive_mask]
        self.healths = self.healths[alive_mask]
        self.attack_ranges = self.attack_ranges[alive_mask]
        self.attack_damages = self.attack_damages[alive_mask]
        self.attack_cooldowns = self.attack_cooldowns[alive_mask]
        self.states = self.states[alive_mask]
        self.state_timers = self.state_timers[alive_mask]
        self.prev_states = self.prev_states[alive_mask]
        self.dodge_chances = self.dodge_chances[alive_mask]
        self.evade_counts = self.evade_counts[alive_mask]
        
        self.soldier_count = len(self.positions)
    
    def reform_formation(self, unit_id: int):
        mask = self.unit_ids == unit_id
        if not np.any(mask):
            return
        
        unit_positions = self.positions[mask]
        centroid = np.mean(unit_positions, axis=0)
        
        num_soldiers = len(unit_positions)
        cols = int(np.sqrt(num_soldiers))
        rows = (num_soldiers + cols - 1) // cols
        
        total_width = cols * FORMATION_SPACING
        total_height = rows * FORMATION_SPACING
        
        start_x = max(FORMATION_SPACING, min(self.width - total_width - FORMATION_SPACING, 
                                              centroid[0] - total_width / 2))
        start_y = max(FORMATION_SPACING, min(self.height - total_height - FORMATION_SPACING, 
                                              centroid[1] - total_height / 2))
        
        new_positions = []
        for idx in range(num_soldiers):
            row = idx // cols
            col = idx % cols
            x = start_x + col * FORMATION_SPACING
            y = start_y + row * FORMATION_SPACING
            new_positions.append([x, y])
        
        self.target_positions[mask] = np.array(new_positions, dtype=np.float32)
        self.states[mask] = UnitState.MOVING
        self.state_timers[mask] = 0.0
        self.units_reformed.add(unit_id)
        self.units_need_reform.discard(unit_id)
    
    def check_battle_end(self) -> Optional[any]:
        """
        تحديد نتيجة المعركة على مستوى التحالفات:
        - ترجع "allies" إذا بقي أي جندي من (PLAYER أو ALLY) ولم يبقَ من ENEMY
        - ترجع "enemies" إذا بقي أي جندي من ENEMY ولم يبقَ من (PLAYER أو ALLY)
        - ترجع -1 إذا لا أحد بقي (draw)
        - ترجع None إذا المعركة ما زالت جارية (كلا الجانبين لا زالا فيه ناجين)
        """
        if not self.battle_started:
            return None

        # قناع الجنود الأحياء
        alive_mask = self.healths > 0

        # إذا لا أحد على قيد الحياة -> تعادل
        if not np.any(alive_mask):
            logger.info("Battle ended in a draw (no survivors)")
            return -1

        # تحالف الحلفاء = Player + Ally
        allies_mask = np.isin(self.teams, [Team.PLAYER, Team.ALLY]) & alive_mask
        enemies_mask = (self.teams == Team.ENEMY) & alive_mask

        allies_alive = np.any(allies_mask)
        enemies_alive = np.any(enemies_mask)

        if allies_alive and not enemies_alive:
            logger.info("Battle ended! Winner: Allies (Player + Ally)")
            return "allies"

        if enemies_alive and not allies_alive:
            logger.info("Battle ended! Winner: Enemies")
            return "enemies"

        # إذا كلا الجانبين لهما ناجين -> المعركة مستمرة
        return None

    def reset_for_new_battle(self):
        """إعادة تعيين حالة العالم لمعركة جديدة مع الحفاظ على الإعدادات الأساسية"""
        self.selected_units.clear()
        self.units_reformed.clear()
        self.units_need_reform.clear()
        self.battle_started = False
        self.battle_id = None
        self.player_units_map.clear()
        self.unit_type_map.clear()
        
        # إعادة تعيين المصفوفات
        self.positions = np.empty((0, 2), dtype=np.float32)
        self.teams = np.empty(0, dtype=np.int32)
        self.unit_ids = np.empty(0, dtype=np.int32)
        self.target_positions = np.empty((0, 2), dtype=np.float32)
        self.healths = np.empty(0, dtype=np.float32)
        self.attack_ranges = np.empty(0, dtype=np.float32)
        self.attack_damages = np.empty(0, dtype=np.float32)
        self.attack_cooldowns = np.empty(0, dtype=np.float32)
        self.states = np.empty(0, dtype=np.int32)
        self.dodge_chances = np.empty(0, dtype=np.float32)
        self.evade_counts = np.empty(0, dtype=np.int32)
        self.state_timers = np.empty(0, dtype=np.float32)
        self.prev_states = np.empty(0, dtype=np.int32)
        self.quadtree = None
        self.soldier_count = 0
        self.next_unit_id = 0
        
        logger.info("World reset for new battle")

    def _clamp_point_to_zone(self, team_id: int, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        قفل نقطة داخل منطقة نشر الفريق (أو داخل حدود الخريطة إذا المعركة بدأت أو المنطقة غير موجودة).
        """
        x, y = float(point[0]), float(point[1])

        # لو المعركة بدأت أو المنطقة غير محددة -> نقيّد للنطاق العام للخرائط
        if self.battle_started or team_id not in self.deployment_zone:
            x = min(max(x, 0.0), float(self.width))
            y = min(max(y, 0.0), float(self.height))
            return (x, y)

        zx, zy, zw, zh = self.deployment_zone[team_id]
        x = min(max(x, zx), zx + zw)
        y = min(max(y, zy), zy + zh)
        return (x, y)

    def generate_battle_report(self) -> dict:
        """إنشاء تقرير مفصل عن نتيجة المعركة (بما في ذلك ملخّص التحالفات)"""
        report = {
            "winner": None,
            "teams": {},
            "alliances": {},
            "surviving_units": []
        }

        # قناع الجنود الأحياء
        alive_mask = self.healths > 0

        # تحديد الفائز على مستوى التحالفات (allies = PLAYER+ALLY vs enemies = ENEMY)
        allies_alive = np.any(alive_mask & np.isin(self.teams, [Team.PLAYER, Team.ALLY]))
        enemies_alive = np.any(alive_mask & (self.teams == Team.ENEMY))

        if not np.any(alive_mask):
            report["winner"] = -1  # تعادل: لا أحد حي
        elif allies_alive and not enemies_alive:
            report["winner"] = "allies"
        elif enemies_alive and not allies_alive:
            report["winner"] = "enemies"
        else:
            report["winner"] = None  # المعركة ما زالت جارية

        # إحصائيات لكل فريق (كما في النسخة الأصلية)
        for team in Team:
            mask_team = self.teams == team
            mask_alive = alive_mask & mask_team

            units = {}
            # إذا لم يكن هناك أي عنصر في mask_team, np.unique سيُرجع مصفوفة فارغة — آمن
            for uid in np.unique(self.unit_ids[mask_team]):
                mask_unit = (self.unit_ids == uid) & mask_alive
                num_soldiers = int(np.sum(mask_unit))
                avg_health = float(np.mean(self.healths[mask_unit])) if num_soldiers > 0 else 0.0
                units[int(uid)] = {
                    "soldiers_remaining": num_soldiers,
                    "average_health": round(avg_health, 2)
                }

            report["teams"][int(team)] = {
                "total_units": int(len(units)),
                "total_soldiers": int(np.sum([u["soldiers_remaining"] for u in units.values()])),
                "units": units
            }

        # ملخّص التحالفات (allies vs enemies)
        # تحالف الحلفاء = PLAYER + ALLY
        allies_mask = alive_mask & np.isin(self.teams, [Team.PLAYER, Team.ALLY])
        enemies_mask = alive_mask & (self.teams == Team.ENEMY)

        def alliance_stats(mask):
            total_soldiers = int(np.sum(mask))
            # حساب عدد الوحدات (قائمة الـ unit_ids التي تحتوي أي جندي حي ضمن الماسك)
            if total_soldiers == 0:
                return {"total_units": 0, "total_soldiers": 0, "average_health": 0.0}
            alive_unit_ids = np.unique(self.unit_ids[mask])
            avg_health = float(np.mean(self.healths[mask])) if total_soldiers > 0 else 0.0
            return {
                "total_units": int(len(alive_unit_ids)),
                "total_soldiers": total_soldiers,
                "average_health": round(avg_health, 2)
            }

        report["alliances"]["allies"] = alliance_stats(allies_mask)
        report["alliances"]["enemies"] = alliance_stats(enemies_mask)

        
        
        alive_mask = self.healths > 0
        for uid in np.unique(self.unit_ids[alive_mask]):
            mask_unit = (self.unit_ids == uid) & alive_mask
            num_soldiers = int(np.sum(mask_unit))
            if num_soldiers > 0:
                avg_health = float(np.mean(self.healths[mask_unit]))
                report["surviving_units"].append({
                    "unit_id": int(uid),
                    "owner_player_id": self.player_units_map.get(uid),
                    "unit_type_id": self.unit_type_map.get(uid),
                    "soldiers_remaining": num_soldiers,
                    "average_health": round(avg_health, 2)
                })
        
        report["battle_id"] = self.battle_id
        
        return report

    def update(self, delta_time: float):
        # إضافة مصفوفات جديدة لأنواع الوحدات
        self.unit_types = np.empty(0, dtype=np.int32)  # نوع الوحدة من UNIT_TYPE_MAP
        self.speeds = np.empty(0, dtype=np.float32)  # سرعة الحركة
        self.defenses = np.empty(0, dtype=np.float32)  # الدفاع
        self.accuracies = np.empty(0, dtype=np.float32)  # الدقة
        self.crit_chances = np.empty(0, dtype=np.float32)  # فرصة الضربة القاضية
        self.stagger_modifiers = np.empty(0, dtype=np.float32)  # معدل التسبب بالسقوط
        self.stagger_resistances = np.empty(0, dtype=np.float32)  # مقاومة السقوط
        self.stagger_timers = np.empty(0, dtype=np.float32)  # مؤقتات السقوط
        
        MovementSystem.update(self.world, delta_time)
        StaggerSystem.update(self.world, delta_time)
        BerserkerRageSystem.update(self.world, delta_time)
        CombatSystem.update(self.world, delta_time)
        # HealingSystem.update(self.world, delta_time)
        # DeathSystem.update(self.world, delta_time)
        # TODO: Add more systems later (Abilities, AI, Death, Sync)
        if self.soldier_count == 0:
            return

        # Build central Quadtree once per frame (alive soldiers only)
        self.quadtree = Quadtree((0, 0, self.width, self.height))
        for i in range(self.soldier_count):
            if self.healths[i] > 0:
                self.quadtree.insert((self.positions[i, 0], self.positions[i, 1], i))

        if self.battle_started:
            moving_mask = (self.states == UnitState.MOVING) | (self.states == UnitState.ATTACKING)

            if np.any(moving_mask):
                diff = self.target_positions[moving_mask] - self.positions[moving_mask]
                distances = np.linalg.norm(diff, axis=1, keepdims=True)
                distances = np.maximum(distances, 0.001)

                max_move = MOVEMENT_SPEED * delta_time
                move_amount = np.minimum(distances, max_move)
                movements = np.zeros_like(self.positions)
                movements[moving_mask] = (diff / distances) * move_amount

                # تطبيق قوة تجنب التداخل (تستخدم Quadtree المركزي)
                movements = self.apply_separation_force(movements)

                # تطبيق الحركة
                self.positions += movements

                # التحقق من الوصول للهدف (فقط للـ MOVING)
                just_moving_mask = self.states == UnitState.MOVING
                if np.any(just_moving_mask):
                    diff_check = self.target_positions[just_moving_mask] - self.positions[just_moving_mask]
                    dist_check = np.linalg.norm(diff_check, axis=1)
                    arrived = dist_check < ARRIVAL_THRESHOLD

                    global_indices = np.where(just_moving_mask)[0]
                    arrived_indices = global_indices[arrived]
                    self.states[arrived_indices] = UnitState.IDLE
                    # reset timers for arrived soldiers
                    self.state_timers[arrived_indices] = 0.0
        else:
            # أثناء وضع النشر لا ننفّذ الحركة الفعلية - الجنود يبقون في مواقعهم لكن لديهم target_positions محدثة
            pass

        # الحفاظ على الجنود داخل الحدود
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height)
        
        if self.battle_started:
            # AI System - باستخدام المنطق الجديد للوحدات
            self.ai_system.update(delta_time)
            
            # Combat System
            self.combat_system.update(delta_time)

        # Update state timers (reset when state changed, increment otherwise)
        if self.soldier_count > 0:
            same_state_mask = (self.states == self.prev_states)
            # increment timers where same state, else reset to 0
            self.state_timers = np.where(same_state_mask, self.state_timers + delta_time, 0.0)
            self.prev_states = self.states.copy()

        # Reform idle units with additional checks (enemy proximity and idle timer)
        for unit_id in list(self.units_need_reform):
            mask = self.unit_ids == unit_id
            if not np.any(mask):
                self.units_need_reform.discard(unit_id)
                continue

            # centroid of the unit
            unit_positions = self.positions[mask]
            centroid = np.mean(unit_positions, axis=0)

            # check for nearby enemy soldiers using central quadtree
            nearby = self.quadtree.query_radius((centroid[0], centroid[1]), AI_SEARCH_RADIUS / 2)
            enemy_nearby = False
            for x, y, idx in nearby:
                if self.teams[idx] != self.teams[np.where(mask)[0][0]] and self.healths[idx] > 0:
                    enemy_nearby = True
                    break

            if enemy_nearby:
                # skip reform while enemies are nearby
                continue

            # ensure all soldiers in unit have been IDLE for at least REFORM_IDLE_TIME
            if not np.all(self.states[mask] == UnitState.IDLE):
                continue

            if not np.all(self.state_timers[mask] >= REFORM_IDLE_TIME):
                continue

            # pass all checks -> reform
            self.reform_formation(unit_id)

        # إزالة الجنود الأموات
        self.remove_dead_soldiers()

    def get_state(self) -> dict:
        state ={
            'positions': self.positions.tolist(),
            'teams': self.teams.tolist(),
            'unit_ids': self.unit_ids.tolist(),
            'healths': self.healths.tolist(),
            'states': self.states.tolist(),
            'selected_units': list(self.selected_units),
            'battle_started': self.battle_started,
            'obstacles': self.obstacles
        }
        if not self.battle_started:
            state['deployment_zones'] = self.deployment_zone
        else:
            state['deployment_zones'] = {}  # بعد بدء المعركة نخليها فاضية

        return state

# ===== Command Validator =====
class CommandValidator:
    REQUIRED_FIELDS = {
        'move': ['unit_id', 'target'],
        'form': ['unit_id', 'start', 'end'],
        'select': ['unit_ids'],
        'move_selected': ['target'],
        'spawn': ['team_id', 'num_soldiers', 'position'],
        'register_unit_types': ['data'],
        'setup_battle': ['battle_id', 'players']
    }
    
    @staticmethod
    def validate(cmd: dict) -> Tuple[bool, str]:
        cmd_type = cmd.get('type')
        
        if not cmd_type:
            return False, "Missing command type"
        
        if cmd_type not in CommandValidator.REQUIRED_FIELDS:
            if cmd_type in ['deselect', 'start_battle']:
                return True, ""
            return False, f"Unknown command type: {cmd_type}"
        
        required = CommandValidator.REQUIRED_FIELDS[cmd_type]
        for field in required:
            if field not in cmd:
                return False, f"Missing required field: {field}"
        
        return True, ""


# ===== Server =====
class GameServer:
    def __init__(self):
        self.world = World()
        self.clients = set()
        self.validator = CommandValidator()

        # Command queue: process commands exactly once per tick at the top of the game loop
        self.command_queue: asyncio.Queue = asyncio.Queue()
        
        self.unit_templates: Dict[str, Dict[str, Any]] = {}
        
        # حالة السيرفر
        self.battle_active = False
        self.battle_setup_complete = False
        
        logger.info("Battle Server initialized - Ready for unit type registration")
    
    async def handle_client(self, websocket):
        self.clients.add(websocket)
        logger.info(f"New client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                # Put raw command into the server queue -- processed at tick boundary
                try:
                    cmd = json.loads(message)
                except Exception:
                    logger.warning("Received invalid JSON command from client")
                    continue
                await self.command_queue.put(cmd)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def send_results_to_world_server(self, report: dict):
        """إرسال نتائج المعركة إلى سيرفر العالم"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(WORLD_SERVER_URL, json=report) as response:
                    if response.status == 200:
                        logger.info(f"Battle results sent successfully for battle_id: {report.get('battle_id')}")
                    else:
                        logger.error(f"Failed to send results: {response.status}")
        except Exception as e:
            logger.error(f"Error sending results to world server: {e}")

    def validate_unit_template(self, template: dict) -> bool:
        """التحقق من صحة قالب الوحدة"""
        required_keys = ['health', 'attack_damage', 'attack_range', 'dodge_chance']
        return all(key in template for key in required_keys)

    async def process_command(self, cmd: dict):
        valid, error = self.validator.validate(cmd)
        if not valid:
            logger.warning(f"Invalid command: {error}")
            return
        
        cmd_type = cmd['type']
        
        # الأوامر الجديدة
        if cmd_type == 'register_unit_types':
            data = cmd['data']
            for unit_type_id, template in data.items():
                if self.validate_unit_template(template):
                    self.unit_templates[unit_type_id] = template
                    logger.info(f"Registered unit type: {unit_type_id}")
                else:
                    logger.warning(f"Invalid template for unit type: {unit_type_id}")
            return
        
        elif cmd_type == 'setup_battle':
            if self.battle_active:
                logger.warning("Cannot setup battle while another is active")
                return
            
            battle_id = cmd['battle_id']
            players = cmd['players']
            
            self.world.battle_id = battle_id
            self.battle_setup_complete = False
            
            for player in players:
                player_id = player['player_id']
                team_id = player['team_id']
                
                for unit_data in player['units']:
                    unit_type_id = unit_data['unit_type_id']
                    
                    if unit_type_id not in self.unit_templates:
                        logger.error(f"Unknown unit type: {unit_type_id}")
                        continue
                    
                    template = self.unit_templates[unit_type_id]
                    self.world.spawn_unit(
                        team_id=team_id,
                        num_soldiers=unit_data['num_soldiers'],
                        start_position=tuple(unit_data['initial_position']),
                        unit_type_id=unit_type_id,
                        unit_template=template,
                        player_id=player_id
                    )
            
            self.battle_setup_complete = True
            logger.info(f"Battle {battle_id} setup complete - Waiting for start command")
            return
        
        # الأوامر الموجودة
        if cmd_type == 'move':
            self.world.move_unit(cmd['unit_id'], tuple(cmd['target']))
        
        elif cmd_type == 'form':
            self.world.form_unit(cmd['unit_id'], tuple(cmd['start']), tuple(cmd['end']))
        
        elif cmd_type == 'select':
            self.world.select_units(cmd['unit_ids'])
        
        elif cmd_type == 'deselect':
            self.world.deselect_all()
        
        elif cmd_type == 'move_selected':
            self.world.move_selected_units(tuple(cmd['target']))
        
        elif cmd_type == 'spawn':
            # للتوافق مع الأوامر القديمة (للاختبار فقط)
            unit_type_id = cmd.get('unit_type_id', 'default')
            template = self.unit_templates.get(unit_type_id, {
                'health': DEFAULT_HEALTH,
                'attack_damage': DEFAULT_ATTACK_DAMAGE,
                'attack_range': DEFAULT_ATTACK_RANGE,
                'dodge_chance': DEFAULT_DODGE_CHANCE
            })
            self.world.spawn_unit(
                cmd['team_id'], 
                cmd['num_soldiers'], 
                tuple(cmd['position']),
                unit_type_id=unit_type_id,
                unit_template=template
            )
        
        elif cmd_type == 'start_battle':
            if not self.battle_setup_complete:
                logger.warning("Cannot start battle - setup not complete")
                return
            self.world.start_battle()
            self.battle_active = True
    
    async def broadcast_state(self, extra: dict = None):
        if self.clients:
            state = self.world.get_state()
            if extra:  # لو في بيانات إضافية زي تقرير المعركة
                state["battle_report"] = extra

            message = json.dumps(state)
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    async def game_loop(self):
        target_fps = 20
        delta_time = 1.0 / target_fps
        
        while True:
            # معالجة الأوامر
            while True:
                try:
                    cmd = self.command_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    await self.process_command(cmd)
            
            # تحديث العالم فقط إذا كانت المعركة نشطة
            if self.battle_active:
                self.world.update(delta_time)
                
                winner = self.world.check_battle_end()
                if winner is not None:
                    report = self.world.generate_battle_report()
                    
                    # إرسال النتائج للعملاء
                    await self.broadcast_state(extra=report)
                    
                    # إرسال النتائج لسيرفر العالم
                    await self.send_results_to_world_server(report)
                    
                    logger.info("Battle Report:\n" + json.dumps(report, indent=4, ensure_ascii=False))
                    
                    # إعادة التعيين للمعركة التالية
                    self.world.reset_for_new_battle()
                    self.battle_active = False
                    self.battle_setup_complete = False
                    
                    logger.info("Server ready for new battle")
            
            # بث الحالة دائماً
            await self.broadcast_state()
            await asyncio.sleep(delta_time)
    
    async def start(self):
        async with websockets.serve(self.handle_client, "localhost", 8765):
            logger.info("Server running on ws://localhost:8765")
            await self.game_loop()


# ===== Main Entry Point =====
if __name__ == "__main__":
    server = GameServer()
    asyncio.run(server.start())
