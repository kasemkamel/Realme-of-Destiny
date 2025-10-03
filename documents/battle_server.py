"""
Real-Time Strategy Game Server
================================
Main server implementation using Entity-Component-System architecture.
Handles game state management, system orchestration, and client communication.
"""

import numpy as np
import asyncio
import websockets
import json
import logging
from typing import List, Tuple, Optional, Set, Dict, Any
from enum import IntEnum
from dataclasses import dataclass
import aiohttp
from unit_definitions import UNIT_TYPE_MAP, UNIT_DATA, STATUS_MAP # Import systems from systems.py
from systems import ( 
    MovementSystem, 
    CollisionSystem,
    StaggerSystem,
    CombatSystem,
    AISystem,
    BerserkerRageSystem,
    HealthDeathSystem
    # Add other ability systems as they're implemented
)

# Import unit definitions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== Constants =====
WORLD_SERVER_URL = "http://localhost:8000/battle_results"

# Game physics constants
TICK_RATE = 20  # Game updates per second
MOVEMENT_SPEED = 50.0
FORMATION_SPACING = 15.0
ARRIVAL_THRESHOLD = 2.0
SEPARATION_RADIUS = 12.0
SEPARATION_FORCE = 25.0
AI_SEARCH_RADIUS = 150.0
REFORM_IDLE_TIME = 3.0

# Default unit stats (fallback values)
DEFAULT_HEALTH = 100.0
DEFAULT_ATTACK_DAMAGE = 10.0
DEFAULT_ATTACK_RANGE = 50.0
DEFAULT_DODGE_CHANCE = 0.1




class UnitState(IntEnum):
    """Enumeration of possible unit states"""
    IDLE = 0
    MOVING = 1
    ATTACKING = 2
    STAGGERED = 3


class Team(IntEnum):
    """Team/faction identifiers"""
    PLAYER = 0
    ENEMY = 1
    ALLY = 2


@dataclass
class UnitTemplate:
    """Template for unit base statistics"""
    unit_type_id: str
    max_health: int
    attack_damage: int
    attack_range: float
    defense: int
    speed: float
    accuracy: float
    block_chance: float
    crit_chance: float
    attack_speed: float
    vision_range: int
    stagger_duration: float
    cost: int
    instant_kill_chance: float
    ammo_amount: int
    weight: float
    # Add other stats as needed


# ===== Spatial Partitioning: Quadtree =====
class Quadtree:
    """
    Spatial partitioning data structure for efficient proximity queries.
    Used for collision detection and neighbor searches.
    """
    
    def __init__(self, boundary: Tuple[float, float, float, float], capacity: int = 4):
        """
        Args:
            boundary: (x, y, width, height) defining the spatial region
            capacity: Maximum points before subdivision
        """
        self.boundary = boundary
        self.capacity = capacity
        self.points: List[Tuple[float, float, int]] = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None
    
    def subdivide(self):
        """Split this node into four quadrants"""
        x, y, w, h = self.boundary
        hw, hh = w / 2, h / 2
        
        self.northwest = Quadtree((x, y, hw, hh), self.capacity)
        self.northeast = Quadtree((x + hw, y, hw, hh), self.capacity)
        self.southwest = Quadtree((x, y + hh, hw, hh), self.capacity)
        self.southeast = Quadtree((x + hw, y + hh, hw, hh), self.capacity)
        self.divided = True
    
    def insert(self, point: Tuple[float, float, int]) -> bool:
        """
        Insert a point (x, y, entity_index) into the quadtree.
        
        Returns:
            True if insertion successful, False otherwise
        """
        x, y, idx = point
        bx, by, bw, bh = self.boundary
        
        # Check if point is within boundary
        if not (bx <= x < bx + bw and by <= y < by + bh):
            return False
        
        # If space available, add here
        if len(self.points) < self.capacity:
            self.points.append(point)
            return True
        
        # Otherwise subdivide and insert into child
        if not self.divided:
            self.subdivide()
        
        return (self.northwest.insert(point) or self.northeast.insert(point) or
                self.southwest.insert(point) or self.southeast.insert(point))
    
    def query_radius(self, center: Tuple[float, float], radius: float) -> List[Tuple[float, float, int]]:
        """
        Find all points within a given radius of center.
        
        Returns:
            List of (x, y, entity_index) tuples within radius
        """
        cx, cy = center
        bx, by, bw, bh = self.boundary
        
        # Quick rejection: check if circle intersects boundary
        closest_x = max(bx, min(cx, bx + bw))
        closest_y = max(by, min(cy, by + bh))
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        
        if dist_sq > radius ** 2:
            return []
        
        # Check points in this node
        found = []
        for x, y, idx in self.points:
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                found.append((x, y, idx))
        
        # Recursively check children
        if self.divided:
            found.extend(self.northwest.query_radius(center, radius))
            found.extend(self.northeast.query_radius(center, radius))
            found.extend(self.southwest.query_radius(center, radius))
            found.extend(self.southeast.query_radius(center, radius))
        
        return found


# ===== Game State Manager =====
class GameState:
    """
    Central game state using Data-Oriented Design principles.
    All entity data stored in parallel NumPy arrays for performance.
    """
    
    def __init__(self, world_width: int = 800, world_height: int = 600):
        self.width = world_width
        self.height = world_height
        
        # Core entity arrays (parallel structure)
        self.positions = np.empty((0, 2), dtype=np.float32)
        self.velocities = np.empty((0, 2), dtype=np.float32)
        self.teams = np.empty(0, dtype=np.int32)
        self.unit_ids = np.empty(0, dtype=np.int32)
        self.unit_types = np.empty(0, dtype=np.int32)
        
        # Target/navigation data
        self.target_positions = np.empty((0, 2), dtype=np.float32)
        self.target_entity_ids = np.empty(0, dtype=np.int32)
        
        # Combat stats
        self.healths = np.empty(0, dtype=np.float32)
        self.max_healths = np.empty(0, dtype=np.float32)
        self.attack_damages = np.empty(0, dtype=np.float32)
        self.attack_ranges = np.empty(0, dtype=np.float32)
        self.attack_speeds = np.empty(0, dtype=np.float32)
        self.attack_cooldowns = np.empty(0, dtype=np.float32)
        self.defenses = np.empty(0, dtype=np.float32)
        self.accuracies = np.empty(0, dtype=np.float32)
        self.vision_ranges = np.empty(0, dtype=np.float32)
        self.stagger_durations = np.empty(0, dtype=np.float32)
        self.instant_kill_chances = np.empty(0, dtype=np.float32)
        self.ammo_amounts = np.empty(0, dtype=np.int32)
        self.current_ammo = np.empty(0, dtype=np.int32)
        self.weights = np.empty(0, dtype=np.float32) 
        
        # Movement stats
        self.speeds = np.empty(0, dtype=np.float32)
        
        # Special abilities
        self.dodge_chances = np.empty(0, dtype=np.float32)
        self.crit_chances = np.empty(0, dtype=np.float32)
        self.stagger_modifiers = np.empty(0, dtype=np.float32)
        self.stagger_resistances = np.empty(0, dtype=np.float32)
        self.stagger_timers = np.empty(0, dtype=np.float32)
        
        # State tracking
        self.states = np.empty(0, dtype=np.int32)
        self.state_timers = np.empty(0, dtype=np.float32)
        self.prev_states = np.empty(0, dtype=np.int32)
        
        # Statistics
        self.evade_counts = np.empty(0, dtype=np.int32)
        self.kill_counts = np.empty(0, dtype=np.int32)
        
        # Spatial indexing (rebuilt each frame)
        self.quadtree: Optional[Quadtree] = None
        
        # Counters
        self.entity_count = 0
        self.next_unit_id = 0
        
    def rebuild_quadtree(self):
        """Rebuild spatial index for efficient proximity queries"""
        self.quadtree = Quadtree((0, 0, self.width, self.height))
        
        # Only insert living entities
        alive_mask = self.healths > 0
        for i in np.where(alive_mask)[0]:
            self.quadtree.insert((
                self.positions[i, 0],
                self.positions[i, 1],
                i
            ))
    
    def add_entities(self, count: int, **initial_data) -> np.ndarray:
        """
        Add new entities to the game state.
        
        Args:
            count: Number of entities to add
            **initial_data: Dictionary of array_name -> values
            
        Returns:
            Array of indices for the newly added entities
        """
        start_idx = self.entity_count
        new_indices = np.arange(start_idx, start_idx + count)
        
        # Resize all arrays
        new_size = self.entity_count + count
        
        # Helper to extend array
        def extend_array(arr, new_values):
            if arr.size == 0:
                return new_values
            return np.concatenate([arr, new_values])
        
        # Positions and velocities
        self.positions = extend_array(self.positions, initial_data.get('positions', np.zeros((count, 2), dtype=np.float32)))
        self.velocities = extend_array(self.velocities, np.zeros((count, 2), dtype=np.float32))
        
        # IDs and types
        self.teams = extend_array(self.teams, initial_data.get('teams', np.zeros(count, dtype=np.int32)))
        self.unit_ids = extend_array(self.unit_ids, initial_data.get('unit_ids', np.zeros(count, dtype=np.int32)))
        self.unit_types = extend_array(self.unit_types, initial_data.get('unit_types', np.zeros(count, dtype=np.int32)))
        
        # Targets
        self.target_positions = extend_array(self.target_positions, initial_data.get('target_positions', self.positions[-count:].copy()))
        self.target_entity_ids = extend_array(self.target_entity_ids, np.full(count, -1, dtype=np.int32))
        
        # Combat stats
        self.healths = extend_array(self.healths, initial_data.get('healths', np.full(count, DEFAULT_HEALTH, dtype=np.float32)))
        self.max_healths = extend_array(self.max_healths, initial_data.get('max_healths', np.full(count, DEFAULT_HEALTH, dtype=np.float32)))
        self.attack_damages = extend_array(self.attack_damages, initial_data.get('attack_damages', np.full(count, DEFAULT_ATTACK_DAMAGE, dtype=np.float32)))
        self.attack_ranges = extend_array(self.attack_ranges, initial_data.get('attack_ranges', np.full(count, DEFAULT_ATTACK_RANGE, dtype=np.float32)))
        self.attack_speeds = extend_array(self.attack_speeds, initial_data.get('attack_speeds', np.ones(count, dtype=np.float32)))
        self.attack_cooldowns = extend_array(self.attack_cooldowns, np.zeros(count, dtype=np.float32))
        self.defenses = extend_array(self.defenses, initial_data.get('defenses', np.zeros(count, dtype=np.float32)))
        self.accuracies = extend_array(self.accuracies, initial_data.get('accuracies', np.ones(count, dtype=np.float32)))
        
        # Movement
        self.speeds = extend_array(self.speeds, initial_data.get('speeds', np.full(count, MOVEMENT_SPEED, dtype=np.float32)))
        
        # Special abilities
        self.dodge_chances = extend_array(self.dodge_chances, initial_data.get('dodge_chances', np.full(count, DEFAULT_DODGE_CHANCE, dtype=np.float32)))
        self.crit_chances = extend_array(self.crit_chances, initial_data.get('crit_chances', np.zeros(count, dtype=np.float32)))
        self.stagger_modifiers = extend_array(self.stagger_modifiers, initial_data.get('stagger_modifiers', np.zeros(count, dtype=np.float32)))
        self.stagger_resistances = extend_array(self.stagger_resistances, initial_data.get('stagger_resistances', np.zeros(count, dtype=np.float32)))
        self.stagger_timers = extend_array(self.stagger_timers, np.zeros(count, dtype=np.float32))
        
        # States
        self.states = extend_array(self.states, initial_data.get('states', np.full(count, UnitState.IDLE, dtype=np.int32)))
        self.state_timers = extend_array(self.state_timers, np.zeros(count, dtype=np.float32))
        self.prev_states = extend_array(self.prev_states, self.states[-count:].copy())
        
        # Statistics
        self.evade_counts = extend_array(self.evade_counts, np.zeros(count, dtype=np.int32))
        self.kill_counts = extend_array(self.kill_counts, np.zeros(count, dtype=np.int32))
        
        self.entity_count = new_size
        return new_indices
    
    def remove_dead_entities(self):
        """Remove entities with health <= 0"""
        if self.entity_count == 0:
            return
        
        alive_mask = self.healths > 0
        num_dead = np.sum(~alive_mask)
        
        if num_dead == 0:
            return
        
        logger.info(f"Removing {num_dead} dead entities")
        
        # Filter all arrays
        self.positions = self.positions[alive_mask]
        self.velocities = self.velocities[alive_mask]
        self.teams = self.teams[alive_mask]
        self.unit_ids = self.unit_ids[alive_mask]
        self.unit_types = self.unit_types[alive_mask]
        self.target_positions = self.target_positions[alive_mask]
        self.target_entity_ids = self.target_entity_ids[alive_mask]
        self.healths = self.healths[alive_mask]
        self.max_healths = self.max_healths[alive_mask]
        self.attack_damages = self.attack_damages[alive_mask]
        self.attack_ranges = self.attack_ranges[alive_mask]
        self.attack_speeds = self.attack_speeds[alive_mask]
        self.attack_cooldowns = self.attack_cooldowns[alive_mask]
        self.defenses = self.defenses[alive_mask]
        self.accuracies = self.accuracies[alive_mask]
        self.speeds = self.speeds[alive_mask]
        self.dodge_chances = self.dodge_chances[alive_mask]
        self.crit_chances = self.crit_chances[alive_mask]
        self.stagger_modifiers = self.stagger_modifiers[alive_mask]
        self.stagger_resistances = self.stagger_resistances[alive_mask]
        self.stagger_timers = self.stagger_timers[alive_mask]
        self.states = self.states[alive_mask]
        self.state_timers = self.state_timers[alive_mask]
        self.prev_states = self.prev_states[alive_mask]
        self.evade_counts = self.evade_counts[alive_mask]
        self.kill_counts = self.kill_counts[alive_mask]
        
        self.entity_count = len(self.positions)


# ===== World Manager =====
class World:
    """
    High-level game world manager.
    Orchestrates systems and manages game state.
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        
        # Game state
        self.state = GameState(width, height)
        
        # Unit templates (loaded from units.npy or registered dynamically)
        self.unit_templates: Dict[str, UnitTemplate] = {}
        
        # Battle management
        self.battle_started = False
        self.battle_id: Optional[str] = None
        
        # Player mappings
        self.player_units_map: Dict[int, str] = {}  # unit_id -> player_id
        self.unit_type_map: Dict[int, str] = {}  # unit_id -> unit_type_id
        
        # Deployment zones (restricted areas before battle starts)
        self.deployment_zones = {
            Team.PLAYER: (50, 50, 300, height - 100),
            Team.ENEMY: (width - 350, 50, 300, height - 100),
            Team.ALLY: (50, 50, 300, height - 100)
        }
        
        # Selection state
        self.selected_units: Set[int] = set()
        
        # Obstacles (for future pathfinding)
        self.obstacles = [
            (200, 150, 100, 50),
            (500, 300, 80, 80),
            (350, 450, 120, 40)
        ]
        
        # Initialize systems
        self._init_systems()
        
        logger.info(f"World initialized: {width}x{height}")
    
    def _init_systems(self):
        """Initialize all game logic systems"""
        self.ai_system = AISystem(self)
        self.movement_system = MovementSystem(self)
        self.collision_system = CollisionSystem(self)
        self.stagger_system = StaggerSystem(self)
        self.combat_system = CombatSystem(self)
        self.health_death_system = HealthDeathSystem(self)
        
        # Ability systems
        self.berserker_system = BerserkerRageSystem(self)
        # Add other ability systems here
        
        logger.info("Systems initialized")
    
    def load_unit_data(self, filepath: str = "units.npy"):
        """
        Load unit base data from .npy file.
        
        Args:
            filepath: Path to the units.npy file
        """
        try:
            unit_data = np.load(filepath)
            
            for unit in unit_data:
                unit_id = int(unit['id'])
                # Find unit name from UNIT_TYPE_MAP (reverse lookup)
                unit_name = next((k for k, v in UNIT_TYPE_MAP.items() if v == unit_id), f"unit_{unit_id}")
                
                template = UnitTemplate(
                    unit_type_id=unit_name,
                    max_health=float(unit['max_health']),
                    attack_damage=float(unit['attack_damage']),
                    attack_range=float(unit['attack_range']),
                    defense=float(unit['defense']),
                    speed=float(unit['speed']),
                    accuracy=float(unit['accuracy']),
                    dodge_chance=float(unit['block_chance']),  # Using block_chance as dodge
                    crit_chance=float(unit['crit_chance']),
                    attack_speed=float(unit['attack_speed'])
                )
                
                self.unit_templates[unit_name] = template
            
            logger.info(f"Loaded {len(self.unit_templates)} unit templates from {filepath}")
            
        except FileNotFoundError:
            logger.warning(f"Unit data file not found: {filepath}. Using defaults.")
        except Exception as e:
            logger.error(f"Error loading unit data: {e}")
    
    def register_unit_template(self, unit_type_id: str, template_data: Dict[str, Any]):
        """
        Register a unit template dynamically.
        
        Args:
            unit_type_id: Unique identifier for this unit type
            template_data: Dictionary with unit stats
        """
        template = UnitTemplate(
            unit_type_id=unit_type_id,
            max_health=template_data.get('health', DEFAULT_HEALTH),
            attack_damage=template_data.get('attack_damage', DEFAULT_ATTACK_DAMAGE),
            attack_range=template_data.get('attack_range', DEFAULT_ATTACK_RANGE),
            defense=template_data.get('defense', 0.0),
            speed=template_data.get('speed', MOVEMENT_SPEED),
            accuracy=template_data.get('accuracy', 1.0),
            dodge_chance=template_data.get('dodge_chance', DEFAULT_DODGE_CHANCE),
            crit_chance=template_data.get('crit_chance', 0.0),
            attack_speed=template_data.get('attack_speed', 1.0)
        )
        
        self.unit_templates[unit_type_id] = template
        logger.info(f"Registered unit template: {unit_type_id}")
    
    def spawn_unit(
        self,
        team_id: int,
        num_soldiers: int,
        position: Tuple[float, float],
        unit_type_id: str,
        player_id: Optional[str] = None
    ) -> int:
        """
        Spawn a new unit with multiple soldiers.
        
        Args:
            team_id: Team affiliation
            num_soldiers: Number of soldiers in the unit
            position: Starting position (x, y)
            unit_type_id: Type of unit to spawn
            player_id: Optional player identifier
            
        Returns:
            unit_id of the spawned unit, or -1 on failure
        """
        # Get unit template
        template = self.unit_templates.get(unit_type_id)
        if not template:
            logger.error(f"Unknown unit type: {unit_type_id}")
            return -1
        
        # Generate unit ID
        unit_id = self.state.next_unit_id
        self.state.next_unit_id += 1
        
        # Track ownership
        self.unit_type_map[unit_id] = unit_type_id
        if player_id:
            self.player_units_map[unit_id] = player_id
        
        # Create grid formation
        cols = int(np.sqrt(num_soldiers))
        rows = (num_soldiers + cols - 1) // cols
        
        positions = []
        for i in range(num_soldiers):
            row = i // cols
            col = i % cols
            x = position[0] + col * FORMATION_SPACING
            y = position[1] + row * FORMATION_SPACING
            positions.append([x, y])
        
        positions = np.array(positions, dtype=np.float32)
        
        # Validate deployment zone if battle hasn't started
        if not self.battle_started and team_id in self.deployment_zones:
            if not self._validate_positions_in_zone(team_id, positions):
                logger.warning(f"Cannot spawn unit {unit_id} outside deployment zone")
                return -1
        
        # Prepare entity data
        entity_data = {
            'positions': positions,
            'teams': np.full(num_soldiers, team_id, dtype=np.int32),
            'unit_ids': np.full(num_soldiers, unit_id, dtype=np.int32),
            'unit_types': np.full(num_soldiers, UNIT_TYPE_MAP.get(unit_type_id, 0), dtype=np.int32),
            'target_positions': positions.copy(),
            'healths': np.full(num_soldiers, template.max_health, dtype=np.float32),
            'max_healths': np.full(num_soldiers, template.max_health, dtype=np.float32),
            'attack_damages': np.full(num_soldiers, template.attack_damage, dtype=np.float32),
            'attack_ranges': np.full(num_soldiers, template.attack_range, dtype=np.float32),
            'attack_speeds': np.full(num_soldiers, template.attack_speed, dtype=np.float32),
            'defenses': np.full(num_soldiers, template.defense, dtype=np.float32),
            'accuracies': np.full(num_soldiers, template.accuracy, dtype=np.float32),
            'speeds': np.full(num_soldiers, template.speed, dtype=np.float32),
            'dodge_chances': np.full(num_soldiers, template.dodge_chance, dtype=np.float32),
            'crit_chances': np.full(num_soldiers, template.crit_chance, dtype=np.float32),
            'states': np.full(num_soldiers, UnitState.IDLE, dtype=np.int32)
        }
        
        # Add entities to game state
        self.state.add_entities(num_soldiers, **entity_data)
        
        logger.info(f"Spawned unit {unit_id} ({unit_type_id}) with {num_soldiers} soldiers for team {team_id}")
        return unit_id
    
    def _validate_positions_in_zone(self, team_id: int, positions: np.ndarray) -> bool:
        """Check if all positions are within deployment zone"""
        zone = self.deployment_zones.get(team_id)
        if not zone:
            return True
        
        zx, zy, zw, zh = zone
        in_zone = (
            (positions[:, 0] >= zx) & (positions[:, 0] <= zx + zw) &
            (positions[:, 1] >= zy) & (positions[:, 1] <= zy + zh)
        )
        return np.all(in_zone)
    
    def move_unit(self, unit_id: int, target: Tuple[float, float]):
        """Move a unit to target position maintaining formation"""
        mask = self.state.unit_ids == unit_id
        if not np.any(mask):
            logger.warning(f"Unit {unit_id} not found")
            return
        
        team_id = int(self.state.teams[mask][0])
        
        # Clamp to deployment zone if battle hasn't started
        if not self.battle_started and team_id in self.deployment_zones:
            target = self._clamp_to_zone(team_id, target)
        
        # Calculate offset for each soldier to maintain formation
        unit_positions = self.state.positions[mask]
        centroid = np.mean(unit_positions, axis=0)
        offsets = unit_positions - centroid
        new_targets = np.array(target, dtype=np.float32) + offsets
        
        if not self.battle_started:
            # Instant teleport during deployment
            self.state.positions[mask] = new_targets
            self.state.target_positions[mask] = new_targets
            self.state.states[mask] = UnitState.IDLE
        else:
            # Gradual movement during battle
            self.state.target_positions[mask] = new_targets
            self.state.states[mask] = UnitState.MOVING
        
        self.state.state_timers[mask] = 0.0
    
    def _clamp_to_zone(self, team_id: int, point: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp point to deployment zone"""
        if self.battle_started or team_id not in self.deployment_zones:
            x = np.clip(point[0], 0, self.width)
            y = np.clip(point[1], 0, self.height)
            return (float(x), float(y))
        
        zx, zy, zw, zh = self.deployment_zones[team_id]
        x = np.clip(point[0], zx, zx + zw)
        y = np.clip(point[1], zy, zy + zh)
        return (float(x), float(y))
    
    def start_battle(self):
        """Start the battle phase"""
        self.battle_started = True
        logger.info("Battle started!")
        
        # Transition moving units to MOVING state
        if self.state.entity_count > 0:
            diffs = np.linalg.norm(
                self.state.target_positions - self.state.positions,
                axis=1
            )
            moving_mask = diffs > 1e-3
            self.state.states[moving_mask] = UnitState.MOVING
            self.state.state_timers[moving_mask] = 0.0
    
    def check_battle_end(self) -> Optional[str]:
        """
        Check if battle has ended.
        
        Returns:
            "allies" if allies won, "enemies" if enemies won,
            -1 for draw, None if battle ongoing
        """
        if not self.battle_started:
            return None
        
        alive_mask = self.state.healths > 0
        
        if not np.any(alive_mask):
            return -1  # Draw
        
        allies_alive = np.any(alive_mask & np.isin(self.state.teams, [Team.PLAYER, Team.ALLY]))
        enemies_alive = np.any(alive_mask & (self.state.teams == Team.ENEMY))
        
        if allies_alive and not enemies_alive:
            return "allies"
        elif enemies_alive and not allies_alive:
            return "enemies"
        
        return None  # Battle ongoing
    
    def generate_battle_report(self) -> Dict[str, Any]:
        """Generate detailed battle report"""
        report = {
            "battle_id": self.battle_id,
            "winner": self.check_battle_end(),
            "teams": {},
            "alliances": {},
            "surviving_units": []
        }
        
        alive_mask = self.state.healths > 0
        
        # Per-team statistics
        for team in Team:
            team_mask = (self.state.teams == team) & alive_mask
            unit_ids_in_team = np.unique(self.state.unit_ids[self.state.teams == team])
            
            team_stats = {
                "units_alive": len(unit_ids_in_team),
                "soldiers_alive": int(np.sum(team_mask)),
                "total_kills": int(np.sum(self.state.kill_counts[self.state.teams == team])),
                "units": []
            }
            
            for unit_id in unit_ids_in_team:
                unit_mask = (self.state.unit_ids == unit_id) & alive_mask
                if np.any(unit_mask):
                    unit_type = self.unit_type_map.get(unit_id, "unknown")
                    player_id = self.player_units_map.get(unit_id)
                    
                    unit_info = {
                        "unit_id": int(unit_id),
                        "unit_type": unit_type,
                        "player_id": player_id,
                        "soldiers_alive": int(np.sum(unit_mask)),
                        "total_kills": int(np.sum(self.state.kill_counts[unit_mask])),
                        "total_evades": int(np.sum(self.state.evade_counts[unit_mask]))
                    }
                    team_stats["units"].append(unit_info)
                    
                    if np.any(unit_mask):
                        report["surviving_units"].append(unit_info)
            
            report["teams"][team.name.lower()] = team_stats
        
        # Alliance statistics (combining player and ally teams)
        ally_mask = alive_mask & np.isin(self.state.teams, [Team.PLAYER, Team.ALLY])
        enemy_mask = alive_mask & (self.state.teams == Team.ENEMY)
        
        report["alliances"] = {
            "allies": {
                "soldiers_alive": int(np.sum(ally_mask)),
                "total_kills": int(np.sum(self.state.kill_counts[np.isin(self.state.teams, [Team.PLAYER, Team.ALLY])]))
            },
            "enemies": {
                "soldiers_alive": int(np.sum(enemy_mask)),
                "total_kills": int(np.sum(self.state.kill_counts[self.state.teams == Team.ENEMY]))
            }
        }
        
        return report
    
    def update(self, dt: float):
        """
        Update game world by one time step.
        Runs all systems in order.
        
        Args:
            dt: Delta time in seconds
        """
        if self.state.entity_count == 0:
            return
        
        # Rebuild spatial index
        self.state.rebuild_quadtree()
        
        # Run systems in order
        self.stagger_system.update(dt)
        self.ai_system.update(dt)
        self.movement_system.update(dt)
        self.collision_system.update(dt)
        self.combat_system.update(dt)
        
        # Ability systems
        self.berserker_system.update(dt)
        # Add other ability system updates here
        
        # Health and death
        self.health_death_system.update(dt)
        
        # Clean up dead entities periodically
        if hasattr(self, '_cleanup_timer'):
            self._cleanup_timer += dt
            if self._cleanup_timer >= 1.0:  # Cleanup every second
                self.state.remove_dead_entities()
                self._cleanup_timer = 0.0
        else:
            self._cleanup_timer = 0.0
    
    def get_client_state(self) -> Dict[str, Any]:
        """
        Generate state snapshot for client rendering.
        
        Returns:
            Dictionary containing all entity data for rendering
        """
        alive_mask = self.state.healths > 0
        
        if not np.any(alive_mask):
            return {
                "units": [],
                "battle_started": self.battle_started
            }
        
        # Group entities by unit_id
        units_data = {}
        
        for i in np.where(alive_mask)[0]:
            unit_id = int(self.state.unit_ids[i])
            
            if unit_id not in units_data:
                units_data[unit_id] = {
                    "unit_id": unit_id,
                    "team": int(self.state.teams[i]),
                    "unit_type": self.unit_type_map.get(unit_id, "unknown"),
                    "soldiers": []
                }
            
            soldier_data = {
                "pos": [float(self.state.positions[i, 0]), float(self.state.positions[i, 1])],
                "health": float(self.state.healths[i]),
                "max_health": float(self.state.max_healths[i]),
                "state": int(self.state.states[i]),
                "target_entity": int(self.state.target_entity_ids[i])
            }
            
            units_data[unit_id]["soldiers"].append(soldier_data)
        
        return {
            "units": list(units_data.values()),
            "battle_started": self.battle_started
        }


# ===== WebSocket Server =====
class BattleServer:
    """
    WebSocket server for real-time battle simulation.
    Handles client connections and game loop.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.world = World()
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        self.game_loop_task = None
        
        logger.info(f"Battle server initialized on {host}:{port}")
    
    async def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send current state to new client
        await self.send_state_to_client(websocket)
    
    async def unregister_client(self, websocket):
        """Remove client connection"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_state_to_client(self, websocket):
        """Send current game state to specific client"""
        try:
            state = self.world.get_client_state()
            await websocket.send(json.dumps({
                "type": "state_update",
                "data": state
            }))
        except Exception as e:
            logger.error(f"Error sending state to client: {e}")
    
    async def broadcast_state(self):
        """Broadcast game state to all connected clients"""
        if not self.clients:
            return
        
        state = self.world.get_client_state()
        message = json.dumps({
            "type": "state_update",
            "data": state
        })
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    async def handle_message(self, websocket, message: str):
        """
        Process incoming client messages.
        
        Message format:
        {
            "type": "command_type",
            "data": { ... }
        }
        """
        try:
            msg = json.loads(message)
            msg_type = msg.get("type")
            data = msg.get("data", {})
            
            if msg_type == "spawn_unit":
                unit_id = self.world.spawn_unit(
                    team_id=data.get("team", Team.PLAYER),
                    num_soldiers=data.get("num_soldiers", 10),
                    position=tuple(data.get("position", [100, 100])),
                    unit_type_id=data.get("unit_type", "warrior"),
                    player_id=data.get("player_id")
                )
                
                await websocket.send(json.dumps({
                    "type": "spawn_response",
                    "unit_id": unit_id
                }))
            
            elif msg_type == "move_unit":
                unit_id = data.get("unit_id")
                target = tuple(data.get("target", [0, 0]))
                self.world.move_unit(unit_id, target)
            
            elif msg_type == "start_battle":
                self.world.battle_id = data.get("battle_id", "battle_001")
                self.world.start_battle()
                await self.broadcast_state()
            
            elif msg_type == "get_state":
                await self.send_state_to_client(websocket)
            
            elif msg_type == "register_unit_type":
                unit_type_id = data.get("unit_type_id")
                template_data = data.get("template")
                if unit_type_id and template_data:
                    self.world.register_unit_template(unit_type_id, template_data)
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def game_loop(self):
        """Main game loop - runs at TICK_RATE"""
        dt = 1.0 / TICK_RATE
        
        while self.running:
            start_time = asyncio.get_event_loop().time()
            
            # Update game world
            self.world.update(dt)
            
            # Check for battle end
            if self.world.battle_started:
                result = self.world.check_battle_end()
                if result is not None:
                    logger.info(f"Battle ended! Winner: {result}")
                    
                    # Generate and send battle report
                    report = self.world.generate_battle_report()
                    await self.send_battle_results(report)
                    
                    # Notify clients
                    await self.broadcast_message({
                        "type": "battle_end",
                        "winner": result,
                        "report": report
                    })
                    
                    # Stop battle
                    self.world.battle_started = False
            
            # Broadcast state to clients
            await self.broadcast_state()
            
            # Maintain consistent tick rate
            elapsed = asyncio.get_event_loop().time() - start_time
            sleep_time = max(0, dt - elapsed)
            await asyncio.sleep(sleep_time)
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast arbitrary message to all clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.add(client)
        
        self.clients -= disconnected
    
    async def send_battle_results(self, report: Dict[str, Any]):
        """Send battle results to world server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(WORLD_SERVER_URL, json=report) as response:
                    if response.status == 200:
                        logger.info("Battle results sent to world server")
                    else:
                        logger.error(f"Failed to send battle results: {response.status}")
        except Exception as e:
            logger.error(f"Error sending battle results: {e}")
    
    async def handle_client(self, websocket, path):
        """Handle individual client connection"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            await self.unregister_client(websocket)
    
    async def start(self):
        """Start the server"""
        self.running = True
        
        # Load unit data
        self.world.load_unit_data("units.npy")
        
        # Start game loop
        self.game_loop_task = asyncio.create_task(self.game_loop())
        
        # Start WebSocket server
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"Battle server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def stop(self):
        """Stop the server"""
        self.running = False
        if self.game_loop_task:
            self.game_loop_task.cancel()
        logger.info("Battle server stopped")


# ===== Entry Point =====
async def main():
    """Main entry point"""
    server = BattleServer(host="localhost", port=8765)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())