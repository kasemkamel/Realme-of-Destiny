import asyncio
import websockets
import json
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
import random

class TileType(Enum):
    WATER = 0
    FLAT_LAND = 1
    MOUNTAIN = 2
    HIGH_GROUND = 3

class BuildingType(Enum):
    BARRACKS = "barracks"
    FARM = "farm"
    WALL = "wall"
    MARKET = "market"
    MINE = "mine"

@dataclass
class Tile:
    x: int
    y: int
    tile_type: TileType
    province_id: Optional[int] = None

@dataclass
class Province:
    id: int
    name: str
    tiles: List[tuple]
    owner_id: Optional[str] = None
    city: Optional['City'] = None

@dataclass
class Building:
    building_type: BuildingType
    x: int
    y: int
    level: int = 1

@dataclass
class City:
    name: str
    province_id: int
    x: int
    y: int
    buildings: List[Building]
    resources: Dict[str, int]

@dataclass
class Player:
    id: str
    name: str
    cities: List[City]
    provinces: List[int]
    resources: Dict[str, int]

class GameServer:
    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height
        self.world_map: List[List[Tile]] = []
        self.provinces: Dict[int, Province] = {}
        self.players: Dict[str, Player] = {}
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.province_counter = 0
        
        self.initialize_world()
    
    def initialize_world(self):
        """إنشاء خريطة العالم"""
        print("جاري إنشاء خريطة العالم...")
        
        # إنشاء الشبكة الأساسية
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # توزيع عشوائي لأنواع الخانات
                rand = random.random()
                if rand < 0.15:
                    tile_type = TileType.WATER
                elif rand < 0.75:
                    tile_type = TileType.FLAT_LAND
                elif rand < 0.90:
                    tile_type = TileType.HIGH_GROUND
                else:
                    tile_type = TileType.MOUNTAIN
                
                row.append(Tile(x, y, tile_type))
            self.world_map.append(row)
        
        # إنشاء المقاطعات
        self.generate_provinces()
        print(f"تم إنشاء {len(self.provinces)} مقاطعة")
    
    def generate_provinces(self):
        """إنشاء المقاطعات بتقسيم الخريطة"""
        province_size = 10
        
        for py in range(0, self.height, province_size):
            for px in range(0, self.width, province_size):
                tiles = []
                
                for y in range(py, min(py + province_size, self.height)):
                    for x in range(px, min(px + province_size, self.width)):
                        if self.world_map[y][x].tile_type != TileType.WATER:
                            tiles.append((x, y))
                            self.world_map[y][x].province_id = self.province_counter
                
                if tiles:
                    province = Province(
                        id=self.province_counter,
                        name=f"مقاطعة {self.province_counter}",
                        tiles=tiles
                    )
                    self.provinces[self.province_counter] = province
                    self.province_counter += 1
    
    async def handle_client(self, websocket):
        """معالجة اتصال الكلاينت"""
        player_id = None
        
        try:
            async for message in websocket:
                data = json.loads(message)
                action = data.get('action')
                
                if action == 'connect':
                    player_id = data.get('player_id')
                    player_name = data.get('player_name', f'لاعب {player_id}')
                    
                    if player_id not in self.players:
                        self.players[player_id] = Player(
                            id=player_id,
                            name=player_name,
                            cities=[],
                            provinces=[],
                            resources={'gold': 1000, 'food': 500, 'wood': 300}
                        )
                    
                    self.connections[player_id] = websocket
                    
                    # إرسال حالة العالم للاعب
                    await self.send_world_state(player_id)
                    print(f"اللاعب {player_name} اتصل بالسيرفر")
                
                elif action == 'claim_province':
                    province_id = data.get('province_id')
                    await self.claim_province(player_id, province_id)
                
                elif action == 'build_city':
                    province_id = data.get('province_id')
                    city_name = data.get('city_name', 'مدينة جديدة')
                    await self.build_city(player_id, province_id, city_name)
                
                elif action == 'get_city':
                    city_index = data.get('city_index')
                    await self.send_city_data(player_id, city_index)
                
                elif action == 'build_building':
                    city_index = data.get('city_index')
                    building_type = data.get('building_type')
                    x = data.get('x')
                    y = data.get('y')
                    await self.build_building(player_id, city_index, building_type, x, y)
        
        except websockets.exceptions.ConnectionClosed:
            print(f"اللاعب {player_id} انقطع اتصاله")
        finally:
            if player_id and player_id in self.connections:
                del self.connections[player_id]
    
    async def send_world_state(self, player_id: str):
        """إرسال حالة العالم للاعب"""
        if player_id not in self.connections:
            return
        
        # تحويل الخريطة لصيغة قابلة للإرسال
        map_data = []
        for row in self.world_map:
            map_row = []
            for tile in row:
                map_row.append({
                    'x': tile.x,
                    'y': tile.y,
                    'type': tile.tile_type.value,
                    'province_id': tile.province_id
                })
            map_data.append(map_row)
        
        provinces_data = {}
        for pid, province in self.provinces.items():
            provinces_data[pid] = {
                'id': province.id,
                'name': province.name,
                'owner_id': province.owner_id,
                'has_city': province.city is not None
            }
        
        player = self.players[player_id]
        
        response = {
            'type': 'world_state',
            'map': map_data,
            'provinces': provinces_data,
            'player': {
                'id': player.id,
                'name': player.name,
                'resources': player.resources,
                'provinces': player.provinces,
                'cities': len(player.cities)
            }
        }
        
        await self.connections[player_id].send(json.dumps(response))
    
    async def claim_province(self, player_id: str, province_id: int):
        """السيطرة على مقاطعة"""
        if province_id not in self.provinces:
            return
        
        province = self.provinces[province_id]
        
        if province.owner_id is None:
            province.owner_id = player_id
            self.players[player_id].provinces.append(province_id)
            
            # إرسال تحديث لجميع اللاعبين
            await self.broadcast_province_update(province_id)
    
    async def build_city(self, player_id: str, province_id: int, city_name: str):
        """بناء مدينة في مقاطعة"""
        if province_id not in self.provinces:
            return
        
        province = self.provinces[province_id]
        
        if province.owner_id != player_id or province.city is not None:
            return
        
        # اختيار موقع المدينة (أول خانة متاحة)
        if not province.tiles:
            return
        
        city_x, city_y = province.tiles[0]
        
        city = City(
            name=city_name,
            province_id=province_id,
            x=city_x,
            y=city_y,
            buildings=[],
            resources={'population': 100}
        )
        
        province.city = city
        self.players[player_id].cities.append(city)
        
        await self.send_world_state(player_id)
    
    async def send_city_data(self, player_id: str, city_index: int):
        """إرسال بيانات مدينة محددة"""
        player = self.players.get(player_id)
        if not player or city_index >= len(player.cities):
            return
        
        city = player.cities[city_index]
        
        response = {
            'type': 'city_data',
            'city': {
                'name': city.name,
                'province_id': city.province_id,
                'x': city.x,
                'y': city.y,
                'resources': city.resources,
                'buildings': [
                    {
                        'type': b.building_type.value,
                        'x': b.x,
                        'y': b.y,
                        'level': b.level
                    } for b in city.buildings
                ]
            }
        }
        
        await self.connections[player_id].send(json.dumps(response))
    
    async def build_building(self, player_id: str, city_index: int, building_type: str, x: int, y: int):
        """بناء مبنى في المدينة"""
        player = self.players.get(player_id)
        if not player or city_index >= len(player.cities):
            return
        
        city = player.cities[city_index]
        
        try:
            b_type = BuildingType(building_type)
            building = Building(b_type, x, y)
            city.buildings.append(building)
            
            await self.send_city_data(player_id, city_index)
        except ValueError:
            pass
    
    async def broadcast_province_update(self, province_id: int):
        """إرسال تحديث المقاطعة لجميع اللاعبين"""
        for player_id in self.connections:
            await self.send_world_state(player_id)
    
    async def start(self, host='localhost', port=8765):
        """تشغيل السيرفر"""
        print(f"السيرفر يعمل على {host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()

if __name__ == '__main__':
    server = GameServer(width=40, height=40)
    asyncio.run(server.start())