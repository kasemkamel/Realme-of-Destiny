import pygame
import websockets
import json
import random
import threading
import asyncio
from enum import Enum
from queue import Queue

# إعدادات الشاشة
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
TILE_SIZE = 20
UI_HEIGHT = 150

class GameState(Enum):
    WORLD_MAP = 1
    CITY_VIEW = 2

class TileType(Enum):
    WATER = 0
    FLAT_LAND = 1
    MOUNTAIN = 2
    HIGH_GROUND = 3

# الألوان
COLORS = {
    'water': (50, 100, 200),
    'flat_land': (100, 200, 100),
    'mountain': (120, 120, 120),
    'high_ground': (150, 150, 100),
    'ui_bg': (40, 40, 40),
    'text': (255, 255, 255),
    'city': (255, 215, 0),
    'selected': (255, 255, 0),
    'owned': (100, 255, 100, 50)
}

class GameClient:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Realme of Destiny")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # حالة اللعبة
        self.running = True
        self.connected = False
        self.game_state = GameState.WORLD_MAP
        self.player_id = f"player_{random.randint(1000, 9999)}"
        self.player_name = f"لاعب {random.randint(1, 100)}"
        
        # بيانات اللعبة
        self.world_map = []
        self.provinces = {}
        self.player_data = {}
        self.current_city = None
        self.current_city_index = None
        
        # كاميرا العالم
        self.camera_x = 0
        self.camera_y = 0
        
        # التفاعل
        self.selected_province = None
        self.selected_building_type = None
        
        # قوائم الانتظار للاتصال
        self.send_queue = Queue()
        self.receive_queue = Queue()
        
        # بدء خيط الاتصال
        self.ws_thread = threading.Thread(target=self.run_websocket, daemon=True)
        self.ws_thread.start()
    
    def run_websocket(self):
        """تشغيل WebSocket في خيط منفصل"""
        asyncio.run(self.websocket_handler())
    
    async def websocket_handler(self):
        """معالج WebSocket"""
        try:
            async with websockets.connect('ws://localhost:8765') as websocket:
                self.connected = True
                print(f"تم الاتصال بالسيرفر كـ {self.player_name}")
                
                # إرسال طلب الاتصال
                await websocket.send(json.dumps({
                    'action': 'connect',
                    'player_id': self.player_id,
                    'player_name': self.player_name
                }))
                
                # حلقة الاستقبال والإرسال
                while self.running:
                    try:
                        # إرسال الرسائل من القائمة
                        while not self.send_queue.empty():
                            message = self.send_queue.get()
                            await websocket.send(json.dumps(message))
                        
                        # استقبال الرسائل
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            data = json.loads(message)
                            self.receive_queue.put(data)
                        except asyncio.TimeoutError:
                            pass
                    
                    except Exception as e:
                        print(f"خطأ في الاتصال: {e}")
                        break
        
        except Exception as e:
            print(f"فشل الاتصال بالسيرفر: {e}")
            self.connected = False
    
    def process_messages(self):
        """معالجة الرسائل المستلمة"""
        while not self.receive_queue.empty():
            data = self.receive_queue.get()
            msg_type = data.get('type')
            
            if msg_type == 'world_state':
                self.world_map = data['map']
                self.provinces = data['provinces']
                self.player_data = data['player']
                print(f"تم استلام خريطة العالم: {len(self.world_map)}x{len(self.world_map[0]) if self.world_map else 0}")
            
            elif msg_type == 'city_data':
                self.current_city = data['city']
                self.game_state = GameState.CITY_VIEW
                print(f"تم فتح المدينة: {self.current_city['name']}")
    
    def send_action(self, action_data):
        """إرسال إجراء للسيرفر"""
        self.send_queue.put(action_data)
    
    def draw_world_map(self):
        """رسم خريطة العالم"""
        if not self.world_map:
            return
        
        # رسم الخانات
        for row in self.world_map:
            for tile in row:
                x = tile['x'] * TILE_SIZE - self.camera_x
                y = tile['y'] * TILE_SIZE - self.camera_y
                
                if -TILE_SIZE < x < SCREEN_WIDTH and -TILE_SIZE < y < SCREEN_HEIGHT - UI_HEIGHT:
                    tile_type = tile['type']
                    
                    if tile_type == TileType.WATER.value:
                        color = COLORS['water']
                    elif tile_type == TileType.FLAT_LAND.value:
                        color = COLORS['flat_land']
                    elif tile_type == TileType.MOUNTAIN.value:
                        color = COLORS['mountain']
                    else:
                        color = COLORS['high_ground']
                    
                    pygame.draw.rect(self.screen, color, (x, y, TILE_SIZE, TILE_SIZE))
                    pygame.draw.rect(self.screen, (0, 0, 0), (x, y, TILE_SIZE, TILE_SIZE), 1)
                    
                    # تمييز المقاطعات المملوكة
                    province_id = tile.get('province_id')
                    if province_id is not None and str(province_id) in self.provinces:
                        province = self.provinces[str(province_id)]
                        if province.get('owner_id') == self.player_id:
                            s = pygame.Surface((TILE_SIZE, TILE_SIZE))
                            s.set_alpha(50)
                            s.fill((100, 255, 100))
                            self.screen.blit(s, (x, y))
        
        # رسم المدن
        cities_drawn = set()
        for province_id, province in self.provinces.items():
            if province.get('has_city') and province_id not in cities_drawn:
                # العثور على موقع المدينة (أول خانة في المقاطعة)
                for row in self.world_map:
                    for tile in row:
                        if tile.get('province_id') == int(province_id):
                            x = tile['x'] * TILE_SIZE - self.camera_x + TILE_SIZE // 2
                            y = tile['y'] * TILE_SIZE - self.camera_y + TILE_SIZE // 2
                            
                            if -50 < x < SCREEN_WIDTH + 50 and -50 < y < SCREEN_HEIGHT - UI_HEIGHT + 50:
                                pygame.draw.circle(self.screen, COLORS['city'], (int(x), int(y)), 8)
                                pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 8, 2)
                            
                            cities_drawn.add(province_id)
                            break
                    if province_id in cities_drawn:
                        break
    
    def draw_ui(self):
        """رسم واجهة المستخدم"""
        # خلفية UI
        pygame.draw.rect(self.screen, COLORS['ui_bg'], 
                        (0, SCREEN_HEIGHT - UI_HEIGHT, SCREEN_WIDTH, UI_HEIGHT))
        
        if self.game_state == GameState.WORLD_MAP:
            # معلومات اللاعب
            y_offset = SCREEN_HEIGHT - UI_HEIGHT + 10
            
            if self.player_data:
                text = self.font.render(f"اللاعب: {self.player_data.get('name', '')}", True, COLORS['text'])
                self.screen.blit(text, (10, y_offset))
                
                y_offset += 30
                resources = self.player_data.get('resources', {})
                res_text = f"ذهب: {resources.get('gold', 0)} | طعام: {resources.get('food', 0)} | خشب: {resources.get('wood', 0)}"
                text = self.small_font.render(res_text, True, COLORS['text'])
                self.screen.blit(text, (10, y_offset))
                
                y_offset += 25
                info_text = f"مقاطعات: {len(self.player_data.get('provinces', []))} | مدن: {self.player_data.get('cities', 0)}"
                text = self.small_font.render(info_text, True, COLORS['text'])
                self.screen.blit(text, (10, y_offset))
            
            # تعليمات
            instructions = [
                "استخدم WASD للتحرك",
                "انقر على مقاطعة للسيطرة عليها (C)",
                "انقر على مقاطعة مملوكة لبناء مدينة (B)",
                "انقر على مدينة لفتحها"
            ]
            
            y_offset = SCREEN_HEIGHT - UI_HEIGHT + 10
            for i, inst in enumerate(instructions):
                text = self.small_font.render(inst, True, COLORS['text'])
                self.screen.blit(text, (SCREEN_WIDTH - 400, y_offset + i * 20))
        
        elif self.game_state == GameState.CITY_VIEW:
            # معلومات المدينة
            if self.current_city:
                y_offset = SCREEN_HEIGHT - UI_HEIGHT + 10
                text = self.font.render(f"المدينة: {self.current_city['name']}", True, COLORS['text'])
                self.screen.blit(text, (10, y_offset))
                
                y_offset += 30
                text = self.small_font.render("اختر مبنى للبناء: 1-Barracks 2-Farm 3-Wall 4-Market 5-Mine", True, COLORS['text'])
                self.screen.blit(text, (10, y_offset))
                
                y_offset += 25
                text = self.small_font.render("انقر على الشبكة لوضع المبنى | ESC للعودة", True, COLORS['text'])
                self.screen.blit(text, (10, y_offset))
                
                if self.selected_building_type:
                    y_offset += 25
                    text = self.small_font.render(f"المبنى المختار: {self.selected_building_type}", True, (255, 255, 0))
                    self.screen.blit(text, (10, y_offset))
    
    def draw_city_view(self):
        """رسم شاشة المدينة"""
        if not self.current_city:
            return
        
        # شبكة البناء (15x15)
        grid_size = 15
        cell_size = 40
        start_x = (SCREEN_WIDTH - grid_size * cell_size) // 2
        start_y = 50
        
        # رسم الشبكة
        for y in range(grid_size):
            for x in range(grid_size):
                rect = pygame.Rect(start_x + x * cell_size, start_y + y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, (80, 80, 80), rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # رسم المباني
        building_colors = {
            'barracks': (200, 50, 50),
            'farm': (255, 200, 50),
            'wall': (100, 100, 100),
            'market': (50, 150, 200),
            'mine': (150, 100, 50)
        }
        
        for building in self.current_city.get('buildings', []):
            bx = building['x']
            by = building['y']
            b_type = building['type']
            
            if 0 <= bx < grid_size and 0 <= by < grid_size:
                rect = pygame.Rect(start_x + bx * cell_size + 2, start_y + by * cell_size + 2, 
                                  cell_size - 4, cell_size - 4)
                color = building_colors.get(b_type, (100, 100, 100))
                pygame.draw.rect(self.screen, color, rect)
                
                # اسم المبنى
                text = self.small_font.render(b_type[:4], True, COLORS['text'])
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)
    
    def handle_world_click(self, pos):
        """معالجة النقر على خريطة العالم"""
        x = (pos[0] + self.camera_x) // TILE_SIZE
        y = (pos[1] + self.camera_y) // TILE_SIZE
        
        if 0 <= y < len(self.world_map) and 0 <= x < len(self.world_map[0]):
            tile = self.world_map[y][x]
            province_id = tile.get('province_id')
            
            if province_id is not None:
                self.selected_province = province_id
                print(f"تم اختيار المقاطعة: {province_id}")
                return province_id
        
        return None
    
    def handle_city_click(self, pos):
        """معالجة النقر على شبكة المدينة"""
        grid_size = 15
        cell_size = 40
        start_x = (SCREEN_WIDTH - grid_size * cell_size) // 2
        start_y = 50
        
        if self.selected_building_type:
            x = (pos[0] - start_x) // cell_size
            y = (pos[1] - start_y) // cell_size
            
            if 0 <= x < grid_size and 0 <= y < grid_size:
                return (x, y)
        
        return None
    
    def run(self):
        """الحلقة الرئيسية للعبة"""
        while self.running:
            self.clock.tick(60)
            
            # معالجة الرسائل
            self.process_messages()
            
            # معالجة الأحداث
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if self.game_state == GameState.WORLD_MAP:
                        if event.key == pygame.K_w:
                            self.camera_y -= TILE_SIZE * 5
                        elif event.key == pygame.K_s:
                            self.camera_y += TILE_SIZE * 5
                        elif event.key == pygame.K_a:
                            self.camera_x -= TILE_SIZE * 5
                        elif event.key == pygame.K_d:
                            self.camera_x += TILE_SIZE * 5
                        
                        elif event.key == pygame.K_c and self.selected_province is not None:
                            print(f"محاولة السيطرة على المقاطعة: {self.selected_province}")
                            self.send_action({
                                'action': 'claim_province',
                                'province_id': self.selected_province
                            })
                        
                        elif event.key == pygame.K_b and self.selected_province is not None:
                            print(f"محاولة بناء مدينة في المقاطعة: {self.selected_province}")
                            self.send_action({
                                'action': 'build_city',
                                'province_id': self.selected_province,
                                'city_name': f'مدينة {random.randint(1, 100)}'
                            })
                    
                    elif self.game_state == GameState.CITY_VIEW:
                        if event.key == pygame.K_ESCAPE:
                            self.game_state = GameState.WORLD_MAP
                            self.current_city = None
                        
                        elif event.key == pygame.K_1:
                            self.selected_building_type = 'barracks'
                        elif event.key == pygame.K_2:
                            self.selected_building_type = 'farm'
                        elif event.key == pygame.K_3:
                            self.selected_building_type = 'wall'
                        elif event.key == pygame.K_4:
                            self.selected_building_type = 'market'
                        elif event.key == pygame.K_5:
                            self.selected_building_type = 'mine'
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # زر الماوس الأيسر
                        if self.game_state == GameState.WORLD_MAP:
                            if event.pos[1] < SCREEN_HEIGHT - UI_HEIGHT:
                                province_id = self.handle_world_click(event.pos)
                                
                                # فتح المدينة إذا كانت موجودة
                                if province_id and str(province_id) in self.provinces:
                                    province = self.provinces[str(province_id)]
                                    if province.get('has_city') and province.get('owner_id') == self.player_id:
                                        # العثور على فهرس المدينة
                                        city_count = 0
                                        for pid in self.player_data.get('provinces', []):
                                            if str(pid) in self.provinces and self.provinces[str(pid)].get('has_city'):
                                                if pid == province_id:
                                                    print(f"فتح المدينة بالفهرس: {city_count}")
                                                    self.send_action({
                                                        'action': 'get_city',
                                                        'city_index': city_count
                                                    })
                                                    self.current_city_index = city_count
                                                    break
                                                city_count += 1
                        
                        elif self.game_state == GameState.CITY_VIEW:
                            pos = self.handle_city_click(event.pos)
                            if pos and self.selected_building_type and self.current_city_index is not None:
                                x, y = pos
                                print(f"بناء {self.selected_building_type} في ({x}, {y})")
                                self.send_action({
                                    'action': 'build_building',
                                    'city_index': self.current_city_index,
                                    'building_type': self.selected_building_type,
                                    'x': x,
                                    'y': y
                                })
                                self.selected_building_type = None
            
            # الرسم
            self.screen.fill((0, 0, 0))
            
            if self.game_state == GameState.WORLD_MAP:
                self.draw_world_map()
            elif self.game_state == GameState.CITY_VIEW:
                self.draw_city_view()
            
            self.draw_ui()
            
            # رسالة الاتصال
            if not self.connected:
                text = self.font.render("جاري الاتصال بالسيرفر...", True, (255, 0, 0))
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 50))
                self.screen.blit(text, text_rect)
            elif not self.world_map:
                text = self.font.render("جاري تحميل الخريطة...", True, (255, 255, 0))
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 50))
                self.screen.blit(text, text_rect)
            
            pygame.display.flip()
        
        pygame.quit()

if __name__ == '__main__':
    client = GameClient()
    client.run()