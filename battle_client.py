import pygame
import asyncio
import websockets
import json
import threading
from typing import Optional, Tuple, List, Set

# ===== Game Client =====
class GameClient:
    def __init__(self, width=800, height=600):
        # إعداد Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Kingdom of Destinies - Prototype")
        self.clock = pygame.time.Clock()
        
        # بيانات حالة اللعبة
        self.world_state = None
        self.state_lock = threading.Lock()
        
        # WebSocket
        self.websocket = None
        self.running = True
        
        # التحكم بالماوس
        self.drag_start = None
        self.dragging = False
        self.selection_box_start = None  # لاختيار الوحدات بالصندوق
        self.selecting = False
        
        # الوحدات المختارة (من السيرفر)
        self.selected_units: Set[int] = set()
        
        # الألوان - تحديث لدعم 3 فرق
        self.COLOR_BG = (40, 44, 52)
        self.COLOR_PLAYER = (220, 50, 50)     # أحمر للاعب (Team 0)
        self.COLOR_ENEMY = (50, 120, 220)      # أزرق للعدو (Team 1)
        self.COLOR_ALLY = (50, 220, 120)       # أخضر للحليف (Team 2)
        self.COLOR_OBSTACLE = (100, 100, 100)
        self.COLOR_DRAG_LINE = (255, 255, 100)
        self.COLOR_SELECTION = (255, 255, 255, 100)  # لون صندوق الاختيار
        self.COLOR_DEPLOYMENT_ZONE = (100, 100, 255, 50)  # لون مناطق النشر
        
        # ألوان أشرطة الصحة
        self.COLOR_HEALTH_BG = (40, 40, 40)
        self.COLOR_HEALTH_HIGH = (50, 205, 50)
        self.COLOR_HEALTH_MEDIUM = (255, 165, 0)
        self.COLOR_HEALTH_LOW = (220, 20, 60)
        
        # حالات الجنود
        self.STATE_IDLE = 0
        self.STATE_MOVING = 1
        self.STATE_ATTACKING = 2
        
        # الخطوط
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.button_font = pygame.font.Font(None, 32)
        
        # زر بدء المعركة
        self.start_button_rect = pygame.Rect(width // 2 - 100, height - 60, 200, 50)
        self.button_color = (70, 150, 70)
        self.button_hover_color = (90, 180, 90)
        self.button_disabled_color = (100, 100, 100)

        self.battle_report = None
    
    async def connect_to_server(self):
        """الاتصال بالخادم واستقبال البيانات"""
        uri = "ws://localhost:8765"
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                print("متصل بالخادم!")
                
                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        state = json.loads(message)
                        
                        with self.state_lock:
                            self.world_state = state
                            # تحديث الوحدات المختارة من السيرفر
                            if 'selected_units' in state:
                                self.selected_units = set(state['selected_units'])
                            if 'battle_report' in state:  # <-- إضافة
                                self.battle_report = state['battle_report']  # <-- إضافة    
                    
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("انقطع الاتصال بالخادم")
                        break
        
        except Exception as e:
            print(f"خطأ في الاتصال: {e}")
    
    async def send_command(self, command: dict):
        """إرسال أمر إلى الخادم"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(command))
            except Exception as e:
                print(f"خطأ في الإرسال: {e}")
    
    def get_unit_at_position(self, pos: Tuple[int, int]) -> Optional[int]:
        """الحصول على unit_id عند نقطة معينة"""
        with self.state_lock:
            state = self.world_state
        
        if not state:
            return None
        
        positions = state.get('positions', [])
        unit_ids = state.get('unit_ids', [])
        teams = state.get('teams', [])
        
        click_radius = 10  # مسافة الكشف
        
        for i, soldier_pos in enumerate(positions):
            dx = soldier_pos[0] - pos[0]
            dy = soldier_pos[1] - pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            
            if dist < click_radius and i < len(teams) and teams[i] == 0:  # فقط وحدات اللاعب
                return unit_ids[i] if i < len(unit_ids) else None
        
        return None
    
    def get_units_in_box(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[int]:
        """الحصول على الوحدات داخل صندوق الاختيار"""
        with self.state_lock:
            state = self.world_state
        
        if not state:
            return []
        
        positions = state.get('positions', [])
        unit_ids = state.get('unit_ids', [])
        teams = state.get('teams', [])
        
        min_x = min(start[0], end[0])
        max_x = max(start[0], end[0])
        min_y = min(start[1], end[1])
        max_y = max(start[1], end[1])
        
        selected = set()
        for i, pos in enumerate(positions):
            if (min_x <= pos[0] <= max_x and 
                min_y <= pos[1] <= max_y and 
                i < len(teams) and teams[i] == 0):  # فقط وحدات اللاعب
                if i < len(unit_ids):
                    selected.add(unit_ids[i])
        
        return list(selected)
    
    def handle_events(self):
        """معالجة مدخلات المستخدم"""
        mouse_pos = pygame.mouse.get_pos()
        
        with self.state_lock:
            battle_started = self.world_state.get('battle_started', False) if self.world_state else False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # التحقق من النقر على زر بدء المعركة
                if not battle_started and self.start_button_rect.collidepoint(event.pos):
                    if event.button == 1:  # النقر الأيسر
                        asyncio.run_coroutine_threadsafe(
                            self.send_command({'type': 'start_battle'}),
                            self.network_loop
                        )
                
                # النقر الأيسر - اختيار وحدات
                elif event.button == 1:
                    # إذا كان Shift مضغوط، لا نلغي الاختيار السابق
                    keys = pygame.key.get_pressed()
                    if not keys[pygame.K_LSHIFT]:
                        asyncio.run_coroutine_threadsafe(
                            self.send_command({'type': 'deselect'}),
                            self.network_loop
                        )
                    
                    # بدء صندوق الاختيار
                    self.selection_box_start = event.pos
                    self.selecting = True
                
                # النقر الأيمن - الأوامر
                elif event.button == 3:
                    if self.selected_units:
                        self.drag_start = event.pos
                        self.dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                # إنهاء اختيار الوحدات
                if event.button == 1 and self.selecting:
                    if self.selection_box_start:
                        # حساب المسافة
                        dx = event.pos[0] - self.selection_box_start[0]
                        dy = event.pos[1] - self.selection_box_start[1]
                        distance = (dx * dx + dy * dy) ** 0.5
                        
                        if distance < 10:  # نقرة بسيطة
                            unit_id = self.get_unit_at_position(event.pos)
                            if unit_id is not None:
                                asyncio.run_coroutine_threadsafe(
                                    self.send_command({
                                        'type': 'select',
                                        'unit_ids': [unit_id]
                                    }),
                                    self.network_loop
                                )
                        else:  # صندوق اختيار
                            selected = self.get_units_in_box(self.selection_box_start, event.pos)
                            if selected:
                                asyncio.run_coroutine_threadsafe(
                                    self.send_command({
                                        'type': 'select',
                                        'unit_ids': selected
                                    }),
                                    self.network_loop
                                )
                    
                    self.selection_box_start = None
                    self.selecting = False
                
                # إنهاء أمر التحريك/التشكيل
                elif event.button == 3 and self.dragging:
                    if self.drag_start and self.selected_units:
                        drag_end = event.pos
                        dx = drag_end[0] - self.drag_start[0]
                        dy = drag_end[1] - self.drag_start[1]
                        distance = (dx * dx + dy * dy) ** 0.5
                        
                        if distance < 10:  # نقرة بسيطة - تحريك
                            asyncio.run_coroutine_threadsafe(
                                self.send_command({
                                    'type': 'move_selected',
                                    'target': list(drag_end)
                                }),
                                self.network_loop
                            )
                        else:  # سحب - تشكيل (للوحدة الأولى المختارة فقط)
                            first_unit = list(self.selected_units)[0]
                            asyncio.run_coroutine_threadsafe(
                                self.send_command({
                                    'type': 'form',
                                    'unit_id': first_unit,
                                    'start': list(self.drag_start),
                                    'end': list(drag_end)
                                }),
                                self.network_loop
                            )
                    
                    self.drag_start = None
                    self.dragging = False
    
    def get_health_color(self, health: float) -> Tuple[int, int, int]:
        """تحديد لون شريط الصحة"""
        health_percentage = health / 100.0
        
        if health_percentage > 0.6:
            return self.COLOR_HEALTH_HIGH
        elif health_percentage > 0.3:
            return self.COLOR_HEALTH_MEDIUM
        else:
            return self.COLOR_HEALTH_LOW
    
    def draw_health_bar(self, x: float, y: float, health: float):
        """رسم شريط الصحة"""
        bar_width = 12
        bar_height = 3
        bar_offset_y = 8
        
        bar_x = int(x - bar_width / 2)
        bar_y = int(y - bar_offset_y)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        
        health_percentage = max(0, min(1, health / 100.0))
        health_bar_width = int(bar_width * health_percentage)
        
        if health_bar_width > 0:
            health_color = self.get_health_color(health)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, health_bar_width, bar_height))
        
        pygame.draw.rect(self.screen, (200, 200, 200), (bar_x, bar_y, bar_width, bar_height), 1)
    
    def get_team_color(self, team: int) -> Tuple[int, int, int]:
        """تحديد لون الفريق"""
        if team == 0:
            return self.COLOR_PLAYER
        elif team == 1:
            return self.COLOR_ENEMY
        elif team == 2:
            return self.COLOR_ALLY
        return (200, 200, 200)
    
    def get_soldier_color(self, team: int, state: Optional[int] = None) -> Tuple[int, int, int]:
        """تحديد لون الجندي بناءً على الفريق والحالة"""
        base_color = self.get_team_color(team)
        
        if state == self.STATE_ATTACKING:
            return tuple(min(255, int(c * 1.3)) for c in base_color)
        
        return base_color
    
    def draw_deployment_zones(self, zones: dict):
        """رسم مناطق النشر"""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for team_id, zone in zones.items():
            x, y, w, h = zone
            color = self.get_team_color(int(team_id))
            zone_color = (*color, 50)  # شفافية
            pygame.draw.rect(surface, zone_color, (x, y, w, h))
            pygame.draw.rect(surface, color, (x, y, w, h), 2)  # إطار
        
        self.screen.blit(surface, (0, 0))
    
    def draw_start_button(self, battle_started: bool):
        """رسم زر بدء المعركة"""
        if battle_started:
            return  # لا نعرض الزر بعد بدء المعركة
        
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.start_button_rect.collidepoint(mouse_pos)
        
        # تحديد اللون
        color = self.button_hover_color if is_hover else self.button_color
        
        # رسم الزر
        pygame.draw.rect(self.screen, color, self.start_button_rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), self.start_button_rect, 3, border_radius=8)
        
        # رسم النص
        text = self.button_font.render("بدء المعركة", True, (255, 255, 255))
        text_rect = text.get_rect(center=self.start_button_rect.center)
        self.screen.blit(text, text_rect)
    
    def render(self):
        """رسم حالة اللعبة"""
        self.screen.fill(self.COLOR_BG)
        
        with self.state_lock:
            state = self.world_state
        
        if state:
            battle_started = state.get('battle_started', False)

            # رسم مناطق النشر إذا لم تبدأ المعركة
            if not battle_started:
                zones = state.get('deployment_zones', {})
                if zones:
                    self.draw_deployment_zones(zones)
            
            # رسم العوائق
            for obstacle in state.get('obstacles', []):
                x, y, w, h = obstacle
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (x, y, w, h))
            
            # رسم الجنود
            positions = state.get('positions', [])
            teams = state.get('teams', [])
            healths = state.get('healths', [])
            states_list = state.get('states', [])
            unit_ids = state.get('unit_ids', [])
            
            for i in range(len(positions)):
                pos = positions[i]
                team = teams[i] if i < len(teams) else 0
                health = healths[i] if i < len(healths) else 100
                soldier_state = states_list[i] if i < len(states_list) else self.STATE_IDLE
                unit_id = unit_ids[i] if i < len(unit_ids) else -1
                
                x, y = pos
                
                # تحديد لون الجندي
                color = self.get_soldier_color(team, soldier_state)
                
                # حجم الجندي (أكبر إذا كان مختاراً)
                soldier_size = 10 if unit_id in self.selected_units else 8
                
                # رسم الجندي
                pygame.draw.rect(
                    self.screen, color,
                    (x - soldier_size // 2, y - soldier_size // 2, soldier_size, soldier_size)
                )
                
                # رسم حلقة للوحدات المختارة
                if unit_id in self.selected_units:
                    pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), soldier_size + 2, 1)
                
                # رسم شريط الصحة
                self.draw_health_bar(x, y, health)
            
            # رسم زر بدء المعركة
            self.draw_start_button(battle_started)
        
        # رسم صندوق الاختيار
        if self.selecting and self.selection_box_start:
            mouse_pos = pygame.mouse.get_pos()
            x1, y1 = self.selection_box_start
            x2, y2 = mouse_pos
            
            rect = pygame.Rect(
                min(x1, x2), min(y1, y2),
                abs(x2 - x1), abs(y2 - y1)
            )
            
            surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.rect(surface, self.COLOR_SELECTION, rect)
            pygame.draw.rect(surface, (255, 255, 255), rect, 2)
            self.screen.blit(surface, (0, 0))
        
        # رسم خط التشكيل
        if self.dragging and self.drag_start:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, self.COLOR_DRAG_LINE, self.drag_start, mouse_pos, 3)
            pygame.draw.circle(self.screen, self.COLOR_DRAG_LINE, self.drag_start, 5)
            pygame.draw.circle(self.screen, self.COLOR_DRAG_LINE, mouse_pos, 5)
        
        # معلومات الشاشة
        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        if state:
            num_soldiers = len(state.get('positions', []))
            num_selected = len(self.selected_units)
            
            soldiers_text = self.small_font.render(f"solders: {num_soldiers}", True, (200, 200, 200))
            self.screen.blit(soldiers_text, (10, 35))
            
            if num_selected > 0:
                selected_text = self.small_font.render(f"choosen: {num_selected} unit", True, (255, 255, 100))
                self.screen.blit(selected_text, (10, 55))
            
            battle_status = "Battle Started" if state.get('battle_started', False) else "Deployment Phase"
            status_color = (255, 100, 100) if state.get('battle_started', False) else (100, 200, 255)
            status_text = self.small_font.render(battle_status, True, status_color)
            self.screen.blit(status_text, (10, 75))
        
        if self.battle_report:
            # خلفية نصف شفافة
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            # عنوان التقرير
            title = self.button_font.render(" Battle Report", True, (255, 215, 0))
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 50))


            # الفائز
            winner = self.battle_report.get("winner")
            if winner == -1:
                winner_text = " Result: Draw"
            else:
                winner_text = f" Winner: Team {winner}"

            winner_surface = self.font.render(winner_text, True, (255, 255, 255))
            self.screen.blit(winner_surface, (self.width // 2 - winner_surface.get_width() // 2, 100))

            # تفاصيل الفرق
            y_offset = 150
            winner = self.battle_report.get("winner")  # قد يكون -1 أو "allies" / "enemies" أو رقم الفريق

            # نحدد أي الفرق هي الفائزة (كمجموعة أرقام فرق) لتلوينها بسهولة
            winning_team_ids = set()
            draw = False

            if winner == -1:
                draw = True
            elif isinstance(winner, str) and winner in ("allies", "enemies"):
                if winner == "allies":
                    # Player(0) + Ally(2) هي التحالف الحليف
                    winning_team_ids = {0, 2}
                else:  # "enemies"
                    winning_team_ids = {1}
            else:
                # حاول تحويل الفائز إلى رقم (قد يأتي كـ string من JSON)
                try:
                    wnum = int(winner)
                    winning_team_ids = {wnum}
                except Exception:
                    # لو شيء غريب، نترك المجموعة فارغة (عرض عادي)
                    winning_team_ids = set()

            # دالة مساعدة لإعطاء اسم الفريق من الـ id
            def team_name(tid: int) -> str:
                if tid == 0:
                    return "Player"
                elif tid == 1:
                    return "Enemy"
                elif tid == 2:
                    return "Ally"
                return f"Team {tid}"

            # عرض كل فريق ووحداته
            for team_id, team_data in self.battle_report.get("teams", {}).items():
                try:
                    tid_int = int(team_id)
                except Exception:
                    tid_int = team_id  # مجرد fallback

                # تحديد اللون بناءً على نتيجة المعركة
                if draw:
                    team_color = (200, 200, 200)  # تعادل = رمادي
                elif isinstance(tid_int, int) and tid_int in winning_team_ids:
                    team_color = (0, 200, 0)   # أخضر للفائز
                else:
                    team_color = (220, 50, 50) # أحمر للخاسر

                name = team_name(int(team_id)) if isinstance(tid_int, int) else f"Team {team_id}"
                team_text = f"{name} - Units: {team_data['total_units']} - Soldiers: {team_data['total_soldiers']}"
                team_surface = self.small_font.render(team_text, True, team_color)
                self.screen.blit(team_surface, (50, y_offset))
                y_offset += 25

                for uid, udata in team_data["units"].items():
                    unit_text = (
                        f"    Unit {uid}: "
                        f"{udata['soldiers_remaining']} soldiers, "
                        f"avg health {udata['average_health']}"
                    )
                    unit_surface = self.small_font.render(unit_text, True, team_color)
                    self.screen.blit(unit_surface, (80, y_offset))
                    y_offset += 20

            # فراغ صغير قبل ملخص التحالفات (إذا وُجد)
            y_offset += 10

            # عرض ملخص التحالفات (allies / enemies) إذا وُجد في التقرير
            alliances = self.battle_report.get("alliances", {})
            if alliances:
                alliances_title = self.small_font.render("Alliances summary:", True, (255, 255, 255))
                self.screen.blit(alliances_title, (50, y_offset))
                y_offset += 25

                for alliance_key in ("allies", "enemies"):
                    stats = alliances.get(alliance_key)
                    if not stats:
                        continue

                    # اختر لون العرض للتحالف بناءً على الفائز
                    if draw:
                        col = (180, 180, 180)
                    else:
                        if (isinstance(winner, str) and winner == alliance_key) or \
                           (isinstance(winner, (int, str)) and alliance_key == "allies" and \
                                (isinstance(winner, int) and winner in (0,2))):
                            col = (0, 200, 0)
                        else:
                            # إذا winner رقمى ومطابق للعدو، أو winner نصي يختلف -> احمر
                            # أبسط قاعدة: لو الفائز هو التحالف الحالي نجعله أخضر، وإلا أحمر
                            col = (0, 200, 0) if (isinstance(winner, str) and winner == alliance_key) else (220, 50, 50)

                    title = f"{alliance_key.capitalize()}: Units {stats['total_units']} | Soldiers {stats['total_soldiers']} | Avg HP {stats['average_health']}"
                    surf = self.small_font.render(title, True, col)
                    self.screen.blit(surf, (60, y_offset))
                    y_offset += 22


        # تعليمات
        instructions = [
            "Left Click: Select a unit",
            "Left Drag: Multi-select units",
            "Shift + Click: Add to selection",
            "Right Click: Move selected",
            "Right Drag: Form selected",
        ]
        y_offset = self.height - 130
        for instruction in instructions:
            text = self.small_font.render(instruction, True, (180, 180, 180))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20
        
        pygame.display.flip()
    
    def run_network(self):
        """تشغيل حلقة الشبكة"""
        self.network_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.network_loop)
        self.network_loop.run_until_complete(self.connect_to_server())
        self.network_loop.close()
    
    def run(self):
        """تشغيل العميل"""
        network_thread = threading.Thread(target=self.run_network, daemon=True)
        network_thread.start()
        
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(60)
        
        pygame.quit()


if __name__ == "__main__":
    client = GameClient()
    try:
        client.run()
    except KeyboardInterrupt:
        print("\nإيقاف العميل...")
    except Exception as e:
        print(f"خطأ: {e}")
        import traceback
        traceback.print_exc()