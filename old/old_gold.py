
    def spawn_unit(self, team_id: int, num_soldiers: int, start_position: Tuple[float, float]) -> int:
        """إنشاء وحدة جديدة من الجنود"""
        unit_id = self.next_unit_id
        self.next_unit_id += 1
        
        # إنشاء تشكيل شبكي بسيط للجنود
        cols = int(np.sqrt(num_soldiers))
        rows = (num_soldiers + cols - 1) // cols
        spacing = 15
        
        new_positions = []
        for i in range(num_soldiers):
            row = i // cols
            col = i % cols
            x = start_position[0] + col * spacing
            y = start_position[1] + row * spacing
            new_positions.append([x, y])
        
        # إضافة الجنود إلى المصفوفات
        new_positions = np.array(new_positions, dtype=np.float32)
        new_teams = np.full(num_soldiers, team_id, dtype=np.int32)
        new_unit_ids = np.full(num_soldiers, unit_id, dtype=np.int32)
        new_targets = new_positions.copy()  # البداية في مكانهم
        
        self.positions = np.vstack([self.positions, new_positions]) if self.soldier_count > 0 else new_positions
        self.teams = np.concatenate([self.teams, new_teams]) if self.soldier_count > 0 else new_teams
        self.unit_ids = np.concatenate([self.unit_ids, new_unit_ids]) if self.soldier_count > 0 else new_unit_ids
        self.target_positions = np.vstack([self.target_positions, new_targets]) if self.soldier_count > 0 else new_targets
        
        self.soldier_count += num_soldiers
        return unit_id
    


    def update(self, delta_time: float):
        """تحديث منطق اللعبة"""
        if self.soldier_count == 0:
            return
        
        # نظام الحركة - تحريك الجنود نحو أهدافهم بناءً على حالتهم
        for state_type in [self.STATE_MOVING, self.STATE_ATTACKING]:
            state_mask = self.states == state_type
            
            if not np.any(state_mask):
                continue
            
            # حساب الفرق بين الموقع الحالي والهدف
            diff = self.target_positions[state_mask] - self.positions[state_mask]
            distances = np.linalg.norm(diff, axis=1, keepdims=True)
            
            # تجنب القسمة على صفر
            distances = np.maximum(distances, 0.001)
            
            # حساب الحركة
            max_move = self.MOVEMENT_SPEED * delta_time
            move_amount = np.minimum(distances, max_move)
            movement = (diff / distances) * move_amount
            
            # تحديث المواقع المؤقتة
            temp_positions = self.positions.copy()
            temp_positions[state_mask] += movement
            
            # التحقق من وصول الجنود المتحركين إلى أهدافهم
            if state_type == self.STATE_MOVING:
                arrived_mask = state_mask & (distances[:, 0] < 5.0)  # عتبة الوصول 5 وحدات
                self.states[arrived_mask] = self.STATE_IDLE
        
        # حساب جميع الحركات
        diff = self.target_positions - self.positions
        distances = np.linalg.norm(diff, axis=1, keepdims=True)
        distances = np.maximum(distances, 0.001)
        
        max_move = self.MOVEMENT_SPEED * delta_time
        move_amount = np.minimum(distances, max_move)
        movements = (diff / distances) * move_amount
        
        # تطبيق قوة تجنب التداخل
        movements = self.apply_separation_force(movements)
        
        # تطبيق الحركة
        self.positions += movements
        
        # الحفاظ على الجنود داخل الحدود
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height)
        
        # نظام الكشف عن الأهداف
        self.update_enemy_targets()
        
        # نظام القتال
        self.update_combat(delta_time)
        
        # إزالة الجنود الأموات
        self.remove_dead_soldiers()
