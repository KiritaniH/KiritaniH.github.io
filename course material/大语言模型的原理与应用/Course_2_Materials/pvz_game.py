import pygame
import random
import math

# 初始化
pygame.init()

# 常量
SCREEN_WIDTH = 900
SCREEN_HEIGHT = 600
GRID_ROWS = 5
GRID_COLS = 9
CELL_WIDTH = 80
CELL_HEIGHT = 100
GRID_START_X = 100
GRID_START_Y = 100

# 颜色
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
YELLOW = (255, 255, 0)
GOLD = (255, 215, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GREEN = (144, 238, 144)
DARK_GRAY = (64, 64, 64)
PEA_GREEN = (0, 200, 0)
ZOMBIE_GREEN = (100, 140, 100)
RED_BROWN = (101, 67, 33)
ORANGE_RED = (255, 69, 0)
SUN_YELLOW = (255, 255, 100)  # 亮黄色，与向日葵区分

# 游戏常量
SUN_VALUE = 25
PEASHOOTER_COST = 100
SUNFLOWER_COST = 50
POTATO_MINE_COST = 25
PEA_DAMAGE = 20
ZOMBIE_HEALTH = 100
ZOMBIE_DAMAGE = 1  # 每帧伤害
PEA_SPEED = 7
ZOMBIE_SPEED = 0.3
SUN_SPEED = 2
POTATO_MINE_DAMAGE = 200  # 秒杀僵尸

# 植物冷却时间（帧数，60 帧=1 秒）
PEASHOOTER_COOLDOWN = 300  # 5 秒
SUNFLOWER_COOLDOWN = 300   # 5 秒
POTATO_MINE_COOLDOWN = 900 # 15 秒

# 僵尸生成间隔（帧数）
ZOMBIE_SPAWN_INTERVAL_START = 720  # 初始 12 秒
ZOMBIE_SPAWN_INTERVAL_END = 300    # 最终 5 秒

class Sun:
    def __init__(self, x, y, value=SUN_VALUE, from_sky=False):
        self.x = x
        self.y = y
        self.target_y = y if from_sky else random.randint(200, 400)
        self.value = value
        self.radius = 25
        self.from_sky = from_sky
        self.collected = False
        self.disappear_timer = 500  # 5 秒后消失

    def update(self):
        if self.from_sky and self.y < self.target_y:
            self.y += SUN_SPEED
        self.disappear_timer -= 1
        return self.disappear_timer > 0

    def draw(self, screen):
        # 绘制阳光
        pygame.draw.circle(screen, SUN_YELLOW, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.radius - 5)
        # 光芒
        for i in range(8):
            angle = i * math.pi / 4
            end_x = self.x + math.cos(angle) * (self.radius + 8)
            end_y = self.y + math.sin(angle) * (self.radius + 8)
            pygame.draw.line(screen, WHITE, (self.x, self.y), (end_x, end_y), 3)
        # 显示数值
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.value), True, BLACK)
        screen.blit(text, (self.x - 8, self.y - 8))

    def check_click(self, pos):
        dist = math.sqrt((pos[0] - self.x)**2 + (pos[1] - self.y)**2)
        return dist < self.radius

class Pea:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = PEA_SPEED
        self.damage = PEA_DAMAGE
        self.radius = 8
        self.active = True

    def update(self):
        self.x += self.speed
        if self.x > SCREEN_WIDTH:
            self.active = False
        return self.active

    def draw(self, screen):
        pygame.draw.circle(screen, PEA_GREEN, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, GREEN, (int(self.x), int(self.y)), self.radius - 2)

    def check_collision(self, zombie):
        dist = math.sqrt((self.x - zombie.x)**2 + (self.y - zombie.y)**2)
        return dist < self.radius + zombie.width // 2

class Plant:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = GRID_START_X + col * CELL_WIDTH + CELL_WIDTH // 2
        self.y = GRID_START_Y + row * CELL_HEIGHT + CELL_HEIGHT // 2
        self.health = 100
        self.max_health = 100

    def draw_health_bar(self, screen):
        if self.health < self.max_health:
            bar_width = 40
            bar_height = 5
            x = self.x - bar_width // 2
            y = self.y - 30
            pygame.draw.rect(screen, RED, (x, y, bar_width, bar_height))
            health_width = int(bar_width * self.health / self.max_health)
            pygame.draw.rect(screen, GREEN, (x, y, health_width, bar_height))

class PotatoMine(Plant):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.armed = False
        self.arm_timer = 0
        self.arm_interval = 300  # 5 秒后武装完成

    def update(self):
        self.arm_timer += 1
        if self.arm_timer >= self.arm_interval:
            self.armed = True
        return self.armed

    def draw(self, screen):
        # 主体（土豆）
        pygame.draw.ellipse(screen, RED_BROWN, (self.x - 20, self.y - 10, 40, 30))
        # 未武装时颜色较暗
        if not self.armed:
            pygame.draw.ellipse(screen, BROWN, (self.x - 15, self.y - 8, 30, 22))
            # 闪烁提示未就绪
            if (self.arm_timer // 10) % 2 == 0:
                pygame.draw.ellipse(screen, RED_BROWN, (self.x - 18, self.y - 12, 36, 34), 2)
        else:
            # 武装完成，红色外壳
            pygame.draw.ellipse(screen, ORANGE_RED, (self.x - 15, self.y - 8, 30, 22))
            # 闪烁红光提示就绪
            if (self.arm_timer // 15) % 2 == 0:
                pygame.draw.circle(screen, RED, (self.x, self.y), 8)
        # 眼睛
        pygame.draw.circle(screen, BLACK, (self.x - 8, self.y - 5), 3)
        pygame.draw.circle(screen, BLACK, (self.x + 8, self.y - 5), 3)
        # 天线
        pygame.draw.line(screen, RED, (self.x, self.y - 10), (self.x, self.y - 20), 2)
        pygame.draw.circle(screen, ORANGE_RED, (self.x, self.y - 22), 4)

        self.draw_health_bar(screen)

class Peashooter(Plant):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.shoot_timer = 0
        self.shoot_interval = 90  # 1.5 秒发射一次

    def update(self):
        self.shoot_timer += 1
        if self.shoot_timer >= self.shoot_interval:
            self.shoot_timer = 0
            return True  # 可以发射
        return False

    def draw(self, screen):
        # 茎
        pygame.draw.rect(screen, GREEN, (self.x - 10, self.y - 20, 20, 40))
        # 头
        pygame.draw.circle(screen, GREEN, (self.x, self.y - 25), 25)
        pygame.draw.circle(screen, LIGHT_GREEN, (self.x, self.y - 25), 20)
        # 嘴
        pygame.draw.circle(screen, DARK_GREEN, (self.x + 15, self.y - 25), 10)
        # 眼睛
        pygame.draw.circle(screen, BLACK, (self.x - 8, self.y - 30), 5)
        pygame.draw.circle(screen, WHITE, (self.x - 10, self.y - 32), 2)

        self.draw_health_bar(screen)

class Sunflower(Plant):
    def __init__(self, row, col):
        super().__init__(row, col)
        self.produce_timer = 0
        self.produce_interval = 600  # 10 秒产生一次阳光

    def update(self):
        self.produce_timer += 1
        if self.produce_timer >= self.produce_interval:
            self.produce_timer = 0
            return True  # 可以产生阳光
        return False

    def draw(self, screen):
        # 茎
        pygame.draw.rect(screen, GREEN, (self.x - 8, self.y - 10, 16, 30))
        # 花盘
        pygame.draw.circle(screen, GOLD, (self.x, self.y - 15), 25)
        pygame.draw.circle(screen, YELLOW, (self.x, self.y - 15), 20)
        # 花瓣
        for i in range(8):
            angle = i * math.pi / 4
            px = self.x + math.cos(angle) * 30
            py = self.y - 15 + math.sin(angle) * 30
            pygame.draw.circle(screen, YELLOW, (int(px), int(py)), 8)
        # 笑脸
        pygame.draw.circle(screen, BLACK, (self.x - 8, self.y - 20), 3)
        pygame.draw.circle(screen, BLACK, (self.x + 8, self.y - 20), 3)
        pygame.draw.arc(screen, BLACK, (self.x - 10, self.y - 18, 20, 15), 0.2, 2.9, 2)

        self.draw_health_bar(screen)

class Zombie:
    def __init__(self, row):
        self.row = row
        self.x = SCREEN_WIDTH - 50
        self.y = GRID_START_Y + row * CELL_HEIGHT + CELL_HEIGHT // 2
        self.width = 40
        self.height = 70
        self.health = ZOMBIE_HEALTH
        self.max_health = ZOMBIE_HEALTH
        self.speed = ZOMBIE_SPEED
        self.damage = ZOMBIE_DAMAGE
        self.eating = False
        self.eat_timer = 0
        self.walk_timer = 0
        self.frozen = False

    def update(self, plants):
        self.eating = False
        self.eat_timer += 1

        # 检查是否有植物在同一行且相邻
        for plant in plants[:]:  # 使用副本以便安全删除
            if plant.row == self.row:
                plant_x = GRID_START_X + plant.col * CELL_WIDTH
                if plant_x <= self.x <= plant_x + CELL_WIDTH:
                    # 检查是否是土豆地雷
                    if isinstance(plant, PotatoMine) and plant.armed:
                        # 土豆地雷立即爆炸，秒杀僵尸
                        self.health -= POTATO_MINE_DAMAGE
                        plants.remove(plant)
                        return "exploding"
                    else:
                        self.eating = True
                        plant.health -= self.damage
                        return "eating"

        self.x -= self.speed
        return "walking"

    def draw(self, screen):
        # 身体
        pygame.draw.rect(screen, ZOMBIE_GREEN, (self.x - 20, self.y - 35, 40, 70))
        pygame.draw.rect(screen, DARK_GRAY, (self.x - 20, self.y - 10, 40, 35))  # 衣服
        # 头
        pygame.draw.circle(screen, ZOMBIE_GREEN, (int(self.x), int(self.y - 45)), 22)
        # 眼睛
        pygame.draw.circle(screen, WHITE, (int(self.x - 8), int(self.y - 50)), 6)
        pygame.draw.circle(screen, WHITE, (int(self.x + 8), int(self.y - 50)), 6)
        pygame.draw.circle(screen, RED, (int(self.x - 8), int(self.y - 50)), 2)
        pygame.draw.circle(screen, RED, (int(self.x + 8), int(self.y - 50)), 2)
        # 嘴巴
        pygame.draw.rect(screen, BLACK, (self.x - 10, self.y - 35, 20, 5))
        # 手臂
        if self.eating:
            pygame.draw.line(screen, ZOMBIE_GREEN, (self.x - 20, self.y - 25), (self.x - 40, self.y - 25), 8)
        else:
            # 行走动画
            arm_offset = math.sin(self.eat_timer * 0.1) * 10
            pygame.draw.line(screen, ZOMBIE_GREEN, (self.x - 20, self.y - 25),
                           (self.x - 35, self.y - 25 + arm_offset), 8)
        # 腿
        leg_offset = math.sin(self.eat_timer * 0.1) * 10
        pygame.draw.line(screen, DARK_GRAY, (self.x - 10, self.y + 35),
                        (self.x - 10, self.y + 50 + leg_offset), 8)
        pygame.draw.line(screen, DARK_GRAY, (self.x + 10, self.y + 35),
                        (self.x + 10, self.y + 50 - leg_offset), 8)

        # 血条
        bar_width = 40
        bar_height = 5
        pygame.draw.rect(screen, RED, (self.x - 20, self.y - 65, bar_width, bar_height))
        health_width = int(bar_width * self.health / self.max_health)
        pygame.draw.rect(screen, GREEN, (self.x - 20, self.y - 65, health_width, bar_height))

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("植物大战僵尸 - 简化版")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)

        self.reset_game()

    def reset_game(self):
        self.plants = []
        self.zombies = []
        self.peas = []
        self.suns = []
        self.sun_count = 150  # 初始阳光
        self.selected_plant = None  # 当前选择的植物类型
        self.game_over = False
        self.win = False
        self.paused = False  # 游戏暂停状态
        self.frame_count = 0
        self.zombies_killed = 0
        self.zombies_to_spawn = 20  # 总共需要消灭的僵尸数
        self.spawned_zombies = 0
        self.last_spawn_row = -1  # 上次生成僵尸的行
        # 每种植物的独立冷却时间
        self.peashooter_cooldown = 0
        self.sunflower_cooldown = 0
        self.potato_mine_cooldown = 0
        # 僵尸生成间隔（动态变化）
        self.zombie_spawn_interval = ZOMBIE_SPAWN_INTERVAL_START
        self.last_zombie_spawn_frame = 0

    def draw_grid(self):
        # 绘制草坪背景
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x = GRID_START_X + col * CELL_WIDTH
                y = GRID_START_Y + row * CELL_HEIGHT
                color = GREEN if (row + col) % 2 == 0 else LIGHT_GREEN
                pygame.draw.rect(self.screen, color, (x, y, CELL_WIDTH, CELL_HEIGHT))
                pygame.draw.rect(self.screen, DARK_GREEN, (x, y, CELL_WIDTH, CELL_HEIGHT), 1)

    def draw_ui(self):
        # 顶部面板
        pygame.draw.rect(self.screen, BROWN, (0, 0, SCREEN_WIDTH, 80))
        pygame.draw.rect(self.screen, LIGHT_BROWN, (0, 0, SCREEN_WIDTH, 80), 3)

        # 阳光显示
        pygame.draw.circle(self.screen, GOLD, (50, 40), 20)
        sun_text = self.font.render(str(self.sun_count), True, WHITE)
        self.screen.blit(sun_text, (75, 25))

        # 植物选择卡片
        # 豌豆射手卡片
        ps_color = GREEN if self.selected_plant == "peashooter" else GRAY
        pygame.draw.rect(self.screen, ps_color, (150, 10, 60, 60))
        pygame.draw.rect(self.screen, WHITE, (150, 10, 60, 60), 2)
        # 绘制豌豆射手图标
        pygame.draw.circle(self.screen, LIGHT_GREEN, (180, 40), 15)
        ps_cost = self.font.render(str(PEASHOOTER_COST), True, WHITE)
        self.screen.blit(ps_cost, (155, 55))
        # 冷却显示
        if self.peashooter_cooldown > 0:
            cooldown_text = self.font.render(str((self.peashooter_cooldown + 59) // 60), True, RED)
            self.screen.blit(cooldown_text, (195, 15))

        # 向日葵卡片
        sf_color = GOLD if self.selected_plant == "sunflower" else GRAY
        pygame.draw.rect(self.screen, sf_color, (230, 10, 60, 60))
        pygame.draw.rect(self.screen, WHITE, (230, 10, 60, 60), 2)
        # 绘制向日葵图标
        pygame.draw.circle(self.screen, YELLOW, (260, 40), 15)
        sf_cost = self.font.render(str(SUNFLOWER_COST), True, WHITE)
        self.screen.blit(sf_cost, (235, 55))
        # 冷却显示
        if self.sunflower_cooldown > 0:
            cooldown_text = self.font.render(str((self.sunflower_cooldown + 59) // 60), True, RED)
            self.screen.blit(cooldown_text, (275, 15))

        # 土豆地雷卡片
        pm_color = ORANGE_RED if self.selected_plant == "potato_mine" else GRAY
        pygame.draw.rect(self.screen, pm_color, (310, 10, 60, 60))
        pygame.draw.rect(self.screen, WHITE, (310, 10, 60, 60), 2)
        # 绘制土豆地雷图标
        pygame.draw.ellipse(self.screen, RED_BROWN, (325, 25, 30, 20))
        pygame.draw.ellipse(self.screen, ORANGE_RED, (330, 28, 20, 14))
        pm_cost = self.font.render(str(POTATO_MINE_COST), True, WHITE)
        self.screen.blit(pm_cost, (315, 55))
        # 冷却显示
        if self.potato_mine_cooldown > 0:
            cooldown_text = self.font.render(str((self.potato_mine_cooldown + 59) // 60), True, RED)
            self.screen.blit(cooldown_text, (355, 15))

        # 铲子
        shovel_color = (139, 139, 139) if self.selected_plant != "shovel" else (100, 100, 100)
        pygame.draw.rect(self.screen, shovel_color, (390, 10, 60, 60))
        pygame.draw.rect(self.screen, WHITE, (390, 10, 60, 60), 2)
        # 绘制铲子图标
        # 铲柄
        pygame.draw.rect(self.screen, BROWN, (415, 25, 8, 30))
        # 铲头
        pygame.draw.polygon(self.screen, (180, 180, 180), [(410, 55), (430, 55), (420, 45)])
        shovel_cost = self.font.render("25", True, WHITE)
        self.screen.blit(shovel_cost, (405, 55))

        # 僵尸进度
        progress_text = self.font.render(f"僵尸：{self.zombies_killed}/{self.zombies_to_spawn}", True, WHITE)
        self.screen.blit(progress_text, (SCREEN_WIDTH - 200, 25))

        # 提示信息
        if self.paused:
            hint_text = self.font.render("游戏暂停 - 按 P 继续", True, YELLOW)
        else:
            hint_text = self.font.render("点击卡片选择植物，点击草坪种植", True, WHITE)
        self.screen.blit(hint_text, (400, 25))

        # 暂停按钮
        pause_text = self.font.render("暂停 (P)", True, WHITE)
        self.screen.blit(pause_text, (SCREEN_WIDTH - 100, 25))

    def get_grid_pos(self, pos):
        x, y = pos
        if x < GRID_START_X or x > GRID_START_X + GRID_COLS * CELL_WIDTH:
            return None, None
        if y < GRID_START_Y or y > GRID_START_Y + GRID_ROWS * CELL_HEIGHT:
            return None, None
        col = (x - GRID_START_X) // CELL_WIDTH
        row = (y - GRID_START_Y) // CELL_HEIGHT
        return row, col

    def can_place_plant(self, row, col):
        for plant in self.plants:
            if plant.row == row and plant.col == col:
                return False
        return True

    def get_plant_at(self, row, col):
        for plant in self.plants:
            if plant.row == row and plant.col == col:
                return plant
        return None

    def spawn_zombie(self):
        if self.spawned_zombies < self.zombies_to_spawn:
            # 最后一波（第 20 个）僵尸可以在任意行生成
            is_last_wave = (self.spawned_zombies == self.zombies_to_spawn - 1)
            if is_last_wave:
                row = random.randint(0, GRID_ROWS - 1)
            else:
                # 不允许连续在同一行生成
                available_rows = [r for r in range(GRID_ROWS) if r != self.last_spawn_row]
                row = random.choice(available_rows)
            self.zombies.append(Zombie(row))
            self.last_spawn_row = row
            self.spawned_zombies += 1

    def spawn_sun_from_sky(self):
        x = random.randint(GRID_START_X + 50, SCREEN_WIDTH - 50)
        self.suns.append(Sun(x, -30, from_sky=True))

    def check_pea_collision(self):
        for pea in self.peas:
            for zombie in self.zombies:
                if pea.check_collision(zombie):
                    zombie.health -= pea.damage
                    pea.active = False
                    break

    def check_game_over(self):
        for zombie in self.zombies:
            if zombie.x < GRID_START_X - 50:
                self.game_over = True
                return

        if self.zombies_killed >= self.zombies_to_spawn:
            self.win = True
            self.game_over = True

    def handle_click(self, pos):
        x, y = pos

        # 检查是否点击 UI 区域
        if y < 80:
            # 豌豆射手卡片
            if 150 <= x <= 210:
                if self.sun_count >= PEASHOOTER_COST:
                    self.selected_plant = "peashooter"
                return
            # 向日葵卡片
            if 230 <= x <= 290:
                if self.sun_count >= SUNFLOWER_COST:
                    self.selected_plant = "sunflower"
                return
            # 土豆地雷卡片
            if 310 <= x <= 370:
                if self.sun_count >= POTATO_MINE_COST:
                    self.selected_plant = "potato_mine"
                return
            # 铲子
            if 390 <= x <= 450:
                self.selected_plant = "shovel"
                return
            return

        # 检查是否点击阳光
        for sun in self.suns[:]:
            if sun.check_click(pos):
                sun.collected = True
                self.sun_count += sun.value
                self.suns.remove(sun)
                return

        # 检查是否点击网格
        row, col = self.get_grid_pos(pos)
        if row is not None and not self.paused:
            if self.selected_plant == "shovel":
                plant = self.get_plant_at(row, col)
                if plant:
                    self.plants.remove(plant)
                    self.sun_count += 25  # 铲子返还 25 阳光，不消耗
                self.selected_plant = None
            elif self.selected_plant == "peashooter" and self.peashooter_cooldown <= 0:
                if self.can_place_plant(row, col) and self.sun_count >= PEASHOOTER_COST:
                    self.plants.append(Peashooter(row, col))
                    self.sun_count -= PEASHOOTER_COST
                    self.peashooter_cooldown = PEASHOOTER_COOLDOWN
                    self.selected_plant = None
            elif self.selected_plant == "sunflower" and self.sunflower_cooldown <= 0:
                if self.can_place_plant(row, col) and self.sun_count >= SUNFLOWER_COST:
                    self.plants.append(Sunflower(row, col))
                    self.sun_count -= SUNFLOWER_COST
                    self.sunflower_cooldown = SUNFLOWER_COOLDOWN
                    self.selected_plant = None
            elif self.selected_plant == "potato_mine" and self.potato_mine_cooldown <= 0:
                if self.can_place_plant(row, col) and self.sun_count >= POTATO_MINE_COST:
                    self.plants.append(PotatoMine(row, col))
                    self.sun_count -= POTATO_MINE_COST
                    self.potato_mine_cooldown = POTATO_MINE_COOLDOWN
                    self.selected_plant = None

    def update(self):
        if self.game_over or self.paused:
            return

        self.frame_count += 1

        # 植物冷却时间递减
        if self.peashooter_cooldown > 0:
            self.peashooter_cooldown -= 1
        if self.sunflower_cooldown > 0:
            self.sunflower_cooldown -= 1
        if self.potato_mine_cooldown > 0:
            self.potato_mine_cooldown -= 1

        # 计算动态僵尸生成间隔（使用对数衰减）
        # 从 12 秒 (720 帧) 逐渐减少到 5 秒 (300 帧)
        if self.spawned_zombies > 0:
            progress = self.spawned_zombies / self.zombies_to_spawn  # 0 到 1
            # 使用对数函数实现先慢后快的衰减
            decay_factor = math.log(1 + 9 * progress) / math.log(10)  # 0 到 1
            self.zombie_spawn_interval = int(
                ZOMBIE_SPAWN_INTERVAL_START -
                (ZOMBIE_SPAWN_INTERVAL_START - ZOMBIE_SPAWN_INTERVAL_END) * decay_factor
            )

        # 生成僵尸
        if self.frame_count - self.last_zombie_spawn_frame >= self.zombie_spawn_interval:
            self.spawn_zombie()
            self.last_zombie_spawn_frame = self.frame_count

        # 生成天空阳光（约 7.5 秒一次）
        if self.frame_count % 450 == 0 and self.spawned_zombies > 0:
            self.spawn_sun_from_sky()

        # 更新植物
        for plant in self.plants[:]:
            if isinstance(plant, Peashooter):
                if plant.update():
                    # 检查该行是否有僵尸
                    has_zombie = any(z.row == plant.row and z.x > plant.x for z in self.zombies)
                    if has_zombie:
                        self.peas.append(Pea(plant.x + 20, plant.y - 25))
            elif isinstance(plant, Sunflower):
                if plant.update():
                    self.suns.append(Sun(plant.x, plant.y - 30))
            elif isinstance(plant, PotatoMine):
                plant.update()  # 只更新计时器，爆炸逻辑在僵尸更新中处理

            if plant.health <= 0:
                self.plants.remove(plant)

        # 更新豌豆
        for pea in self.peas[:]:
            if not pea.update():
                self.peas.remove(pea)

        # 更新僵尸
        for zombie in self.zombies[:]:
            zombie.update(self.plants)
            if zombie.health <= 0:
                self.zombies.remove(zombie)
                self.zombies_killed += 1

        # 更新阳光
        for sun in self.suns[:]:
            if not sun.update():
                self.suns.remove(sun)

        # 检查碰撞
        self.check_pea_collision()

        # 检查游戏结束
        self.check_game_over()

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_grid()

        # 绘制植物
        for plant in self.plants:
            plant.draw(self.screen)

        # 绘制僵尸
        for zombie in self.zombies:
            zombie.draw(self.screen)

        # 绘制豌豆
        for pea in self.peas:
            pea.draw(self.screen)

        # 绘制阳光
        for sun in self.suns:
            sun.draw(self.screen)

        self.draw_ui()

        # 游戏暂停画面
        if self.paused:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            pause_text = self.big_font.render("游戏暂停", True, WHITE)
            pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            self.screen.blit(pause_text, pause_rect)
            resume_text = self.font.render("按 P 继续游戏", True, WHITE)
            resume_rect = resume_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
            self.screen.blit(resume_text, resume_rect)

        # 游戏结束画面
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))

            if self.win:
                text = self.big_font.render("胜利!", True, GOLD)
            else:
                text = self.big_font.render("游戏结束!", True, RED)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(text, text_rect)

            restart_text = self.font.render("按 R 重新开始，按 Q 退出", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键
                        if self.game_over:
                            continue
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:  # P 键暂停/继续
                        if self.game_over:
                            continue
                        self.paused = not self.paused
                    elif self.game_over:
                        if event.key == pygame.K_r:
                            self.reset_game()
                        elif event.key == pygame.K_q:
                            running = False
                    elif not self.paused:
                        if event.key == pygame.K_1:
                            self.selected_plant = "peashooter"
                        elif event.key == pygame.K_2:
                            self.selected_plant = "sunflower"
                        elif event.key == pygame.K_3:
                            self.selected_plant = "potato_mine"

            self.update()
            self.draw()

        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
