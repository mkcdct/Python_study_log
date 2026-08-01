import pygame
import random
import os # 파일/경로 관련 작업을 위해 필요.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# os.path.abspath(__file__) : 현재 실행중인 파일의 절대경로를 반환.
# os.path.dirname() : 그 경로에서 파일명을 떼고 폴더 경로만 추출
import math

# pygame 초기화
pygame.init()

# 화면 크기 (가로 800, 세로 600)
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("나의 첫 파이게임")

try:
    player_img = pygame.image.load("playerShip1_blue.png").convert_alpha() # 이미지를 현재 화면이 쓰고 있는 픽셀포맷과 같은 형식으로 변환.(투명도(알파 채널)는 유지하면서.)
    player_img = pygame.transform.scale(player_img, (50, 50)) # 이미지 크기 조정
except (pygame.error, FileNotFoundError) as e:
    print(f"플레이어 이미지 로드 실패: {e}")
    player_img = None

obstacle_imgs = []
obstacle_filenames = [
        "meteorBrown_big1.png",
        "meteorBrown_big2.png",
        "meteorGrey_big1.png",
        "meteorGrey_big2.png"]
obstacle_sizes = [30, 50, 70] # 장애물 사이즈

for filename in obstacle_filenames:
    try:
        img = pygame.image.load(os.path.join(BASE_DIR, filename)).convert_alpha()
        img = pygame.transform.scale(img, (50, 50))
        obstacle_imgs.append(img)
    except (pygame.error, FileNotFoundError) as e:
        print(f"장애물 이미지 로드 실패 ({filename}): {e}")
if not obstacle_imgs:
    obstacle_imgs = None # 전부 실패 시 기존처럼 사각형 대체
    # 아래와 다른 이유 : 장애물 이미지 하나만 실패해도 None값이 실행되므로, 별도로 처리해야.

try:
    powerup_img = pygame.image.load("powerupGreen_shield.png").convert_alpha()
    powerup_img = pygame.transform.scale(powerup_img, (30, 30))
except (pygame.error, FileNotFoundError) as e:
    print(f"파워업 이미지 로드 실패: {e}")
    powerup_img = None

try:
    background_img = pygame.image.load("backgroundimg_space.png").convert()
    background_img = pygame.transform.scale(background_img, (800, 600))
except (pygame.error, FileNotFoundError) as e:
    print(f"배경 이미지 로드 실패: {e}")
    background_img = None
pygame.mixer.init() # 소리 재생을 위한 초기화.
try:
    collision_sound = pygame.mixer.Sound("collision.wav")
except (pygame.error, FileNotFoundError) as e:
    print(f"사운드 로드 실패: {e}")
    collision_sound = None
try:
    gameover_sound = pygame.mixer.Sound("gameover.wav")
except (pygame.error, FileNotFoundError) as e:
    print(f"사운드 로드 실패: {e}")
    gameover_sound = None
try:
    pygame.mixer.music.load("bgm.mp3")
    pygame.mixer.music.play(-1)
except (pygame.error, FileNotFoundError) as e:
    print(f"배경음악 로드 실패: {e}")

STATE_START = "start"
STATE_PLAYING = "playing"
STATE_GAMEOVER = "gameover"

BEST_TIME_FILE = "best_time.txt" # 저장할 파일 이름을 상수로 빼둔 것.

def load_best_time():
    """게임 시작 시 저장된 최고기록을 불러옴."""
    if os.path.exists(BEST_TIME_FILE):
        try:
            with open(BEST_TIME_FILE, "r") as f: # 읽기모드로 엶. with를 쓰면 블록이 끝날 때 파일이 자동으로 닫힘.
                return float(f.read().strip())
        except (ValueError, OSError) as e:
            print(f"최고기록 읽기 실패: {e}")
            return 0.0
    return 0.0 # if os.path.exists(BEST_TIME_FILE) 조건이 False일 때 반환.

def save_best_time(new_best):
    """ 새 최고기록을 파일에 저장"""
    try:
        with open(BEST_TIME_FILE, "w") as f: # 쓰기모드로 엶.
            f.write(str(new_best)) # 파일에는 텍스트만 쓸 수 있음.
    except OSError as e:
        print(f"최고기록 저장 실패: {e}")

best_time = load_best_time()

# 장애물 추가
def create_obstacle():
    size = random.choice(obstacle_sizes)
    base_x = random.randint(0, 800 - size)
    rect = pygame.Rect(base_x, random.randint(-600, 0), size, size)

    if obstacle_imgs:
        chosen_img = random.choice(obstacle_imgs)
        img = pygame.transform.scale(chosen_img, (size, size))
    else:
        img = None

    return {"rect": rect, "img": img, "base_x": base_x, "phase": random.uniform(0, math.pi * 2), "amplitude": random.randint(30, 80), "hp" : 3, "active" : True, "respawn_at" : 0}
   # random.randint : 정수 랜덤, random.uniform : 실수 랜덤.

is_slowed = False
slow_until = 0
slow_factor = 0.5
slow_duration = 4000
is_respawning = False
respawn_until = 0

def reset_game():
    global player, score, game_state, obstacles, obstacle_speed, start_time, is_new_record, current_level, survival_time
    global powerup, invincible, invincible_until
    global is_respawning, respawn_until
    global is_slowed, slow_until
    global missiles, last_missile_time
    # global 사용함으로써 전역 변수임을 명시.
    player = pygame.Rect(400, 300, 50, 50)
    score = 3
    game_state = STATE_PLAYING
    obstacle_speed = 4
    obstacles = [create_obstacle() for _ in range(3)]
    start_time = pygame.time.get_ticks() # reset함수가 호출된 시점의 시각 값(고정 값).
    is_new_record = False
    current_level = 0
    powerup = None
    invincible = False
    invincible_until = 0 # 무적이 언제 끝나는지(tick 기준)
    is_respawning = False
    respawn_until = 0
    survival_time = 0 # 생존시간 초기화
    is_slowed = False
    slow_until = 0
    missiles = []
    last_missile_time = 0

reset_game() # 처음 시작할 때 한 번 호출

game_state = STATE_START # 처음엔 시작 화면 상태로 시작. 즉, reset 함수 호출하고, game_state를 START로 덮어씀.

powerup_types = ['invincible', 'life', 'slow']

def create_powerup(x, y):
    ptype = random.choice(powerup_types)
    return {'rect': pygame.Rect(x, y, 30, 30), 'type' : ptype}
 
# 폰트 객체 생성
font = pygame.font.SysFont(None, 50) # None : pygame이 알아서 폰트 선택.
#text_surface = font.render("Collision!", True, (255, 255, 255))


# 플레이어의 이동 속도
speed = 5

clock = pygame.time.Clock() # 프레임 속도 조절용

missile_speed = 10 # 미사일 속도
missile_cooldown = 400 # 밀리초 단위, 이 시간마다 자동으로 한 발씩 발사.


# 게임 루프 실행 여부
running = True
while running:
    # 창 닫기(x 버튼) 이벤트 감지
    for event in pygame.event.get(): # 1회성 이벤트에 대한.
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if game_state == STATE_START:
                if event.key == pygame.K_SPACE:
                    game_state = STATE_PLAYING
                    start_time = pygame.time.get_ticks()
            elif game_state == STATE_GAMEOVER:
                if event.key == pygame.K_r:
                    reset_game()
                    start_time = pygame.time.get_ticks()  
    
    if game_state == STATE_PLAYING:
        if is_respawning:
            if pygame.time.get_ticks() >= respawn_until:
                is_respawning = False
                start_time += 3000 # 멈춰있던 3초만큼 시작시간을 밀어서 생존시간에 영향
        else:
            keys = pygame.key.get_pressed() # 함수 호출하고, 그 결과값을 keys에 저장.
            if keys[pygame.K_LEFT]:
                player.x -= speed
            if keys[pygame.K_RIGHT]:
                player.x += speed
            if keys[pygame.K_UP]:
                player.y -= speed
            if keys[pygame.K_DOWN]:
                player.y += speed

            # 미사일 자동 발사
            now_ticks = pygame.time.get_ticks()
            if now_ticks - last_missile_time >= missile_cooldown:
                missiles.append(pygame.Rect(player.centerx -3, player.top - 15, 6, 15))
                last_missile_time = now_ticks

            # 미사일 이동 및 화면 밖으로 나간 미사일 제거
            for missile in missiles[:]: # 복사본을 만들어서 반복문 돌림.
                missile.y -= missile_speed
                if missile.bottom < 0:
                    missiles.remove(missile)

            for obs in obstacles:
                if not obs["active"]:
                    # 파괴된 장애물은 respawn_at 시간이 되면 새 장애물로 부활
                    if pygame.time.get_ticks() >= obs["respawn_at"]:
                        size = random.choice(obstacle_sizes)
                        obs["base_x"] = random.randint(0, 800 - size)
                        obs["rect"].size = (size, size)
                        obs["rect"].y = 0
                        obs["rect"].x = obs["base_x"]
                        obs["phase"] = random.uniform(0, math.pi * 2)
                        obs["amplitude"] = random.randint(30, 80)
                        obs["hp"] = 3
                        obs["active"] = True
                        if obstacle_imgs:
                            chosen_img = random.choice(obstacle_imgs)
                            obs["img"] = pygame.transform.scale(chosen_img, (size, size))
                    continue # 비활성 상태인 동안은 이동/충돌 로직을 건너뜀.

                current_speed = obstacle_speed * slow_factor if is_slowed else obstacle_speed
                obs["rect"].y += current_speed # 장애물의 이동 속도가 점점 빨라짐.

                # 좌우 흔들림 계산 (이 부분이 빠져 있었음)
                t = pygame.time.get_ticks() / 500
                offset = math.sin(t + obs["phase"]) * obs["amplitude"]
                obs["rect"].x = obs["base_x"] + offset

                if obs["rect"].top > 600:
                    size = random.choice(obstacle_sizes)
                    obs["base_x"] = random.randint(0, 800 - size)
                    obs["rect"].x = obs["base_x"]
                    obs["rect"].size = (size, size)
                    obs["rect"].y = 0
                    obs["phase"] = random.uniform(0, math.pi * 2)
                    obs["amplitude"] = random.randint(30, 80)
                    if obstacle_imgs:
                        chosen_img = random.choice(obstacle_imgs)
                        obs["img"] = pygame.transform.scale(chosen_img, (size, size))

            # 미사일-장애물 충돌 체크
            for missile in missiles[:]:
                for obs in obstacles:
                    if not obs["active"]:
                        continue    
                    if missile.colliderect(obs["rect"]):
                        obs["hp"] -= 1
                        missiles.remove(missile)
                        if obs["hp"] <= 0:
                            obs["active"] = False
                            obs["respawn_at"] = pygame.time.get_ticks() + 1000 # 1초 후에 부활
                        break # 한 번 충돌하면 더 이상 체크하지 않음.        

            survival_time = (pygame.time.get_ticks() - start_time) // 1000 # 밀리초에서 초 단위로 변환.
            
            new_level = survival_time // 10
            if new_level > current_level:
                current_level = new_level
                obstacles.append(create_obstacle())
                obstacle_speed += 0.3
            # 무적상태 만료 체크(제일 먼저) & 장애물 속도 감소 상태 만료 체크
            if invincible and pygame.time.get_ticks() > invincible_until:
                invincible = False
            if is_slowed and pygame.time.get_ticks() > slow_until:
                is_slowed = False

            for obs in obstacles:
                if game_state != STATE_PLAYING:
                    break
                if player.colliderect(obs["rect"]):
                    if invincible:
                        continue
                
                    if collision_sound:
                        collision_sound.play()
                    score -= 1
                    player.x, player.y = 400, 300
                    # 충돌 후 게임을 재시작할 때, 장애물의 위치도 초기화.
                    size = random.choice(obstacle_sizes)
                    obs["base_x"] = random.randint(0, 800 - size)
                    obs["rect"].size = (size, size)
                    obs["rect"].y = 0
                    obs["rect"].x = obs["base_x"]
                    obs["phase"] = random.uniform(0, math.pi * 2)
                    obs["amplitude"] = random.randint(30, 80)
                    if obstacle_imgs:
                        chosen_img = random.choice(obstacle_imgs)
                        obs["img"] = pygame.transform.scale(chosen_img, (size, size))
                        
                    if score <= 0:
                        game_state = STATE_GAMEOVER
                        if gameover_sound:
                            gameover_sound.play()
                        if survival_time > best_time:
                            best_time = survival_time
                            save_best_time(best_time)
                            is_new_record = True
                        else:
                            is_new_record = False
                    else: # 게임오버가 아니면 3초 카운트다운
                        is_respawning = True
                        respawn_until = pygame.time.get_ticks() + 3000

            # 파워업이 없는 상태라면, 아주 낮은 확률로 하나 생성
            if powerup is None and random.random() < 0.002:
                powerup = create_powerup(random.randint(0, 800 - 30), 0)

            # 파워업이 존재하면, 낙하시킴
            if powerup is not None:
                powerup['rect'].y += obstacle_speed
                if powerup['rect'].top > 600:
                    powerup = None
            
            # 파워업 획득 체크
            if powerup is not None and player.colliderect(powerup['rect']):
                if powerup['type'] == 'invincible':
                    invincible = True
                    invincible_until = pygame.time.get_ticks() + 5000
                elif powerup['type'] == 'slow':
                    is_slowed = True
                    slow_until = pygame.time.get_ticks() + slow_duration
                elif powerup['type'] == 'life':
                    score += 1
                powerup = None

    # 사각형이 틀 밖으로 나가는 것을 방지
    if player.left < 0:
        player.left = 0
    if player.right > 800:
        player.right = 800
    if player.top < 0:
        player.top = 0
    if player.bottom > 600:
        player.bottom = 600 # 파이게임 좌표계는 y좌표가 아래로 갈수록 증가.
        
    # 화면 채우기
    if background_img:
        screen.blit(background_img, (0, 0))
    else:
        screen.fill((135, 206, 235))

    if game_state == STATE_START:
        title_surface = font.render("Space Dodger", True, (255, 255, 255))
        prompt_surface = font.render("Press SPACE to start", True, (255, 255, 255))
        screen.blit(title_surface, (300, 250))
        screen.blit(prompt_surface, (250, 300))
    elif game_state == STATE_PLAYING:
        if player_img:
            screen.blit(player_img, player)
            if invincible:
                pygame.draw.rect(screen, (0, 255, 255), player, 3)
        else:
            player_color = (0, 255, 0) if invincible else (255, 0, 0)
            pygame.draw.rect(screen, player_color, player)
        for obs in obstacles:
            if obs["img"]:
                screen.blit(obs["img"], obs["rect"])
            else:
                pygame.draw.rect(screen, (0,0,0), obs["rect"])
        for missile in missiles:
            pygame.draw.rect(screen, (255, 255, 0), missile)
        
        if powerup is not None:
            color_map = {'invincible': (255, 255, 0), 'life' : (0, 255, 0), 'slow' : (0, 150, 255)}
            if powerup_img:
                screen.blit(powerup_img, powerup['rect'])
                pygame.draw.rect(screen, color_map[powerup['type']], powerup['rect'], 3)
            else:
                pygame.draw.rect(screen, color_map[powerup['type']], powerup['rect'])

        # 그리는 순서 중요 : 순서대로 덮어씌워짐.
        if is_respawning:
            remaining_ms = respawn_until - pygame.time.get_ticks() # 목표시간까지 얼마나 남았는지
            count = remaining_ms // 1000 + 1
            count = max(1, min(3, count)) # count가 어떤 상황에서도 1, 2, 3중 하나로만 나오도록
            countdown_surface = font.render(str(count), True, (255, 255, 0))
            text_rect = countdown_surface.get_rect(center=(400, 300))
            screen.blit(countdown_surface, text_rect)

        '''text_surface = font.render("Collision!", True, (255, 255, 255))
        screen.blit(text_surface, (350, 250))
        print("Collision!")
        '''

        score_surface = font.render(f"Score : {score}", True, (255, 255, 255))
        screen.blit(score_surface, (20, 20))
        survival_surface = font.render(f"Time : {survival_time}s", True, (255, 255, 255))
        screen.blit(survival_surface, (20, 70))
        level_surface = font.render(f"Level : {current_level}", True, (255, 255, 255))
        screen.blit(level_surface, (20, 120))
        if invincible:
            invincible_surface = font.render("Invincible", True, (0, 255, 255))
            screen.blit(invincible_surface, (20, 170))
        if is_slowed:
            slow_surface = font.render('slowed obstacles', True, (0, 150, 255))
            screen.blit(slow_surface, (20, 220))

    elif game_state == STATE_GAMEOVER:
        over_surface = font.render("game over", True, (255, 0 , 0))
        screen.blit(over_surface, (300, 250))
        
        restart_surface = font.render("Press R to resart", True, (255, 0, 0))
        screen.blit(restart_surface, (250, 300))
        
        time_surface = font.render(f"Survival Time : {survival_time}s", True, (255, 0, 0))
        screen.blit(time_surface, (250, 390))

        if is_new_record:
            record_surface = font.render("New Record!", True, (255, 215, 0))
            screen.blit(record_surface, (280, 460))

    # 화면 업데이트(screen에 그린 모든 내용을 실제 모니터에 출력)
    pygame.display.flip()
    clock.tick(60) # 초당 60프레임으로 고정
pygame.quit()



