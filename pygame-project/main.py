import pygame
import random
import os # 파일/경로 관련 작업을 위해 필요.

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
try:
    obstacle_img = pygame.image.load("meteorBrown_big3.png").convert_alpha()
    obstacle_img = pygame.transform.scale(obstacle_img, (50, 50))
except (pygame.error, FileNotFoundError) as e:
    print(f"장애물 이미지 로드 실패: {e}")
    obstacle_img = None

try:
    powerup_img = pygame.image.load("powerupGreen_shield.png").convert_alpha()
    powerup_img = pygame.transform.scale(powerup_img, (40, 40))
except (pygame.error, FileNotFoundError) as e:
    print(f"파워업 이미지 로드 실패: {e}")
    powerup_img = None

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
    return pygame.Rect(random.randint(0, 750), random.randint(-600, 0), 50, 50)
obstacles = [create_obstacle() for _ in range(3)] # 장애물 3개


def reset_game():
    global player, score, game_state, obstacles, obstacle_speed, start_time, is_new_record, current_level
    global powerup, invincible, invincible_until
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

reset_game() # 처음 시작할 때 한 번 호출

game_state = STATE_START # 처음엔 시작 화면 상태로 시작. 즉, reset 함수 호출하고, game_state를 START로 덮어씀.

def create_powerup():
    return pygame.Rect(random.randint(0, 750), random.randint(-600, 0), 40, 40)
# 폰트 객체 생성
font = pygame.font.SysFont(None, 50) # None : pygame이 알아서 폰트 선택.
#text_surface = font.render("Collision!", True, (255, 255, 255))


# 플레이어의 이동 속도
speed = 5

clock = pygame.time.Clock() # 프레임 속도 조절용


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
        keys = pygame.key.get_pressed() # 함수 호출하고, 그 결과값을 keys에 저장.
        if keys[pygame.K_LEFT]:
            player.x -= speed
        if keys[pygame.K_RIGHT]:
            player.x += speed
        if keys[pygame.K_UP]:
            player.y -= speed
        if keys[pygame.K_DOWN]:
            player.y += speed
        
        for obs in obstacles:
            obs.y += obstacle_speed # 장애물의 이동 속도가 점점 빨라짐.
            if obs.top > 600:
                obs.y = 0
                obs.x = random.randint(0, 750)
                obstacle_speed += 0.1

        survival_time = (pygame.time.get_ticks() - start_time) // 1000 # 밀리초에서 초 단위로 변환.
        
        new_level = survival_time // 10
        if new_level > current_level:
            current_level = new_level
            obstacles.append(create_obstacle())
            obstacle_speed += 1
        # 무적상태 만료 체크(제일 먼저)
        if invincible and pygame.time.get_ticks() > invincible_until:
            invincible = False

        for obs in obstacles:
            if game_state != STATE_PLAYING:
                break
            if player.colliderect(obs):
                if invincible:
                    continue
            
                if collision_sound:
                    collision_sound.play()
                score -= 1
                player.x, player.y = 400, 300
                # 충돌 후 게임을 재시작할 때, 장애물의 위치도 초기화.
                obs.y = 0
                obs.x = random.randint(0, 750)
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

        # 파워업이 없는 상태라면, 아주 낮은 확률로 하나 생성
        if powerup is None and random.random() < 0.002:
            powerup = create_powerup()

        # 파워업이 존재하면, 낙하시킴
        if powerup is not None:
            powerup.y += obstacle_speed
            if powerup.top > 600:
                powerup = None
        
        # 파워업 획득 체크
        if powerup is not None and player.colliderect(powerup):
            invincible = True
            invincible_until = pygame.time.get_ticks() + 5000
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
        
    # 화면을 하늘색으로 채우기
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
            if obstacle_img:
                screen.blit(obstacle_img, obs)
            else:
                pygame.draw.rect(screen, (0,0,0), obs)
        if powerup is not None:
            if powerup_img:
                screen.blit(powerup_img, powerup)
            else:
                pygame.draw.rect(screen, (255, 255, 0), powerup)
        # 그리는 순서 중요 : 순서대로 덮어씌워짐.


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

    elif game_state == STATE_GAMEOVER:
        over_surface = font.render("game over", True, (255, 0 , 0))
        screen.blit(over_surface, (300, 250))
        
        restart_surface = font.render("Press R to resart", True, (0, 0, 0))
        screen.blit(restart_surface, (250, 300))
        
        time_surface = font.render(f"Survival Time : {survival_time}s", True, (0, 0, 0))
        screen.blit(time_surface, (220, 390))

        if is_new_record:
            record_surface = font.render("New Record!", True, (255, 215, 0))
            screen.blit(record_surface, (280, 460))

    # 화면 업데이트(screen에 그린 모든 내용을 실제 모니터에 출력)
    pygame.display.flip()
    clock.tick(60) # 초당 60프레임으로 고정
pygame.quit()



