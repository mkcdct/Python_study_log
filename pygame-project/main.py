import pygame

# pygame 초기화
pygame.init()

# 폰트 객체 생성
font = pygame.font.SysFont(None, 50) # None : pygame이 알아서 폰트 선택.
#text_surface = font.render("Collision!", True, (255, 255, 255))
# 화면 크기 (가로 800, 세로 600)
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("나의 첫 파이게임")

#사각형(플레이어) 초기 위치와 크기
player = pygame.Rect(400, 300, 50, 50) # x, y, 너비, 높이
speed = 5
score = 3
game_over = False
# 장애물 추가
obstacle = pygame.Rect(600, 300, 50, 50)

clock = pygame.time.Clock() # 프레임 속도 조절용
# 폰트 객체 생성
font = pygame.font.SysFont(None, 50) # None : pygame이 알아서 폰트 선택.


# 게임 루프 실행 여부
running = True
while running:
    # 창 닫기(x 버튼) 이벤트 감지
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if not game_over:
    # 키 입력 감지(누르고 있는 동안 계속 반응)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player.x -= speed
        if keys[pygame.K_RIGHT]:
            player.x += speed
        if keys[pygame.K_UP]:
            player.y -= speed
        if keys[pygame.K_DOWN]:
            player.y += speed
    
    # 사각형이 틀 밖으로 나가는 것을 방지
    if player.left < 0:
        player.left = 0
    if player.right > 800:
        player.right = 800
    if player.top < 0:
        player.top = 0
    if player.bottom > 600:
        player.bottom = 600
        
    # 화면을 하늘색으로 채우기
    screen.fill((135, 206, 235))
    pygame.draw.rect(screen, (255, 0, 0), player) # 빨간 사각형 생성
    pygame.draw.rect(screen, (0, 0, 0), obstacle)

    if player.colliderect(obstacle):
        score -= 1
        player.x, player.y = 400, 300
        if score <= 0:
            game_over = True
        

        '''text_surface = font.render("Collision!", True, (255, 255, 255))
        screen.blit(text_surface, (350, 250))
        print("Collision!")
        '''

    score_surface = font.render(f"Score : {score}", True, (255, 255, 255))
    screen.blit(score_surface, (20, 20))

    if game_over:
        over_surface = font.render("game over", True, (255, 0 , 0))
        screen.blit(over_surface, (300, 250))

    # 화면 업데이트(screen에 그린 모든 내용을 실제 모니터에 출력)
    pygame.display.flip()
    clock.tick(60) # 초당 60프레임으로 고정
pygame.quit()



