import pygame

# pygame 초기화
pygame.init()

# 화면 크기 (가로 800, 세로 600)
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("나의 첫 파이게임")

# 게임 루프 실행 여부
running = True
while running:
    # 창 닫기(x 버튼) 이벤트 감지
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # 화면을 하늘색으로 채우기
    screen.fill((135, 206, 235))

    # 화면 업데이트
    pygame.display.flip()

pygame.quit()