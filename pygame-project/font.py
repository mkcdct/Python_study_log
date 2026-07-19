import pygame
pygame.init()

font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 50)
print(font)

test = font.render("가나다", True, (255, 255, 255))
print(test.get_size())